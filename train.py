import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import yaml
import os
import torch.nn.functional as F
from model import IMFModel, debug_print,PatchDiscriminator
from VideoDataset import VideoDataset
from EMODataset import EMODataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import monitor_gradients, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
from loss import LPIPSPerceptualLoss,VGGPerceptualLoss,wasserstein_loss,hinge_loss,vanilla_gan_loss,gan_loss_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from modules.model import Vgg19, ImagePyramide, Transform

def load_config(config_path):
    return OmegaConf.load(config_path)

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()

    def forward(self, x_hat, x):
        """
        Compute the pixel-wise L1 loss between the synthesized frame and the current frame.
        
        Args:
        x_hat (torch.Tensor): The synthesized frame
        x (torch.Tensor): The current frame
        
        Returns:
        torch.Tensor: The computed pixel-wise loss
        """
        return torch.mean(torch.abs(x_hat - x))


def multiscale_discriminator_loss(real_preds, fake_preds, loss_type='lsgan'):
    if loss_type == 'lsgan':
        real_loss = sum(torch.mean((real_pred - 1)**2) for real_pred in real_preds)
        fake_loss = sum(torch.mean(fake_pred**2) for fake_pred in fake_preds)
    elif loss_type == 'vanilla':
        real_loss = sum(F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) for real_pred in real_preds)
        fake_loss = sum(F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred)) for fake_pred in fake_preds)
    else:
        raise NotImplementedError(f'Loss type {loss_type} is not implemented.')
    
    return ((real_loss + fake_loss) * 0.5).requires_grad_()

# from torch.optim.lr_scheduler import ReduceLROnPlateau

def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.initial_learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    pixel_loss_fn = PixelLoss()
    vgg = Vgg19()
    pyramid = ImagePyramide(config.training.scales, 3)


    # EMA setup
    ema = EMA(config.training.ema_decay)
    ema_model = IMFModel(latent_dim=config.model.latent_dim, base_channels=config.model.base_channels, num_layers=config.model.num_layers)
    ema_model.load_state_dict(model.state_dict())



    model, discriminator, optimizer_g, optimizer_d, train_dataloader, pixel_loss_fn, vgg, pyramid,ema_model,ema = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader, pixel_loss_fn, vgg, pyramid,ema_model,ema
    )

    # Set noise level and style mixing probability
    model.set_noise_level(config.training.noise_magnitude)
    model.set_style_mix_prob(config.training.style_mixing_prob)

    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        total_g_loss = 0
        total_d_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            source_frames = batch['frames']
            batch_size, num_frames, channels, height, width = source_frames.shape

            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # Gradient accumulation setup
            accumulation_steps = config.training.gradient_accumulation_steps
            effective_batch_size = batch_size * accumulation_steps

            for ref_idx in range(0, num_frames, config.training.every_xref_frames):
                x_reference = source_frames[:, ref_idx]

                for current_idx in range(num_frames):
                    if current_idx == ref_idx:
                        continue
                    
                    x_current = source_frames[:, current_idx]

                    x_reconstructed, diagnostics = model(x_current, x_reference)

                    l_p = pixel_loss_fn(x_reconstructed, x_current)

                    pyramide_real = pyramid(x_current)
                    pyramide_generated = pyramid(x_reconstructed)
                    l_v = 0
                    for scale in config.training.scales:
                        x_vgg = vgg(pyramide_generated[f'prediction_{scale}'])
                        y_vgg = vgg(pyramide_real[f'prediction_{scale}'])
                        for i, weight in enumerate(config.loss.weights['perceptual']):
                            l_v += weight * torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()

                    # Train Discriminator
                    x_current.requires_grad = True
                    real_outputs = discriminator(x_current)
                    fake_outputs = discriminator(x_reconstructed.detach())
                    d_loss = multiscale_discriminator_loss(real_outputs, fake_outputs, loss_type='lsgan')

                    # Conditional application of R1 regularization
                    if config.training.use_r1_reg and batch_idx % config.training.r1_interval == 0:
                        r1_reg = 0
                        for real_output in real_outputs:
                            grad_real = torch.autograd.grad(
                                outputs=real_output.sum(), inputs=x_current, create_graph=True
                            )[0]
                            r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                        d_loss = d_loss + config.training.r1_gamma * r1_reg

                    d_loss = d_loss / accumulation_steps
                    accelerator.backward(d_loss)

                    # Train Generator
                    fake_outputs = discriminator(x_reconstructed)
                    g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

                    g_loss = (config.training.lambda_pixel * l_p +
                              config.training.lambda_perceptual * l_v +
                              config.training.lambda_adv * g_loss_gan)

                    g_loss = g_loss / accumulation_steps
                    accelerator.backward(g_loss)

                    total_g_loss += g_loss.item() * accumulation_steps
                    total_d_loss += d_loss.item() * accumulation_steps

                    if (batch_idx + 1) % accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizer_g.step()
                        optimizer_d.step()
                        optimizer_g.zero_grad()
                        optimizer_d.zero_grad()

                        # Update EMA model
                        ema.update_model_average(ema_model, model)

                    progress_bar.update(1)
                    progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})

                # Sample and save reconstructions
                if batch_idx % config.training.save_steps == 0:
                    sample_path = f"recon_epoch_{epoch+1}_batch_{batch_idx}.png"
                    sample_recon(model, (x_reconstructed, x_reference), accelerator, sample_path, 
                                num_samples=config.logging.sample_size)

         # Calculate average losses for the epoch
        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
        
        # Step the schedulers
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)
        
        # Logging
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "avg_g_loss": avg_g_loss,
                "avg_d_loss": avg_d_loss,
                "lr_g": optimizer_g.param_groups[0]['lr'],
                "lr_d": optimizer_d.param_groups[0]['lr']
            })

        progress_bar.close()

        # Checkpoint saving
        if (epoch + 1) % config.checkpoints.interval == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'ema_model_state_dict': ema_model.state_dict(),
                'discriminator_state_dict': unwrapped_discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, f"{config.checkpoints.dir}/checkpoint_{epoch+1}.pth")

    # Final model saving
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f"{config.checkpoints.dir}/final_model.pth")
    
def main():
    config = load_config('config.yaml')
    torch.cuda.empty_cache()
    wandb.init(project='IMF', config=OmegaConf.to_container(config, resolve=True))

    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        cpu=config.accelerator.cpu
    )

    model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers
    )
    add_gradient_hooks(model)

    discriminator = PatchDiscriminator(ndf=config.discriminator.ndf)
    add_gradient_hooks(discriminator)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VideoDataset(
        root_dir=config.dataset.root_dir,
        transform=transform,
        frame_skip=config.dataset.frame_skip
    )

    dataset = EMODataset(
        use_gpu=True,
        remove_background=True,
        width=256,
        height=256,
        sample_rate=24,
        img_scale=(1.0, 1.0),
        video_dir=config.dataset.root_dir,
        json_file=config.dataset.json_file,
        transform=transform,
        apply_crop_warping=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=True,
        collate_fn=gpu_padded_collate 
    )

    train(config, model, discriminator, dataloader, accelerator)

if __name__ == "__main__":
    main()