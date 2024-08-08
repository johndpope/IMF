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
from helper import normalize,visualize_latent_token, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
from loss import LPIPSPerceptualLoss,VGGPerceptualLoss,wasserstein_loss,hinge_loss,vanilla_gan_loss,gan_loss_fn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from vggloss import VGGLoss
from stylegan import EMA

def load_config(config_path):
    return OmegaConf.load(config_path)


# from torch.optim.lr_scheduler import ReduceLROnPlateau
def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.initial_learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

    # dynamic learning rate
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    ema = EMA(model, decay=0.9999)

    model, discriminator, optimizer_g, optimizer_d, train_dataloader,ema = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader,ema
    )
    ema.register()
    # Use the unified gan_loss_fn
    gan_loss_type = config.loss.type
    perceptual_loss_fn = VGGPerceptualLoss().to(accelerator.device)
    # perceptual_loss_fn = LPIPSPerceptualLoss().to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()
    


    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma  # R1 regularization strength


    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        total_g_loss = 0
        total_d_loss = 0



        for batch_idx, batch in enumerate(train_dataloader):
            source_frames = batch['frames']
            batch_size, num_frames, channels, height, width = source_frames.shape

            
            for ref_idx in range(0, num_frames, config.training.every_xref_frames):  # Step by 16 for reference frames

                x_reference = source_frames[:, ref_idx]

                for current_idx in range(num_frames):
                    if current_idx == ref_idx:
                        continue  # Skip when current frame is the reference frame
                    
                    x_current = source_frames[:, current_idx]

                    # Forward pass with style mixing and noise addition
                    x_reconstructed, outputs = model(x_current, x_reference, style_mixing_prob, noise_magnitude)
                    x_reconstructed = normalize(x_reconstructed)

                    
                    # B. Loss Calculation
                    # 1. Pixel-wise Loss
                    l_p = pixel_loss_fn(x_reconstructed, x_current)

                    # 2. Perceptual Loss
                    l_v = perceptual_loss_fn(x_reconstructed, x_current)

                    # 3. GAN Loss
                    # Train Discriminator
                    optimizer_d.zero_grad()
                    
                    # R1 regularization
                    x_current.requires_grad = True
                    real_outputs = discriminator(x_current)
                    r1_reg = 0
                    for real_output in real_outputs:
                        grad_real = torch.autograd.grad(
                            outputs=real_output.sum(), inputs=x_current, create_graph=True
                        )[0]
                        r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                    
                    fake_outputs = discriminator(x_reconstructed.detach())
                    d_loss = gan_loss_fn(real_outputs, fake_outputs, gan_loss_type)

                    # Add R1 regularization to the discriminator loss
                    d_loss = d_loss + r1_gamma * r1_reg


                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    
                    accelerator.backward(d_loss)
                    optimizer_d.step()
                    ema.update()
                    # Train Generator
                    optimizer_g.zero_grad()
                    fake_outputs = discriminator(x_reconstructed)
                    g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)


                    # 4. Total Loss
                    g_loss = (config.training.lambda_pixel * l_p +
                            config.training.lambda_perceptual * l_v +
                            config.training.lambda_adv * g_loss_gan)

                    # C. Optimization
                    accelerator.backward(g_loss)
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer_g.step()


                    total_g_loss += g_loss.item()
                    total_d_loss += d_loss.item()
                    progress_bar.update(1)
                    progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})
            # Sample and save reconstructions
                sample_path = f"recon_epoch_{epoch+1}_batch_{ref_idx}.png"
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
                    "batch_g_loss": g_loss.item(),
                    "batch_d_loss": d_loss.item(),
                    "pixel_loss": l_p.item(),
                    "perceptual_loss": l_v.item(),
                    "gan_loss": g_loss_gan.item(),
                    "batch": batch_idx + epoch * len(train_dataloader),
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # dataset = EMODataset(
    #     use_gpu=True,
    #     remove_background=False,
    #     width=256,
    #     height=256,
    #     sample_rate=24,
    #     img_scale=(1.0, 1.0),
    #     video_dir=config.dataset.root_dir,
    #     json_file=config.dataset.json_file,
    #     transform=transform,
    #     apply_crop_warping=False
    # )


    dataset = VideoDataset("/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images", 
                           transform=transform, 
                           frame_skip=0, 
                           num_frames=240)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=1,
        # persistent_workers=True,
        # pin_memory=True,
        collate_fn=gpu_padded_collate 
    )

    train(config, model, discriminator, dataloader, accelerator)

if __name__ == "__main__":
    main()