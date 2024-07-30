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
from model import IMFModel, PatchDiscriminator, init_weights, debug_print
from VideoDataset import VideoDataset
from torchvision.utils import save_image
from helper import monitor_gradients, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips

def load_config(config_path):
    return OmegaConf.load(config_path)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features[:36].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.vgg(x)

def pixel_loss(x_hat, x):
    return nn.L1Loss()(x_hat, x)

def perceptual_loss(vgg, x_hat, x):
    losses = []
    for i in range(4):
        x_hat_features = vgg(x_hat[:, :, i::2, i::2])
        x_features = vgg(x[:, :, i::2, i::2])
        losses.append(F.l1_loss(x_hat_features, x_features))
    return sum(losses)

def adversarial_loss(discriminator, x_hat):
    fake_outputs = discriminator(x_hat)
    return sum(-torch.mean(output) for output in fake_outputs)

def hinge_loss(real_outputs, fake_outputs):
    real_loss = sum(torch.mean(F.relu(1 - out)) for out in real_outputs)
    fake_loss = sum(torch.mean(F.relu(1 + out)) for out in fake_outputs)
    return real_loss + fake_loss

def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.learning_rate_d, betas=(0.5, 0.999))

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )

    vgg_loss = VGGLoss().to(accelerator.device)
    lpips_loss = lpips.LPIPS(net='alex').to(accelerator.device)

    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude

    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            # Train Discriminator
            for _ in range(config.training.n_critic):
                optimizer_d.zero_grad()

                with torch.no_grad():
                    reconstructed_frames = model(current_frames, reference_frames)

                real_outputs = discriminator(current_frames)
                fake_outputs = discriminator(reconstructed_frames.detach())

                d_loss = hinge_loss(real_outputs, fake_outputs)

                accelerator.backward(d_loss)
                optimizer_d.step()

                total_d_loss += d_loss.item()

            # Train Generator
            optimizer_g.zero_grad()

            # Style mixing and noise injection
            tc = model.latent_token_encoder(current_frames)
            tr = model.latent_token_encoder(reference_frames)

            noise = torch.randn_like(tc) * noise_magnitude
            tc = tc + noise
            tr = tr + noise

            if torch.rand(()).item() < style_mixing_prob:
                rand_tc = tc[torch.randperm(tc.size(0))]
                rand_tr = tr[torch.randperm(tr.size(0))]

                mix_tc = [rand_tc if torch.rand(()).item() < 0.5 else tc for _ in range(len(model.implicit_motion_alignment))]
                mix_tr = [rand_tr if torch.rand(()).item() < 0.5 else tr for _ in range(len(model.implicit_motion_alignment))]
            else:
                mix_tc = [tc] * len(model.implicit_motion_alignment)
                mix_tr = [tr] * len(model.implicit_motion_alignment)

            reconstructed_frames = model(current_frames, reference_frames, mix_tc, mix_tr)

            pixel_l = pixel_loss(reconstructed_frames, current_frames)
            perceptual_l = perceptual_loss(vgg_loss, reconstructed_frames, current_frames)
            adv_l = adversarial_loss(discriminator, reconstructed_frames)

            g_loss = (config.training.lambda_pixel * pixel_l +
                      config.training.lambda_perceptual * perceptual_l +
                      config.training.lambda_adv * adv_l)

            accelerator.backward(g_loss)
            optimizer_g.step()

            total_g_loss += g_loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})

            wandb.log({
                "batch_g_loss": g_loss.item(),
                "batch_d_loss": d_loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader)
            })

        progress_bar.close()

        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)

        if accelerator.is_main_process:
            accelerator.print(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                              f"Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

            wandb.log({
                "epoch_g_loss": avg_g_loss,
                "epoch_d_loss": avg_d_loss,
                "epoch": epoch
            })

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

        if epoch % config.logging.sample_interval == 0:
            sample_path = f"recon_epoch_{epoch+1}.png"
            sample_frames = sample_recon(model, next(iter(train_dataloader)), accelerator, sample_path, 
                                        num_samples=config.logging.sample_size)
            
            wandb.log({"sample_reconstruction": wandb.Image(sample_path)})

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
    discriminator = PatchDiscriminator(ndf=config.discriminator.ndf)

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
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=True
    )

    train(config, model, discriminator, dataloader, accelerator)

if __name__ == "__main__":
    main()