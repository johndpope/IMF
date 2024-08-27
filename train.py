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
from model import IMFModel, debug_print,PatchDiscriminator,MultiScalePatchDiscriminator
from VideoDataset import VideoDataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
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
from torch.optim import AdamW, SGD
from transformers import Adafactor

def load_config(config_path):
    return OmegaConf.load(config_path)

def get_video_repeat(epoch, max_epochs, initial_repeat, final_repeat):
    return max(final_repeat, initial_repeat - (initial_repeat - final_repeat) * (epoch / max_epochs))

def get_ema_decay(epoch, max_epochs, initial_decay=0.95, final_decay=0.9999):
    return min(final_decay, initial_decay + (final_decay - initial_decay) * (epoch / max_epochs))

def get_noise_magnitude(epoch, max_epochs, initial_magnitude=0.1, final_magnitude=0.001):
    """
    Calculate the noise magnitude for the current epoch.
    
    Args:
    epoch (int): Current epoch number
    max_epochs (int): Total number of epochs
    initial_magnitude (float): Starting noise magnitude
    final_magnitude (float): Ending noise magnitude
    
    Returns:
    float: Calculated noise magnitude for the current epoch
    """
    return max(final_magnitude, initial_magnitude - (initial_magnitude - final_magnitude) * (epoch / max_epochs))


def get_layer_wise_learning_rates(model):
    params = []
    params.append({'params': model.dense_feature_encoder.parameters(), 'lr': 1e-4})
    params.append({'params': model.latent_token_encoder.parameters(), 'lr': 5e-5})
    params.append({'params': model.latent_token_decoder.parameters(), 'lr': 5e-5})
    params.append({'params': model.implicit_motion_alignment.parameters(), 'lr': 1e-4})
    params.append({'params': model.frame_decoder.parameters(), 'lr': 2e-4})
    params.append({'params': model.mapping_network.parameters(), 'lr': 1e-4})
    return params


def train(config, model, discriminator, train_dataloader, accelerator):
    layer_wise_params = get_layer_wise_learning_rates(model)
    optimizer_g = AdamW(layer_wise_params, lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    if config.training.use_ema:
        ema = EMA(model, decay=config.training.ema_decay)
    else:
        ema = None

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )
    if ema:
        ema = accelerator.prepare(ema)
        ema.register()

    gan_loss_type = config.loss.type
    perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()

    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma

    global_step = 0

    for epoch in range(config.training.num_epochs):
        video_repeat = get_video_repeat(epoch, config.training.num_epochs, 
                                        config.training.initial_video_repeat, 
                                        config.training.final_video_repeat)

        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        epoch_g_loss = 0
        epoch_d_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            source_frames = batch['frames']
            batch_size, num_frames, channels, height, width = source_frames.shape

            for _ in range(int(video_repeat)):
                if config.training.use_many_xrefs:
                    ref_indices = range(0, num_frames, config.training.every_xref_frames)
                else:
                    ref_indices = [0]

                for ref_idx in ref_indices:
                    x_reference = source_frames[:, ref_idx]

                    for current_idx in range(num_frames):
                        if current_idx == ref_idx:
                            continue

                        x_current = source_frames[:, current_idx]

                        # Forward Pass
                        f_r = model.dense_feature_encoder(x_reference)
                        t_r = model.latent_token_encoder(x_reference)
                        t_c = model.latent_token_encoder(x_current)

                        noise_r = torch.randn_like(t_r) * noise_magnitude
                        noise_c = torch.randn_like(t_c) * noise_magnitude
                        t_r = t_r + noise_r
                        t_c = t_c + noise_c

                        if torch.rand(()).item() < style_mixing_prob:
                            batch_size = t_c.size(0)
                            rand_indices = torch.randperm(batch_size)
                            rand_t_c = t_c[rand_indices]
                            rand_t_r = t_r[rand_indices]
                            mix_mask = torch.rand(batch_size, 1, device=t_c.device) < 0.5
                            mix_mask = mix_mask.float()
                            mix_t_c = t_c * mix_mask + rand_t_c * (1 - mix_mask)
                            mix_t_r = t_r * mix_mask + rand_t_r * (1 - mix_mask)
                        else:
                            mix_t_c = t_c
                            mix_t_r = t_r

                        m_c = model.latent_token_decoder(mix_t_c)
                        m_r = model.latent_token_decoder(mix_t_r)

                        aligned_features = []
                        for i in range(len(model.implicit_motion_alignment)):
                            f_r_i = f_r[i]
                            align_layer = model.implicit_motion_alignment[i]
                            m_c_i = m_c[i] 
                            m_r_i = m_r[i]
                            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
                            aligned_features.append(aligned_feature)

                        x_reconstructed = model.frame_decoder(aligned_features)
                        x_reconstructed = normalize(x_reconstructed)

                        # Loss Calculation and Optimization
                        # Discriminator
                        optimizer_d.zero_grad()
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
                        d_loss = d_loss + r1_gamma * r1_reg

                        accelerator.backward(d_loss)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizer_d.step()

                        # Generator
                        optimizer_g.zero_grad()
                        fake_outputs = discriminator(x_reconstructed)
                        g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

                        l_p = pixel_loss_fn(x_reconstructed, x_current).mean()
                        l_v = perceptual_loss_fn(x_reconstructed, x_current).mean()

                        g_loss = (config.training.lambda_pixel * l_p +
                                  config.training.lambda_perceptual * l_v +
                                  config.training.lambda_adv * g_loss_gan)

                        accelerator.backward(g_loss)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer_g.step()

                        if ema:
                            ema.update()

                        epoch_g_loss += g_loss.item()
                        epoch_d_loss += d_loss.item()

                        # Logging
                        if accelerator.is_main_process and global_step % config.logging.log_every == 0:
                            wandb.log({
                                "noise_magnitude": noise_magnitude,
                                "g_loss": g_loss.item(),
                                "d_loss": d_loss.item(),
                                "pixel_loss": l_p.item(),
                                "perceptual_loss": l_v.item(),
                                "gan_loss": g_loss_gan.item(),
                                "global_step": global_step,
                                "lr_g": optimizer_g.param_groups[0]['lr'],
                                "lr_d": optimizer_d.param_groups[0]['lr']
                            })

                        if global_step % config.logging.sample_every == 0:
                            sample_path = f"recon_step_{global_step}.png"
                            sample_recon(model, (x_reconstructed, x_current, x_reference), accelerator, sample_path, 
                                         num_samples=config.logging.sample_size)

                        global_step += 1

                        progress_bar.update(1)
                        progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})
                        # Free up memory
                        del g_loss, d_loss, l_p, l_v, g_loss_gan
                        torch.cuda.empty_cache()

      

        progress_bar.close()

        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / (len(train_dataloader) * num_frames * len(ref_indices))
        avg_d_loss = epoch_d_loss / (len(train_dataloader) * num_frames * len(ref_indices))

        # Step the schedulers
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)

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
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature
        
    )

    discriminator = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDataset("/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images", 
                           transform=transform, 
                           frame_skip=0, 
                           num_frames=240)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=1,
        shuffle=True,
        # persistent_workers=True,
        pin_memory=True,
        collate_fn=gpu_padded_collate 
    )

   # Count parameters for both models
    imf_params, imf_breakdown = count_model_params(model, verbose=False)
    disc_params, disc_breakdown = count_model_params(discriminator, verbose=False)

    accelerator.print("🎯 Model parameters:")
    accelerator.print(f"   IMFModel: {imf_params:.2f}M")
    accelerator.print(f"   Discriminator: {disc_params:.2f}M")
 
    if config.logging.print_model_details:
        accelerator.print("\nIMFModel parameter breakdown:")
        for layer_type, count in sorted(imf_breakdown.items(), key=lambda x: x[1], reverse=True):
            accelerator.print(f"{layer_type:<20} {count:,}")
        
        accelerator.print("\nDiscriminator parameter breakdown:")
        for layer_type, count in sorted(disc_breakdown.items(), key=lambda x: x[1], reverse=True):
            accelerator.print(f"{layer_type:<20} {count:,}")


    train(config, model, discriminator, dataloader, accelerator)

if __name__ == "__main__":
    main()