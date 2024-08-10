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

def load_config(config_path):
    return OmegaConf.load(config_path)



def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

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
    perceptual_loss_fn = VGGPerceptualLoss().to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()

    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma if config.training.use_r1_reg else 0

    global_step = 0

    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        total_g_loss = 0
        total_d_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            source_frames = batch['frames']
            batch_size, num_frames, channels, height, width = source_frames.shape

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

                    # Forward pass and loss calculation
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

                    l_p = pixel_loss_fn(x_reconstructed, x_current)
                    l_v = perceptual_loss_fn(x_reconstructed, x_current)

                    # Train Discriminator
                    if batch_idx % config.training.gradient_accumulation_steps == 0:
                        optimizer_d.zero_grad()

                    if config.training.use_r1_reg:
                        x_current.requires_grad = True
                    real_outputs = discriminator(x_current)
                    r1_reg = 0
                    if config.training.use_r1_reg:
                        for real_output in real_outputs:
                            grad_real = torch.autograd.grad(
                                outputs=real_output.sum(), inputs=x_current, create_graph=True
                            )[0]
                            r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                    
                    fake_outputs = discriminator(x_reconstructed.detach())
                    d_loss = gan_loss_fn(real_outputs, fake_outputs, gan_loss_type)
                    d_loss = d_loss + r1_gamma * r1_reg

                    d_loss = d_loss / config.training.gradient_accumulation_steps
                    accelerator.backward(d_loss)

                    if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        optimizer_d.step()
                        if ema:
                            ema.update()

                    # Train Generator
                    if batch_idx % config.training.gradient_accumulation_steps == 0:
                        optimizer_g.zero_grad()

                    fake_outputs = discriminator(x_reconstructed)
                    g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

                    g_loss = (config.training.lambda_pixel * l_p +
                              config.training.lambda_perceptual * l_v +
                              config.training.lambda_adv * g_loss_gan)

                    g_loss = g_loss / config.training.gradient_accumulation_steps
                    accelerator.backward(g_loss)

                    if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer_g.step()

                    total_g_loss += g_loss.item() * config.training.gradient_accumulation_steps
                    total_d_loss += d_loss.item() * config.training.gradient_accumulation_steps
                    progress_bar.update(1)
                    progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})
                
                    global_step += 1
                
                    if global_step % config.training.save_steps == 0:
                        sample_path = f"recon_step_{global_step}.png"
                        sample_recon(model, (x_reconstructed, x_reference), accelerator, sample_path, 
                                    num_samples=config.logging.sample_size)

        progress_bar.close()

        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)

        if accelerator.is_main_process:
            wandb.log({
                "epoch_g_loss": avg_g_loss,
                "epoch_d_loss": avg_d_loss,
                "epoch": epoch,
                "lr_g": optimizer_g.param_groups[0]['lr'],
                "lr_d": optimizer_d.param_groups[0]['lr']
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

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), f"{config.checkpoints.dir}/final_model.pth")

# The main function remains the same
      
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