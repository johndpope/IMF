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
from vggloss import VGGLoss

def load_config(config_path):
    return OmegaConf.load(config_path)



def train_no_gan(config, model, train_dataloader, accelerator):
    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-4)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Set up exponential moving average of model weights
    ema_model = accelerator.unwrap_model(model).to(accelerator.device)
    ema_model.load_state_dict(accelerator.unwrap_model(model).state_dict())

    mse_loss = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='alex').to(accelerator.device)

    os.makedirs(config.checkpoints.dir, exist_ok=True)

    start_epoch = 0
    # Checkpoint restoration logic here (omitted for brevity)

    for epoch in range(start_epoch, config.training.num_epochs):
        model.train()
        total_mse_loss = 0
        total_perceptual_loss = 0
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}", 
                            disable=not accelerator.is_local_main_process)
        
        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            # Forward pass
            reconstructed_frames = model(current_frames, reference_frames)

            # Add noise to latent tokens for improved training dynamics
            tc = model.latent_token_encoder(current_frames)
            tr = model.latent_token_encoder(reference_frames)

            noise = torch.randn_like(tc) * config.training.noise_magnitude
            tc = tc + noise
            tr = tr + noise

            # Perform style mixing regularization
            if torch.rand(()).item() < config.training.style_mixing_prob:
                rand_tc = tc[torch.randperm(tc.size(0))]
                rand_tr = tr[torch.randperm(tr.size(0))]

                mix_tc = [rand_tc if torch.rand(()).item() < 0.5 else tc for _ in range(len(model.imf.implicit_motion_alignment))]
                mix_tr = [rand_tr if torch.rand(()).item() < 0.5 else tr for _ in range(len(model.imf.implicit_motion_alignment))]
            else:
                mix_tc = [tc] * len(model.imf.implicit_motion_alignment)
                mix_tr = [tr] * len(model.imf.implicit_motion_alignment)

            m_c, m_r = model.imf.process_tokens(mix_tc, mix_tr)

            fr = model.imf.dense_feature_encoder(reference_frames)

            aligned_features = []
            for i in range(len(model.imf.implicit_motion_alignment)):
                f_r_i = fr[i]
                align_layer = model.imf.implicit_motion_alignment[i]
                m_c_i = m_c[i][i]
                m_r_i = m_r[i][i]
                aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
                aligned_features.append(aligned_feature)

            reconstructed_frames = model.frame_decoder(aligned_features)
            mse = mse_loss(reconstructed_frames, current_frames)
            perceptual = lpips_loss(reconstructed_frames, current_frames).mean()
            loss = mse + config.training.perceptual_weight * perceptual

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Update exponential moving average of weights
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), accelerator.unwrap_model(model).parameters()):
                    p_ema.copy_(p.lerp(p_ema, config.training.ema_decay))

            total_mse_loss += mse.item()
            total_perceptual_loss += perceptual.item()
            total_loss += loss.item()
            
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            wandb.log({
                "batch_mse_loss": mse.item(),
                "batch_perceptual_loss": perceptual.item(),
                "batch_total_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader)
            })

        progress_bar.close()
        avg_mse_loss = total_mse_loss / len(train_dataloader)
        avg_perceptual_loss = total_perceptual_loss / len(train_dataloader)
        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                          f"MSE: {avg_mse_loss:.4f}, "
                          f"LPIPS: {avg_perceptual_loss:.4f}, "
                          f"Avg: {avg_loss:.4f}")
        wandb.log({
            "epoch": epoch + 1,
            "avg_mse_loss": avg_mse_loss,
            "avg_perceptual_loss": avg_perceptual_loss,
            "avg_total_loss": avg_loss,
        })

        # Checkpoint saving
        if (epoch + 1) % config.checkpoints.interval == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{config.checkpoints.dir}/checkpoint_{epoch+1}.pth")

        if epoch % config.logging.sample_interval == 0:
            sample_path = f"recon_epoch_{epoch+1}.png"
            sample_recon(ema_model, next(iter(train_dataloader)), accelerator, sample_path, 
                         num_samples=config.logging.sample_size)
            wandb.log({"sample_reconstruction": wandb.Image(sample_path)})

    return ema_model


# from torch.optim.lr_scheduler import ReduceLROnPlateau
def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.initial_learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

    # dynamic learning rate
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )
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

                    # A. Forward Pass
                    # 1. Dense Feature Encoding
                    f_r = model.dense_feature_encoder(x_reference)

                    # 2. Latent Token Encoding (with noise addition)
                    t_r = model.latent_token_encoder(x_reference)
                    t_c = model.latent_token_encoder(x_current)

                    # Add noise to latent tokens
                    noise_r = torch.randn_like(t_r) * noise_magnitude
                    noise_c = torch.randn_like(t_c) * noise_magnitude
                    t_r = t_r + noise_r
                    t_c = t_c + noise_c

                    # Style mixing (optional, based on probability)
                    if torch.rand(()).item() < style_mixing_prob:
                        rand_t_c = t_c[torch.randperm(t_c.size(0))]
                        rand_t_r = t_r[torch.randperm(t_r.size(0))]
                        mix_t_c = [rand_t_c if torch.rand(()).item() < 0.5 else t_c for _ in range(len(model.implicit_motion_alignment))]
                        mix_t_r = [rand_t_r if torch.rand(()).item() < 0.5 else t_r for _ in range(len(model.implicit_motion_alignment))]
                    else:
                        mix_t_c = [t_c] * len(model.implicit_motion_alignment)
                        mix_t_r = [t_r] * len(model.implicit_motion_alignment)

                    # 3. Latent Token Decoding
                    m_c, m_r = model.process_tokens(mix_t_c, mix_t_r)

                    # 4. Implicit Motion Alignment
                    # Implicit Motion Alignment
                    aligned_features = []
                    for i in range(len(model.implicit_motion_alignment)):
                        f_r_i = f_r[i]
                        align_layer = model.implicit_motion_alignment[i]
                        m_c_i = m_c[i][i]
                        m_r_i = m_r[i][i]
                        aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
                        aligned_features.append(aligned_feature)


                    # 5. Frame Decoding
                    x_reconstructed = model.frame_decoder(aligned_features)

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

    # dataset = VideoDataset(
    #     root_dir=config.dataset.root_dir,
    #     transform=transform
    # )

    dataset = EMODataset(
        use_gpu=True,
        remove_background=False,
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
        num_workers=1,
        # persistent_workers=True,
        # pin_memory=True,
        collate_fn=gpu_padded_collate 
    )

    train_no_gan(config, model, discriminator, dataloader, accelerator)

if __name__ == "__main__":
    main()