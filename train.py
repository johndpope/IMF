import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
import yaml
import os
import torch.nn.functional as F
from model import IMFModel, debug_print,MultiScalePatchDiscriminator
from VideoDataset import VideoDataset
from EMODataset import EMODataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import log_loss_landscape,log_grad_flow,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
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



def train(config, model, discriminator, train_dataloader, val_loader, accelerator):
    # optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_g = AdamW(model.parameters(), lr=config.training.learning_rate_g, weight_decay=0.01)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.initial_learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

    # dynamic learning rate
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.25, patience=10, verbose=True)
    scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.25, patience=10, verbose=True)


    # Make EMA conditional based on config
    if config.training.use_ema:
        ema = EMA(model, decay=config.training.ema_decay)
    else:
        ema = None

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )
    # Prepare EMA if it's being used
    if ema:
        ema = accelerator.prepare(ema)
        ema.register()

    # Use the unified gan_loss_fn
    gan_loss_type = config.loss.type
    # perceptual_loss_fn = VGGPerceptualLoss().to(accelerator.device)
    # perceptual_loss_fn = LPIPSPerceptualLoss().to(accelerator.device)
    perceptual_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()
    


    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma  # R1 regularization strength


    global_step = 0

    for epoch in range(config.training.num_epochs):
        video_repeat = get_video_repeat(epoch, config.training.num_epochs, 
                                        config.training.initial_video_repeat, 
                                        config.training.final_video_repeat)
        
        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        total_g_loss = 0
        total_d_loss = 0
         
        current_decay = get_ema_decay(epoch, config.training.num_epochs)
        ema.decay = current_decay

        for batch_idx, batch in enumerate(train_dataloader):
            # Repeat the current video for the specified number of times
            for _ in range(int(video_repeat)):
                source_frames = batch['frames']
                batch_size, num_frames, channels, height, width = source_frames.shape

                ref_idx = 0

                # Calculate noise magnitude for this epoch
                noise_magnitude = get_noise_magnitude(
                    epoch, 
                    config.training.num_epochs, 
                    initial_magnitude=config.training.initial_noise_magnitude,
                    final_magnitude=config.training.final_noise_magnitude
                )
                if config.training.use_many_xrefs:
                    ref_indices = range(0, num_frames, config.training.every_xref_frames)
                else:
                    ref_indices = [0]  # Only use the first frame as reference

                for ref_idx in ref_indices:
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
                        # Style mixing (optional, based on probability)
                        # print(f"Original t_c shape: {t_c.shape}")
                        # print(f"Original t_r shape: {t_r.shape}")

                        if torch.rand(()).item() < style_mixing_prob:
                            batch_size = t_c.size(0)
                            rand_indices = torch.randperm(batch_size)
                            rand_t_c = t_c[rand_indices]
                            rand_t_r = t_r[rand_indices]
                            
                            # print(f"rand_t_c shape: {rand_t_c.shape}")
                            # print(f"rand_t_r shape: {rand_t_r.shape}")
                            
                            # Create a mask for mixing
                            mix_mask = torch.rand(batch_size, 1, device=t_c.device) < 0.5
                            mix_mask = mix_mask.float()
                            
                            # print(f"mix_mask shape: {mix_mask.shape}")
                            
                            # Mix the tokens
                            mix_t_c = t_c * mix_mask + rand_t_c * (1 - mix_mask)
                            mix_t_r = t_r * mix_mask + rand_t_r * (1 - mix_mask)
                        else:
                            # print(f"no mixing...")
                            mix_t_c = t_c
                            mix_t_r = t_r

                        # print(f"Final mix_t_c shape: {mix_t_c.shape}")
                        # print(f"Final mix_t_r shape: {mix_t_r.shape}")

                        # Now use mix_t_c and mix_t_r for the rest of the processing
                        m_c = model.latent_token_decoder(mix_t_c)
                        m_r = model.latent_token_decoder(mix_t_r)


                        # Visualize latent tokens (do this every N batches to avoid overwhelming I/O)
                        # if batch_idx % config.logging.visualize_every == 0:
                        #     os.makedirs(f"latent_visualizations/epoch_{epoch}", exist_ok=True)
                        #     visualize_latent_token(
                        #         t_r,  # Visualize the first token in the batch
                        #         f"latent_visualizations/epoch_{epoch}/t_r_token_reference_batch{batch_idx}.png"
                        #     )
                        #     visualize_latent_token(
                        #         m_c[0],  # Visualize the first token in the batch
                        #         f"latent_visualizations/epoch_{epoch}/m_c_token_current_batch{batch_idx}.png"
                        #     )


                        # 4. Implicit Motion Alignment
                        # Implicit Motion Alignment
                        aligned_features = []
                        for i in range(len(model.implicit_motion_alignment)):
                            f_r_i = f_r[i]
                            align_layer = model.implicit_motion_alignment[i]
                            m_c_i = m_c[i] 
                            m_r_i = m_r[i]
                            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
                            aligned_features.append(aligned_feature)


                        # 5. Frame Decoding
                        x_reconstructed = model.frame_decoder(aligned_features)
                        x_reconstructed = normalize(x_reconstructed) # 🤷 images are washed out - or over saturated...

                        # B. Loss Calculation
                        # 1. Pixel-wise Loss
                        l_p = pixel_loss_fn(x_reconstructed, x_current).mean()

                        # 2. Perceptual Loss
                        l_v = perceptual_loss_fn(x_reconstructed, x_current).mean()


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
                        # Update EMA if it's being used
                        if ema:
                            ema.update()
                        # Train Generator
                        optimizer_g.zero_grad()
                        fake_outputs = discriminator(x_reconstructed)
                        g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)


                        # 4. Total Loss
                        g_loss = (config.training.lambda_pixel * l_p.mean() +
                            config.training.lambda_perceptual * l_v.mean() +
                            config.training.lambda_adv * g_loss_gan.mean())

                        # C. Optimization
                        accelerator.backward(g_loss)
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer_g.step()


                        total_g_loss += g_loss.item()
                        total_d_loss += d_loss.item()
                        progress_bar.update(1)
                        progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})
                    
                        # Update global step
                        global_step += 1
                    
                    # Sample and save reconstructions every save_steps
                    sample_path = f"recon_step_{global_step}.png"
                    sample_recon(model, (x_reconstructed, x_current,x_reference), accelerator, sample_path, 
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
                            "ema":current_decay,
                            "noise_magnitude": noise_magnitude,
                            "batch_g_loss": g_loss.item(),
                            "batch_d_loss": d_loss.item(),
                            "pixel_loss": l_p.item(),
                            "perceptual_loss": l_v.item(),
                            "gan_loss": g_loss_gan.item(),
                            "batch": batch_idx + epoch * len(train_dataloader),
                            "lr_g": optimizer_g.param_groups[0]['lr'],
                            "lr_d": optimizer_d.param_groups[0]['lr']
                        })

                    # Log gradient flow for generator and discriminator
                    criterion = [perceptual_loss_fn,pixel_loss_fn]
                    if accelerator.is_main_process and batch_idx % config.logging.log_every == 0:
                        log_grad_flow(model.named_parameters(),global_step),
                        log_grad_flow(discriminator.named_parameters(),global_step)
                        # log_loss_landscape(model, criterion, val_loader, global_step)



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
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature,
        use_mlgffn=config.model.use_mlgffn
        
    )
    add_gradient_hooks(model)

    # discriminator = PatchDiscriminator(ndf=config.discriminator.ndf) Original
    discriminator = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)

    add_gradient_hooks(discriminator)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    full_dataset = VideoDataset("/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/images", 
                                transform=transform, 
                                frame_skip=0, 
                                num_frames=240)
    
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])




    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        collate_fn=gpu_padded_collate 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        collate_fn=gpu_padded_collate 
    )

   # Count parameters for both models
    imf_params, imf_breakdown = count_model_params(model, verbose=config.logging.print_model_details)
    disc_params, disc_breakdown = count_model_params(discriminator, verbose=config.logging.print_model_details)




    train(config, model, discriminator, train_dataloader, val_loader,  accelerator)

if __name__ == "__main__":
    main()