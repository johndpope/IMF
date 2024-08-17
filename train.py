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
from model import IMFModel, debug_print,MultiScalePatchDiscriminator,IMFPatchDiscriminator
from VideoDataset import VideoDataset
from EMODataset import EMODataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import plot_grad_flow,visualize_attention_maps,log_loss_landscape,log_grad_flow,count_model_params,normalize, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
from loss import wasserstein_loss,hinge_loss,vanilla_gan_loss,gan_loss_fn

import random
from vggloss import VGGLoss
from stylegan import EMA
from torch.optim import AdamW, SGD
from transformers import Adafactor
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR


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



def train(config, model, discriminator, train_dataloader, val_loader, accelerator):

    # layerwise params - 
    # layer_wise_params = get_layer_wise_learning_rates(model)

    # Generator optimizer
    optimizer_g = AdamW( model.parameters(),
        lr=config.training.learning_rate_g,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.training.weight_decay )

    # Discriminator optimizer
    optimizer_d = AdamW(
        discriminator.parameters(),
        lr=config.training.learning_rate_d,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.training.weight_decay
    )

    # dynamic learning rate
    # scheduler_g = ReduceLROnPlateau(optimizer_g, mode='min', factor=0.25, patience=10, verbose=True)
    # scheduler_d = ReduceLROnPlateau(optimizer_d, mode='min', factor=0.25, patience=10, verbose=True)

    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g,gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d,gamma=0.9)

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
    perceptual_loss_fn = lpips.LPIPS(net='alex',spatial=True).to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()
    


    style_mixing_prob = config.training.style_mixing_prob
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
        if ema:
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
                            align_layer= model.implicit_motion_alignment[i]
                            m_c_i = m_c[i] 
                            m_r_i = m_r[i]
                            aligned_feature, intermediate_outputs = align_layer(m_c_i, m_r_i, f_r_i)
                            if isinstance(intermediate_outputs, dict) and 'attn_weights' in intermediate_outputs:
                                attn_weights = intermediate_outputs['attn_weights']
                                if global_step % config.logging.visualize_every == 0:
                                    visualize_attention_maps(attn_weights, f"./visualisations/attention_maps_epoch_{epoch}_batch_{batch_idx}_layer_{i}_{global_step}.png")
                            aligned_features.append(aligned_feature)


                        # 5. Frame Decoding
                        x_reconstructed = model.frame_decoder(aligned_features)
                        x_reconstructed = normalize(x_reconstructed) # 🤷 images are washed out - or over saturated...
           
                        save_image(x_reconstructed, "x_reconstructed.png", normalize=True)
                        save_image(x_current, "x_current.png", normalize=True)
                        
                        
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
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=config.training.clip_grad_norm)
                        
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
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad_norm)
                        optimizer_g.step()


                        total_g_loss += g_loss.item()
                        total_d_loss += d_loss.item()
                        progress_bar.update(1)
                        progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})
                    
                        # Update global step
                        global_step += 1
                    

                        if global_step % config.logging.visualize_every == 0:
                            plt = plot_grad_flow(model.named_parameters())
                            plt.savefig(f'grad_flow_epoch_{epoch}_batch_{batch_idx}.png')
                            #plt.close()

                            plt = plot_grad_flow(discriminator.named_parameters())
                            plt.savefig(f'grad_flow_epoch_discrim_{epoch}_batch_{batch_idx}.png')
                            plt.close()
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
                        # Existing logs
                        log_dict = {
                            "ema": current_decay,
                            "noise_magnitude": noise_magnitude,
                            "batch_g_loss": g_loss.item(),
                            "batch_d_loss": d_loss.item(),
                            "pixel_loss": l_p.item(),
                            "perceptual_loss": l_v.item(),
                            "gan_loss": g_loss_gan.item(),
                            "batch": batch_idx + epoch * len(train_dataloader),
                        }

                        # Add layer-wise learning rates
                        component_names = [
                            'dense_feature_encoder',
                            'latent_token_encoder',
                            'latent_token_decoder',
                            'implicit_motion_alignment',
                            'frame_decoder',
                            'mapping_network'
                        ]
                        for i, param_group in enumerate(optimizer_g.param_groups):
                            log_dict[f"lr_g_{component_names[i]}"] = param_group['lr']
                        log_dict["lr_d"] = optimizer_d.param_groups[0]['lr']

                        # Add gradient norms for each component of the generator
                        for component in component_names:
                            params = getattr(model, component).parameters()
                            grad_norms = [torch.norm(p.grad.detach()) for p in params if p.grad is not None]
                            if grad_norms:
                                grad_norm = torch.norm(torch.stack(grad_norms))
                                log_dict[f"grad_norm_{component}"] = grad_norm.item()
                            else:
                                log_dict[f"grad_norm_{component}"] = 0.0

                        # Add gradient norm for the discriminator
                        disc_grad_norms = [torch.norm(p.grad.detach()) for p in discriminator.parameters() if p.grad is not None]
                        if disc_grad_norms:
                            disc_grad_norm = torch.norm(torch.stack(disc_grad_norms))
                            log_dict["grad_norm_discriminator"] = disc_grad_norm.item()
                        else:
                            log_dict["grad_norm_discriminator"] = 0.0

                        # Log to wandb
                        wandb.log(log_dict)

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
        use_mlgffn=config.model.use_mlgffn,
        use_enhanced_generator=config.model.use_enhanced_generator,
        use_skip=config.model.use_skip
    )
    
    add_gradient_hooks(model)

    d0 = IMFPatchDiscriminator(ndf=config.discriminator.ndf) 
    d1 = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
    discriminator = d1 if config.training.use_multiscale_discriminator else d0
    
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