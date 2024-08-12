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
from model import IMFModel, debug_print,MultiScalePatchDiscriminator,IMFPatchDiscriminator,ADADiscriminator
from VideoDataset import VideoDataset
from EMODataset import EMODataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import log_loss_landscape,log_grad_flow,count_model_params, add_gradient_hooks, sample_recon,compute_r1_penalty
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
from loss import wasserstein_loss,hinge_loss,vanilla_gan_loss,gan_loss_fn,compute_gradient_penalty
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

import random
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
    params.append({'params': model.latent_token_encoder.parameters(), 'lr': 1e-4})
    params.append({'params': model.latent_token_decoder.parameters(), 'lr': 1e-4})
    params.append({'params': model.implicit_motion_alignment.parameters(), 'lr': 1e-4})
    params.append({'params': model.frame_decoder.parameters(), 'lr': 1e-4})
    params.append({'params': model.mapping_network.parameters(), 'lr': 1e-4})
    return params



def train(config, model, discriminator, train_dataloader, val_loader, accelerator):
    # Layerwise params
    layer_wise_params = get_layer_wise_learning_rates(model)

    # StyleGAN2 ADA augmentation
    discriminator = ADADiscriminator(discriminator) if config.training.ada_augmentation else discriminator
    
    # Generator optimizer
    optimizer_g = AdamW(layer_wise_params,
        lr=config.training.learning_rate_g,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.training.weight_decay)

    # Discriminator optimizer
    optimizer_d = AdamW(
        discriminator.parameters(),
        lr=config.training.learning_rate_d,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.training.weight_decay
    )

    # Learning rate schedulers
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=100, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(optimizer_d, T_max=100, eta_min=1e-6)

    # EMA setup
    ema = EMA(model, decay=config.training.ema_decay) if config.training.use_ema else None

    # Prepare models and data loaders
    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )
    if ema:
        ema = accelerator.prepare(ema)
        ema.register()

    # Loss functions
    gan_loss_type = config.loss.type
    perceptual_loss_fn = lpips.LPIPS(net='alex').to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()

    # Training parameters
    style_mixing_prob = config.training.style_mixing_prob
    r1_gamma = config.training.r1_gamma

    # Monitoring variables
    global_step = 0
    d_loss_history = []
    g_loss_history = []
    update_ratio = config.training.initial_update_ratio
    update_ratio_adjustment = config.training.update_ratio_adjustment
    g_updates_per_d_update = config.training.g_updates_per_d_update

    for epoch in range(config.training.num_epochs):
        video_repeat = get_video_repeat(epoch, config.training.num_epochs, 
                                        config.training.initial_video_repeat, 
                                        config.training.final_video_repeat)
        use_ada = config.training.ada_augmentation
        
        model.train()
        discriminator.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        total_g_loss = 0
        total_d_loss = 0

        current_decay = get_ema_decay(epoch, config.training.num_epochs)
        if ema:
            ema.decay = current_decay 

        for batch_idx, batch in enumerate(train_dataloader):
            for _ in range(int(video_repeat)):
                source_frames = batch['frames']
                batch_size, num_frames, channels, height, width = source_frames.shape

                noise_magnitude = get_noise_magnitude(
                    epoch, 
                    config.training.num_epochs, 
                    initial_magnitude=config.training.initial_noise_magnitude,
                    final_magnitude=config.training.final_noise_magnitude
                )

                ref_indices = range(0, num_frames, config.training.every_xref_frames) if config.training.use_many_xrefs else [0]

                for ref_idx in ref_indices:
                    x_reference = source_frames[:, ref_idx]

                    for current_idx in range(num_frames):
                        if current_idx == ref_idx:
                            continue
                        
                        x_current = source_frames[:, current_idx]
                        
                        # Train Discriminator
                        if global_step % g_updates_per_d_update == 0:
                            for _ in range(int(update_ratio)):
                                optimizer_d.zero_grad()
                                
                                with torch.no_grad():
                                    x_reconstructed = model(x_current, x_reference, style_mixing_prob, noise_magnitude)
                                
                                real_outputs = discriminator(x_current, update_ada=use_ada)
                                fake_outputs = discriminator(x_reconstructed, update_ada=use_ada)
                                
                                d_loss = gan_loss_fn(real_outputs, fake_outputs, gan_loss_type)
                                
                                gradient_penalty = compute_gradient_penalty(discriminator, x_current, x_reconstructed)
                                r1_reg = compute_r1_penalty(discriminator, x_current)
                                d_loss = d_loss + config.training.lambda_gp * gradient_penalty + r1_gamma * r1_reg

                                accelerator.backward(d_loss)
                                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=config.training.clip_grad_norm)
                                optimizer_d.step()

                            d_loss_history.append(d_loss.item())
                            total_d_loss += d_loss.item()

                        # Train Generator
                        for _ in range(g_updates_per_d_update):
                            optimizer_g.zero_grad()
                            
                            x_reconstructed = model(x_current, x_reference, style_mixing_prob, noise_magnitude)
                            
                            fake_outputs = discriminator(x_reconstructed)
                            g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

                            l_p = pixel_loss_fn(x_reconstructed, x_current).mean()
                            l_v = perceptual_loss_fn(x_reconstructed, x_current).mean()

                            g_loss = (config.training.lambda_pixel * l_p +
                                    config.training.lambda_perceptual * l_v +
                                    config.training.lambda_adv * g_loss_gan)

                            accelerator.backward(g_loss)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad_norm)
                            optimizer_g.step()

                            if ema:
                                ema.update()

                            g_loss_history.append(g_loss.item())
                            total_g_loss += g_loss.item()

                        # Update ratio based on loss history
                        if len(d_loss_history) > 100 and len(g_loss_history) > 100:
                            d_avg = sum(d_loss_history[-100:]) / 100
                            g_avg = sum(g_loss_history[-100:]) / 100
                            
                            if d_avg < g_avg:
                                update_ratio = max(1, update_ratio - update_ratio_adjustment)
                            else:
                                update_ratio += update_ratio_adjustment

                        progress_bar.update(1)
                        progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item() if 'd_loss' in locals() else 0:.4f}"})
                        
                        global_step += 1
                        
                        # Sample and save reconstructions
                        if global_step % config.logging.save_steps == 0:
                            sample_path = f"recon_step_{global_step}.png"
                            sample_recon(model, (x_reconstructed, x_current, x_reference), accelerator, sample_path, 
                                        num_samples=config.logging.sample_size)

                        # Adjust ADA probability
                        if use_ada and (global_step + 1) % config.training.ada_interval == 0:
                            discriminator.adjust_ada_p(
                                target_r_t=config.training.ada_target_r_t,
                                ada_kimg=config.training.ada_kimg,
                                ada_interval=config.training.ada_interval,
                                batch_size=config.training.batch_size
                            )

                    # Logging
                    if accelerator.is_main_process:
                        log_dict = {
                            "ada_p": discriminator.get_ada_p() if use_ada else 0,
                            "ema": current_decay,
                            "noise_magnitude": noise_magnitude,
                            "batch_g_loss": g_loss.item(),
                            "batch_d_loss": d_loss.item() if 'd_loss' in locals() else 0,
                            "pixel_loss": l_p.item(),
                            "perceptual_loss": l_v.item(),
                            "gan_loss": g_loss_gan.item(),
                            "update_ratio": update_ratio,
                            "batch": batch_idx + epoch * len(train_dataloader),
                            "global_step": global_step,
                        }

                        # Add layer-wise learning rates and gradient norms
                        component_names = [
                            'dense_feature_encoder', 'latent_token_encoder', 'latent_token_decoder',
                            'implicit_motion_alignment', 'frame_decoder', 'mapping_network'
                        ]
                        for i, param_group in enumerate(optimizer_g.param_groups):
                            log_dict[f"lr_g_{component_names[i]}"] = param_group['lr']
                            params = getattr(model, component_names[i]).parameters()
                            grad_norms = [torch.norm(p.grad.detach()) for p in params if p.grad is not None]
                            if grad_norms:
                                grad_norm = torch.norm(torch.stack(grad_norms))
                                log_dict[f"grad_norm_{component_names[i]}"] = grad_norm.item()
                            else:
                                log_dict[f"grad_norm_{component_names[i]}"] = 0.0

                        log_dict["lr_d"] = optimizer_d.param_groups[0]['lr']
                        disc_grad_norms = [torch.norm(p.grad.detach()) for p in discriminator.parameters() if p.grad is not None]
                        if disc_grad_norms:
                            disc_grad_norm = torch.norm(torch.stack(disc_grad_norms))
                            log_dict["grad_norm_discriminator"] = disc_grad_norm.item()
                        else:
                            log_dict["grad_norm_discriminator"] = 0.0

                        wandb.log(log_dict)

                        # Log gradient flow
                        if batch_idx % config.logging.log_every == 0:
                            log_grad_flow(model.named_parameters(), global_step)
                            log_grad_flow(discriminator.named_parameters(), global_step)

        progress_bar.close()

        # Calculate average losses for the epoch
        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
        
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


    full_dataset = VideoDataset("/media/2TB/celebvhq/35666/images", 
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