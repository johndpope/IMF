import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from accelerate import Accelerator
from tqdm.auto import tqdm
import wandb
from model import IMFModel, MultiScalePatchDiscriminator,IMFPatchDiscriminator,ADADiscriminator
# from VideoDataset import VideoDataset,gpu_padded_collate
from OneVideoDataset import VideoDataset,gpu_padded_collate
from helper import log_grad_flow,count_model_params, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from loss import gan_loss_fn,compute_gradient_penalty
from torch.optim.lr_scheduler import CosineAnnealingLR
from stylegan import EMA


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
    # layerwise params
    layer_wise_params = get_layer_wise_learning_rates(model)

    # stylegan2 ada augmentation
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

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )
    if ema:
        ema = accelerator.prepare(ema)
        ema.register()

    # Loss functions
    gan_loss_type = config.loss.type
    perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=config.loss.lpips_spatial).to(accelerator.device)
    pixel_loss_fn = nn.L1Loss()

    style_mixing_prob = config.training.style_mixing_prob
    r1_gamma = config.training.r1_gamma

    global_step = 0

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
      
       # Update these lines to use the correct keys
            x_current = batch['source_frames']
            x_reference = batch['ref_frames']


            noise_magnitude = get_noise_magnitude(
                epoch, 
                config.training.num_epochs, 
                initial_magnitude=config.training.initial_noise_magnitude,
                final_magnitude=config.training.final_noise_magnitude
            )

                    

            x_reconstructed = model(x_current, x_reference, style_mixing_prob, noise_magnitude)
            
            # Compute losses that don't require gradients
            l_p = pixel_loss_fn(x_reconstructed, x_current).mean()
            l_v = perceptual_loss_fn(x_reconstructed, x_current).mean()

            # Train Discriminator
            optimizer_d.zero_grad()
            
            x_current.requires_grad = True
            real_outputs = discriminator(x_current, update_ada=use_ada)
            fake_outputs = discriminator(x_reconstructed.detach())
            
            # Compute discriminator loss
            d_loss = gan_loss_fn(real_outputs, fake_outputs, gan_loss_type)
            
            # R1 regularization
            if r1_gamma > 0:
                r1_reg = 0
                for real_output in real_outputs:
                    grad_real = torch.autograd.grad(
                        outputs=real_output.sum(), inputs=x_current, create_graph=True
                    )[0]
                    r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                d_loss = d_loss + r1_gamma * r1_reg

            # Gradient penalty
            if config.training.lambda_gp > 0:
                gradient_penalty = compute_gradient_penalty(discriminator, x_current, x_reconstructed.detach())
                d_loss = d_loss + config.training.lambda_gp * gradient_penalty

            accelerator.backward(d_loss)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=config.training.clip_grad_norm)
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            
            # Recompute fake outputs for generator training
            fake_outputs = discriminator(x_reconstructed)
            g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

            g_loss = (config.training.lambda_pixel * l_p +
                        config.training.lambda_perceptual * l_v +
                        config.training.lambda_adv * g_loss_gan)

            accelerator.backward(g_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad_norm)
            optimizer_g.step()

            if ema:
                ema.update()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            global_step += 1
            
            # Logging and visualization
            if global_step % config.logging.save_steps == 0:
                sample_path = f"recon_step_{global_step}.png"
                sample_recon(model, (x_reconstructed, x_current, x_reference), accelerator, sample_path, 
                                num_samples=config.logging.sample_size)
                if accelerator.is_main_process:
                    log_dict = {
                        "ada_p": discriminator.get_ada_p() if use_ada else 0,
                        "ema": current_decay,
                        "noise_magnitude": noise_magnitude,
                        "batch_g_loss": g_loss.item(),
                        "batch_d_loss": d_loss.item(),
                        "pixel_loss": l_p.item(),
                        "perceptual_loss": l_v.item(),
                        "gan_loss": g_loss_gan.item(),
                        "global_step": global_step,
                    }
                    wandb.log(log_dict)

            if use_ada and (global_step + 1) % config.training.ada_interval == 0:
                discriminator.adjust_ada_p(
                    target_r_t=config.training.ada_target_r_t,
                    ada_kimg=config.training.ada_kimg,
                    ada_interval=config.training.ada_interval,
                    batch_size=config.training.batch_size
                )
            progress_bar.update(1)
            progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})


            if accelerator.is_main_process and batch_idx % config.logging.log_every == 0:
                log_grad_flow(model.named_parameters(), global_step)
                log_grad_flow(discriminator.named_parameters(), global_step)

        progress_bar.close()

        # End of epoch operations
        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)
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


    full_dataset = VideoDataset(config.dataset.extracted_frames, 
                                transform=transform)
    
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