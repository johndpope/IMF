import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from model import IMFModel,debug_print
from torchvision.utils import save_image
import torchvision.transforms as transforms

from VideoDataset import VideoDataset
from decord import VideoReader, cpu
from accelerate import Accelerator
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm  # Changed this line
import yaml
from PIL import Image
import decord
from typing import List, Tuple, Dict, Any
from memory_profiler import profile
from torch.optim import AdamW
import wandb



def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

import numpy as np
from collections import defaultdict

def monitor_gradients(model, epoch, batch_idx, log_interval=10):
    """
    Monitor gradients of the model parameters.
    
    :param model: The neural network model
    :param epoch: Current epoch number
    :param batch_idx: Current batch index
    :param log_interval: How often to log gradient statistics
    """
    if batch_idx % log_interval == 0:
        grad_stats = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats['norm'].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                
                grad_stats['names'].append(name)
        
        if grad_stats['norm']:
            avg_norm = np.mean(grad_stats['norm'])
            max_norm = np.max(grad_stats['norm'])
            min_norm = np.min(grad_stats['norm'])
            
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"Gradient norms - Avg: {avg_norm:.4f}, Max: {max_norm:.4f}, Min: {min_norm:.4f}")
            
            # Identify layers with unusually high or low gradients
            threshold_high = avg_norm * 10  # Adjust this multiplier as needed
            threshold_low = avg_norm * 0.1  # Adjust this multiplier as needed
            
            for name, norm in zip(grad_stats['names'], grad_stats['norm']):
                if norm > threshold_high:
                    print(f"High gradient in {name}: {norm:.4f}")
                elif norm < threshold_low:
                    print(f"Low gradient in {name}: {norm:.4f}")
        else:
            print("No gradients to monitor")

def sample_recon(model, data, accelerator, output_path, num_samples=8):
    model.eval()
    with torch.no_grad():
        current_frames, reference_frames = data
        batch_size = current_frames.size(0)
        num_samples = min(num_samples, batch_size)  # Ensure we don't exceed the batch size
        current_frames, reference_frames = current_frames[:num_samples], reference_frames[:num_samples]
        
        # Encode frames
        tc = model.latent_token_encoder(current_frames)
        tr = model.latent_token_encoder(reference_frames)
        fr = model.dense_feature_encoder(reference_frames)

        # Get aligned features from IMF
        aligned_features = model.imf(current_frames, reference_frames)

        # Reconstruct frames
        reconstructed_frames = model.frame_decoder(aligned_features)

        # Prepare original and reconstructed frames for saving
        orig_frames = torch.cat((reference_frames, current_frames), dim=0)
        recon_frames = torch.cat((reference_frames, reconstructed_frames), dim=0)
        frames = torch.cat((orig_frames, recon_frames), dim=0)
        
        # Unnormalize frames
        frames = frames * 0.5 + 0.5
        
        # Ensure we have a valid output directory
        if output_path:
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                output_dir = '.'  # Use current directory if no directory is specified
            os.makedirs(output_dir, exist_ok=True)
            
            # Save frames as a grid
            save_image(accelerator.gather(frames), output_path, nrow=num_samples, padding=2, normalize=False)
            accelerator.print(f"Saved sample reconstructions to {output_path}")
        else:
            accelerator.print("Warning: No output path provided. Skipping image save.")

        # Log images to wandb
        wandb_images = []
        for i in range(num_samples):
            wandb_images.extend([
                wandb.Image(reference_frames[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"Reference {i}"),
                wandb.Image(current_frames[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"Current {i}"),
                wandb.Image(reconstructed_frames[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"Reconstructed {i}")
            ])

        wandb.log({"Sample Reconstructions": wandb_images})

        return frames


@profile
def train(config, model, train_dataloader, accelerator, ema_decay=0.999, style_mixing_prob=0.9, r1_gamma=10):
    debug_print("Config:", config)
    debug_print("Training config:", config.get('training', {}))

    learning_rate = config.get('training', {}).get('learning_rate', None)
    if learning_rate is None:
        raise ValueError("Learning rate not found in config")
    
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        raise ValueError(f"Invalid learning rate: {learning_rate}. Must be a valid number.")
    
    debug_print(f"Learning rate: {learning_rate}")

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Set up exponential moving average of model weights
    ema_model = accelerator.unwrap_model(model).to(accelerator.device)
    ema_model.load_state_dict(accelerator.unwrap_model(model).state_dict())

    mse_loss = nn.MSELoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoints']['dir'], exist_ok=True)

    start_epoch = 0
    # If a checkpoint exists, restore the latest one
    if os.path.isdir(config['checkpoints']['dir']):
        checkpoint_list = sorted([f for f in os.listdir(config['checkpoints']['dir']) if f.endswith('.pth')], reverse=True)
        if checkpoint_list:
            checkpoint_path = os.path.join(config['checkpoints']['dir'], checkpoint_list[0])
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
            accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            accelerator.print(f"Restored from {checkpoint_path}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}", 
                            disable=not accelerator.is_local_main_process)
        
        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            debug_print(f"Batch {batch_idx} input shapes - current_frames: {current_frames.shape}, reference_frames: {reference_frames.shape}")

            optimizer.zero_grad()

            # Forward pass
            reconstructed_frames = model(current_frames, reference_frames)
            debug_print(f"Reconstructed frames shape: {reconstructed_frames.shape}")

            loss = mse_loss(reconstructed_frames, current_frames)

            if torch.isnan(loss):
                print("NaN loss detected. Skipping this batch.")
                continue

            # R1 regularization
            if batch_idx % 16 == 0:
                current_frames.requires_grad_(True)
                reference_frames.requires_grad_(True)
                
                with torch.enable_grad():
                    reconstructed_frames = model(current_frames, reference_frames)
                    
                    r1_loss = torch.autograd.grad(
                        outputs=reconstructed_frames.sum(), 
                        inputs=[current_frames, reference_frames], 
                        create_graph=True, 
                        allow_unused=True
                    )
                    
                    if r1_loss[0] is not None and r1_loss[1] is not None:
                        r1_loss_current = r1_loss[0].pow(2).reshape(r1_loss[0].shape[0], -1).sum(1).mean()
                        r1_loss_reference = r1_loss[1].pow(2).reshape(r1_loss[1].shape[0], -1).sum(1).mean()
                        r1_loss_total = r1_loss_current + r1_loss_reference
                        loss = loss + r1_gamma * 0.5 * r1_loss_total * 16
                    else:
                        debug_print("Warning: r1_loss is None. Skipping R1 regularization for this batch.")

                current_frames.requires_grad_(False)
                reference_frames.requires_grad_(False)

            accelerator.backward(loss)

            # Gradient clipping
            clip_gradients(accelerator.unwrap_model(model), max_norm=1.0)

            # Monitor gradients
            grad_norm = gradient_norm(accelerator.unwrap_model(model))
            monitor_gradients(accelerator.unwrap_model(model), epoch, batch_idx)

            # Adaptive learning rate
            adaptive_learning_rate(optimizer, grad_norm, threshold=1.0, factor=0.5)

            optimizer.step()

            # Update exponential moving average of weights
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), accelerator.unwrap_model(model).parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_decay))

            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # Log batch loss to wandb
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader),
                "gradient_norm": grad_norm,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # Periodically monitor weights and gradients
            if batch_idx % 100 == 0:
                monitor_weights_and_gradients(accelerator.unwrap_model(model))

        progress_bar.close()
        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Average Loss: {avg_loss:.4f}")

        # Log epoch metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
        })

        # Save checkpoint
        if (epoch + 1) % config['checkpoints']['interval'] == 0:
            checkpoint_path = os.path.join(config['checkpoints']['dir'], f"checkpoint_{epoch+1}.pth")
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            accelerator.print(f"Saved checkpoint: {checkpoint_path}")

            # Log model checkpoint to wandb
            wandb.save(checkpoint_path)

        if epoch % config['logging']['sample_interval'] == 0:
            sample_path = f"recon_epoch_{epoch+1}.png"
            sample_frames = sample_recon(ema_model, next(iter(train_dataloader)), accelerator, sample_path, 
                                        num_samples=config['logging']['sample_size'])
    
            # Log sample image to wandb
            wandb.log({"sample_reconstruction": wandb.Image(sample_path)})

    return ema_model

def hook_fn(name):
    def hook(grad):
        if torch.isnan(grad).any():
            # print(f"🔥 NaN gradient detected in {name}")
            return torch.zeros_like(grad)  # Replace NaN with zero
        elif torch.isinf(grad).any():
            # print(f"🔥 Inf gradient detected in {name}")
            return torch.clamp(grad, -1e6, 1e6)  # Clamp infinite values
        # else:
            # grad_norm = grad.norm().item()
            # if grad_norm > 1000:
                # print(f"⚠️ Large gradient norm ({grad_norm:.2f}) detected in {name}")
        return grad
    return hook

def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))

def gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def adaptive_learning_rate(optimizer, gradient_norm, threshold=1.0, factor=0.5):
    if gradient_norm > threshold:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor
        print(f"Learning rate adjusted to {optimizer.param_groups[0]['lr']}")

def monitor_weights_and_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}:")
            print(f"  Weight - Mean: {param.data.mean():.4f}, Std: {param.data.std():.4f}")
            print(f"  Gradient - Mean: {param.grad.data.mean():.4f}, Std: {param.grad.data.std():.4f}")

def initialize_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param.data)
            else:
                torch.nn.init.uniform_(param.data)
        elif 'bias' in name:
            torch.nn.init.constant_(param.data, 0.0)
        else:
            print(f"Parameter {name} not initialized")

def initialize_model(model):
    
    # Initialize other parts
    for module in [model.latent_token_encoder, model.latent_token_decoder, model.implicit_motion_alignment, model.frame_decoder]:
        initialize_weights(module)

    print("Model initialization completed.")
@profile
def main():
    # Load configuration
    config = load_config('config.yaml')

    wandb.init(project='IMF', config=config,resume="allow")

    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision=config['accelerator']['mixed_precision'],
        cpu=config['accelerator']['cpu']
    )

    # Create model
    model = IMFModel(
        latent_dim=config['model']['latent_dim'],
        base_channels=config['model']['base_channels'],
        num_layers=config['model']['num_layers']
    )


    # hook for gradients
    # initialize_model(model)
    add_gradient_hooks(model)

    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    debug_print("Loading VideoDataset...")
    dataset = VideoDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        frame_skip=config['dataset']['frame_skip']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4
    )

    debug_print("Training...")
    # Train the model
    train(
        config,
        model,
        dataloader,
        accelerator,
        ema_decay=config['training']['ema_decay'],
        style_mixing_prob=config['training']['style_mixing_prob'],
        r1_gamma=config['training']['r1_gamma']
    )
    # Close wandb run
    wandb.finish()
if __name__ == "__main__":
    main()