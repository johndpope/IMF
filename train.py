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
from model import IMFModel,PatchDiscriminator, init_weights
from VideoDataset import VideoDataset
from torchvision.utils import save_image
from collections import defaultdict
import numpy as np
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features[:36].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.vgg(x)

def pixel_loss(x_hat, x):
    # Resize x_hat to match the dimensions of x
    x_hat_resized = F.interpolate(x_hat, size=x.shape[2:], mode='bilinear', align_corners=False)
    return nn.L1Loss()(x_hat_resized, x)

def perceptual_loss(vgg, x_hat, x):
    # Resize x_hat to match x
    x_hat_resized = F.interpolate(x_hat, size=x.shape[2:], mode='bilinear', align_corners=False)
    
    losses = []
    for i in range(4):
        x_hat_features = vgg(x_hat_resized[:, :, i::2, i::2])
        x_features = vgg(x[:, :, i::2, i::2])
        losses.append(F.l1_loss(x_hat_features, x_features))
    
    return sum(losses)

def adversarial_loss(discriminator, x_hat):
    fake_outputs = discriminator(x_hat)
    return sum(-torch.mean(output) for output in fake_outputs)

def discriminator_loss(discriminator, x_hat, x):
    real_outputs = discriminator(x)
    fake_outputs = discriminator(x_hat.detach())
    real_loss = sum(torch.mean(torch.relu(1 - output)) for output in real_outputs)
    fake_loss = sum(torch.mean(torch.relu(1 + output)) for output in fake_outputs)
    return real_loss + fake_loss

def r1_regularization(discriminator, x):
    x.requires_grad_(True)
    real_outputs = discriminator(x)
    grad_real = torch.autograd.grad(
        outputs=sum(torch.sum(output) for output in real_outputs),
        inputs=x,
        create_graph=True,
        retain_graph=True,
    )[0]
    r1_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return r1_penalty


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

def sample_recon(model, data, accelerator, output_path, num_samples=4):
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
        aligned_features = model(current_frames, reference_frames)

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
    
def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], 
                             betas=(config['optimizer']['beta1'], config['optimizer']['beta2']))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config['training']['learning_rate'], 
                             betas=(config['optimizer']['beta1'], config['optimizer']['beta2']))

    model,  discriminator,optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )

    vgg_loss = VGGLoss().to(accelerator.device)

    # Check for existing checkpoints
    start_epoch = 0
    if os.path.isdir(config['checkpoints']['dir']):
        checkpoint_list = sorted([f for f in os.listdir(config['checkpoints']['dir']) if f.endswith('.pth')], reverse=True)
        if checkpoint_list:
            checkpoint_path = os.path.join(config['checkpoints']['dir'], checkpoint_list[0])
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
            accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            accelerator.unwrap_model(discriminator).load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            accelerator.print(f"Restored from {checkpoint_path}")

    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        # discriminator.train()
        total_loss_g = 0
        total_loss_d = 0

        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")

        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            # Generator step
            optimizer_g.zero_grad()

            reconstructed_frames = model(current_frames, reference_frames)

            # print(f"Current frames shape: {current_frames.shape}")
            # print(f"Reconstructed frames shape: {reconstructed_frames.shape}")

            loss_pixel = pixel_loss(reconstructed_frames, current_frames)
            loss_perceptual = perceptual_loss(vgg_loss, reconstructed_frames, current_frames)
            loss_adv = adversarial_loss(discriminator, reconstructed_frames)

            # print(f"Pixel loss: {loss_pixel.item()}")
            # print(f"Perceptual loss: {loss_perceptual.item()}")
            # print(f"Adversarial loss: {loss_adv.item()}")

            loss_g = config['training']['lambda_pixel'] * loss_pixel + \
                     config['training']['lambda_perceptual'] * loss_perceptual + \
                     config['training']['lambda_adv'] * loss_adv

            accelerator.backward(loss_g)
            # Monitor gradients before optimizer step
            monitor_gradients(model, epoch, batch_idx)
            optimizer_g.step()

            # Discriminator step
            optimizer_d.zero_grad()

            loss_d = discriminator_loss(discriminator, reconstructed_frames.detach(), current_frames)

            # R1 regularization
            if batch_idx % config['training']['r1_interval'] == 0:
                r1_penalty = r1_regularization(discriminator, current_frames)
                loss_d += config['training']['r1_gamma'] * r1_penalty * config['training']['r1_interval']

            accelerator.backward(loss_d)
            optimizer_d.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()

            progress_bar.update(1)
            progress_bar.set_postfix({"G_Loss": f"{loss_g.item():.4f}", "D_Loss": f"{loss_d.item():.4f}"})

            if accelerator.is_main_process:
                wandb.log({
                    "batch_loss_g": loss_g.item(),
                    "batch_loss_d": loss_d.item(),
                    "batch": batch_idx + epoch * len(train_dataloader)
                })

        progress_bar.close()

        avg_loss_g = total_loss_g / len(train_dataloader)
        avg_loss_d = total_loss_d / len(train_dataloader)

        if accelerator.is_main_process:
            accelerator.print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], "
                              f"Avg G Loss: {avg_loss_g:.4f}, Avg D Loss: {avg_loss_d:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "avg_loss_g": avg_loss_g,
                "avg_loss_d": avg_loss_d
            })

        if (epoch + 1) % config['checkpoints']['interval'] == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_discriminator = accelerator.unwrap_model(discriminator)
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'discriminator_state_dict': unwrapped_discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, f"{config['checkpoints']['dir']}/checkpoint_{epoch+1}.pth")


        if epoch % config['logging']['sample_interval'] == 0:
                sample_path = f"recon_epoch_{epoch+1}.png"
                sample_frames = sample_recon(model, next(iter(train_dataloader)), accelerator, sample_path, 
                                            num_samples=config['logging']['sample_size'])
        
                
                # Log sample image to wandb
                wandb.log({"sample_reconstruction": wandb.Image(sample_path)})
def main():
    config = load_config('config.yaml')
    torch.cuda.empty_cache()
    wandb.init(project='IMF', config=config)

    accelerator = Accelerator(
        mixed_precision=config['accelerator']['mixed_precision'],
        cpu=config['accelerator']['cpu']
    )

    model = IMFModel(
        latent_dim=config['model']['latent_dim'],
        base_channels=config['model']['base_channels'],
        num_layers=config['model']['num_layers']
    )

    discriminator = PatchDiscriminator(ndf=config['discriminator']['ndf'])
    # discriminator.apply(init_weights)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VideoDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        frame_skip=config['dataset']['frame_skip']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        shuffle=True
    )

    train(config, model, discriminator, dataloader, accelerator)

    wandb.finish()

if __name__ == "__main__":
    main()