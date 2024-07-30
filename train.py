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
from helper import monitor_gradients,add_gradient_hooks,sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf

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

def wasserstein_loss(y_pred, y_true):
    return torch.mean(y_true * y_pred)



def discriminator_loss(discriminator, real_images, fake_images, config):
    real_validity = discriminator(real_images)
    fake_validity = discriminator(fake_images.detach())
    
    real_loss = torch.mean(F.relu(1.0 - real_validity[0]))
    fake_loss = torch.mean(F.relu(1.0 + fake_validity[0]))
    
    return real_loss + fake_loss

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    # print(f"Real samples shape: {real_samples.shape}")
    # print(f"Fake samples shape: {fake_samples.shape}")

    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    # print(f"Alpha shape: {alpha.shape}")

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    # print(f"Interpolates shape: {interpolates.shape}")

    d_interpolates = discriminator(interpolates)
    # print(f"Discriminator output type: {type(d_interpolates)}")
    # if isinstance(d_interpolates, list):
    #     print(f"Discriminator output shapes: {[o.shape for o in d_interpolates]}")
    # else:
    #     print(f"Discriminator output shape: {d_interpolates.shape}")

    # Use the first output of the discriminator (feature map)
    fake = torch.ones_like(d_interpolates[0]).to(real_samples.device)
    # print(f"Fake tensor shape: {fake.shape}")

    try:
        gradients = torch.autograd.grad(
            outputs=d_interpolates[0],  # Use the feature map directly
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # print(f"Gradients shape: {gradients.shape}")

        gradients = gradients.view(gradients.size(0), -1)
        # print(f"Reshaped gradients shape: {gradients.shape}")

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        # print(f"Gradient penalty value: {gradient_penalty.item()}")

        return gradient_penalty
    except Exception as e:
        print(f"Error in gradient computation: {str(e)}")
        raise


def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.learning_rate_d, betas=(0.5, 0.999))

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )

    vgg_loss = VGGLoss().to(accelerator.device)
    flash_forward_steps = 100
    nan_counter = 0
    max_consecutive_nans = 5

    start_epoch = 0
    if os.path.isdir(config.checkpoints.dir):
        checkpoint_list = sorted([f for f in os.listdir(config.checkpoints.dir) if f.endswith('.pth')], reverse=True)
        if checkpoint_list:
            checkpoint_path = os.path.join(config.checkpoints.dir, checkpoint_list[0])
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
            accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            accelerator.unwrap_model(discriminator).load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            accelerator.print(f"Restored from {checkpoint_path}")

    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        total_loss_g = 0
        total_loss_d = 0

        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            try:
                # Train Discriminator
                for _ in range(config.training.n_critic):
                    optimizer_d.zero_grad()
                    
                    with torch.no_grad():
                        reconstructed_frames = model(current_frames, reference_frames)
                    
                    real_validity = discriminator(current_frames)[0]
                    fake_validity = discriminator(reconstructed_frames.detach())[0]
                    
                    gradient_penalty = compute_gradient_penalty(discriminator, current_frames, reconstructed_frames)
                    
                    loss_d = -torch.mean(real_validity) + torch.mean(fake_validity) + config.training.lambda_gp * gradient_penalty

                    # Apply R1 regularization
                    if batch_idx % config.training.r1_interval == 0:
                        r1_penalty = r1_regularization(discriminator, current_frames)
                        loss_d += config.training.r1_gamma * r1_penalty

                    if torch.isnan(loss_d):
                        raise ValueError("NaN discriminator loss detected")

                    accelerator.backward(loss_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=config.training.clip_grad_norm)
                    optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                
                reconstructed_frames = model(current_frames, reference_frames)
                
                fake_validity = discriminator(reconstructed_frames)[0]
                
                loss_pixel = pixel_loss(reconstructed_frames, current_frames)
                loss_perceptual = perceptual_loss(vgg_loss, reconstructed_frames, current_frames)
                loss_g = -torch.mean(fake_validity) + config.training.lambda_pixel * loss_pixel + config.training.lambda_perceptual * loss_perceptual

                if torch.isnan(loss_g):
                    raise ValueError("NaN generator loss detected")

                accelerator.backward(loss_g)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad_norm)
                optimizer_g.step()

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
                nan_counter = 0

            except ValueError as e:
                nan_counter += 1
                accelerator.print(f"NaN detected (count: {nan_counter}). {str(e)}. Attempting to flash forward (next movie).")
                
                if nan_counter >= max_consecutive_nans:
                    accelerator.print("Too many consecutive NaNs. Stopping training.")
                    return

                for _ in range(flash_forward_steps):
                    try:
                        next(iter(train_dataloader))
                    except StopIteration:
                        accelerator.print("Reached end of dataset while flashing forward. Resetting dataloader.")
                        train_dataloader = accelerator.prepare(DataLoader(train_dataloader.dataset, batch_size=config.training.batch_size, shuffle=True))
                        break

                progress_bar.update(flash_forward_steps)
                continue

        progress_bar.close()

        avg_loss_g = total_loss_g / len(train_dataloader)
        avg_loss_d = total_loss_d / len(train_dataloader)

        if accelerator.is_main_process:
            accelerator.print(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                              f"Avg G Loss: {avg_loss_g:.4f}, Avg D Loss: {avg_loss_d:.4f}")

            wandb.log({
                "epoch": epoch + 1,
                "avg_loss_g": avg_loss_g,
                "avg_loss_d": avg_loss_d
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
    try:
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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()