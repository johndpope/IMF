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
from model import IMFModel,PatchDiscriminator, init_weights,debug_print
from VideoDataset import VideoDataset
from torchvision.utils import save_image
from helper import monitor_gradients,add_gradient_hooks,sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips


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
    mse_loss = nn.MSELoss()
    lpips_loss = lpips.LPIPS(net='alex').to(accelerator.device)
    flash_forward_steps = 100
    nan_counter = 0
    max_consecutive_nans = 5
    step_counter = 0  # Add this line to keep track of steps

    # Add gradient monitoring
    # add_gradient_hooks(model)
    # add_gradient_hooks(discriminator)

    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma
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
        total_mse_loss = 0
        total_perceptual_loss = 0
        total_loss = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            step_counter += 1  # Increment step counter

            debug_print(f"Batch {batch_idx} input shapes - current_frames: {current_frames.shape}, reference_frames: {reference_frames.shape}")

            # Forward pass
            reconstructed_frames = model(current_frames, reference_frames)
            debug_print(f"Reconstructed frames shape: {reconstructed_frames.shape}")

            # Add noise to latent tokens for improved training dynamics
            tc = model.latent_token_encoder(current_frames)
            tr = model.latent_token_encoder(reference_frames)
            debug_print(f"Latent token shapes - tc: {tc.shape}, tr: {tr.shape}")

            noise_magnitude = 0.1
            noise = torch.randn_like(tc) * noise_magnitude
            tc = tc + noise
            tr = tr + noise

            # Perform style mixing regularization
            if torch.rand(()).item() < style_mixing_prob:
                rand_tc = tc[torch.randperm(tc.size(0))]
                rand_tr = tr[torch.randperm(tr.size(0))]

                mix_tc = [rand_tc if torch.rand(()).item() < 0.5 else tc for _ in range(len(model.implicit_motion_alignment))]
                mix_tr = [rand_tr if torch.rand(()).item() < 0.5 else tr for _ in range(len(model.implicit_motion_alignment))]
            else:
                mix_tc = [tc] * len(model.implicit_motion_alignment)
                mix_tr = [tr] * len(model.implicit_motion_alignment)

            debug_print(f"Mixed token shapes - mix_tc: {[t.shape for t in mix_tc]}, mix_tr: {[t.shape for t in mix_tr]}")

            m_c, m_r = model.process_tokens(mix_tc, mix_tr)

            fr = model.dense_feature_encoder(reference_frames)
            debug_print(f"Dense feature encoder output shapes: {[f.shape for f in fr]}")

            aligned_features = []
            for i in range(len(model.implicit_motion_alignment)):
                f_r_i = fr[i]
                align_layer = model.implicit_motion_alignment[i]
                m_c_i = m_c[i][i]  # Access the i-th element of the i-th sublist
                m_r_i = m_r[i][i]  # Access the i-th element of the i-th sublist
                debug_print(f"Layer {i} input shapes - f_r_i: {f_r_i.shape}, m_c_i: {m_c_i.shape}, m_r_i: {m_r_i.shape}")
                aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
                debug_print(f"Layer {i} aligned feature shape: {aligned_feature.shape}")
                aligned_features.append(aligned_feature)

            with torch.set_grad_enabled(True):
                reconstructed_frames = model.frame_decoder(aligned_features)
                debug_print(f"Final reconstructed frames shape: {reconstructed_frames.shape}")
                mse = mse_loss(reconstructed_frames, current_frames)
                perceptual = lpips_loss(reconstructed_frames, current_frames).mean()
                loss = mse + 0.1 * perceptual

    
                 # R1 regularization for better training stability  
                if batch_idx % 16 == 0:
                    current_frames.requires_grad_(True)
                    reference_frames.requires_grad_(True)
                    
                    with torch.enable_grad():
                        reconstructed_frames = model(current_frames, reference_frames)
                        debug_print(f"Reconstructed frames shape before R1: {reconstructed_frames.shape}")
                        debug_print(f"Current frames shape before R1: {current_frames.shape}")
                        
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
                            debug_print(f"r1_loss_current shape: {r1_loss_current.shape}")
                            debug_print(f"r1_loss_reference shape: {r1_loss_reference.shape}")
                            loss = loss + r1_gamma * 0.5 * r1_loss_total * 16
                        else:
                            debug_print("Warning: r1_loss is None. Skipping R1 regularization for this batch.")

                    current_frames.requires_grad_(False)
                    reference_frames.requires_grad_(False)
            accelerator.backward(loss)
             # Monitor gradients before optimizer step
            # monitor_gradients(model, epoch, batch_idx)
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


            optimizer_g.step()
            optimizer_g.zero_grad()

            # Update exponential moving average of weights
            with torch.no_grad():
                for p_ema, p in zip(model.parameters(), accelerator.unwrap_model(model).parameters()):
                    p_ema.copy_(p.lerp(p_ema, config.training.ema_decay))

            total_mse_loss += mse.item()
            total_perceptual_loss += perceptual.item()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

            # Log batch loss to wandb
            wandb.log({
                "batch_mse_loss": mse.item(),
                "batch_perceptual_loss": perceptual.item(),
                "batch_total_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader)
            })

            # Save reconstruction sample every 250 steps
            if step_counter % config.training.save_steps == 0:
                sample_path = f"recon_step_{step_counter}.png"
                sample_frames = sample_recon(model, (current_frames, reference_frames), accelerator, sample_path, 
                                            num_samples=config.logging.sample_size)
                
                wandb.log({"sample_reconstruction": wandb.Image(sample_path)})

        progress_bar.close()

        avg_loss_g = total_loss / len(train_dataloader)
        # avg_loss_d = total_loss_d / len(train_dataloader)

        if accelerator.is_main_process:
            accelerator.print(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                              f"Avg G Loss: {avg_loss_g:.4f}") #  Avg D Loss: {avg_loss_d:.4f}

            wandb.log({
                "batch_mse_loss": mse.item(),
                "batch_perceptual_loss": perceptual.item(),
                "batch_total_loss": loss.item(),
                "batch": batch_idx + epoch * len(train_dataloader)
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
        # add_gradient_hooks(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


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