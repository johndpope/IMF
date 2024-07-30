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
from torchvision.utils import save_image
from helper import monitor_gradients, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
from perceptual import PerceptualLoss


def load_config(config_path):
    return OmegaConf.load(config_path)


def wasserstein_loss(real_outputs, fake_outputs):
    real_loss = sum(torch.mean(out) for out in real_outputs)
    fake_loss = sum(torch.mean(out) for out in fake_outputs)
    return fake_loss - real_loss

def hinge_loss(real_outputs, fake_outputs):
    real_loss = sum(torch.mean(F.relu(1 - out)) for out in real_outputs)
    fake_loss = sum(torch.mean(F.relu(1 + out)) for out in fake_outputs)
    return real_loss + fake_loss

def vanilla_gan_loss(real_outputs, fake_outputs):
    real_loss = sum(F.binary_cross_entropy_with_logits(out, torch.ones_like(out)) for out in real_outputs)
    fake_loss = sum(F.binary_cross_entropy_with_logits(out, torch.zeros_like(out)) for out in fake_outputs)
    return real_loss + fake_loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features[:36].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.vgg(x)


def pixel_loss(x_hat, x):
    return nn.L1Loss()(x_hat, x)

def adversarial_loss(discriminator, x_hat):
    fake_outputs = discriminator(x_hat)
    return sum(-torch.mean(output) for output in fake_outputs)



def train(config, model, discriminator, train_dataloader, accelerator):
    optimizer_g = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate_g, betas=(config.optimizer.beta1, config.optimizer.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.training.learning_rate_d, betas=(config.optimizer.beta1, config.optimizer.beta2))

    model, discriminator, optimizer_g, optimizer_d, train_dataloader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_dataloader
    )

    vgg_loss = VGGLoss().to(accelerator.device)
    perceptual_loss_fn = PerceptualLoss(accelerator.device, weights={'vgg19': 20.0, 'vggface': 4.0, 'gaze': 5.0,'lpips':10.0})
    
    style_mixing_prob = config.training.style_mixing_prob
    noise_magnitude = config.training.noise_magnitude
    r1_gamma = config.training.r1_gamma


    # Select the loss function based on config
    if config.loss.type == "wasserstein":
        loss_fn = wasserstein_loss
    elif config.loss.type == "hinge":
        loss_fn = hinge_loss
    elif config.loss.type == "vanilla":
        loss_fn = vanilla_gan_loss
    else:
        raise ValueError(f"Unsupported loss type: {config.loss.type}")


    for epoch in range(config.training.num_epochs):
        model.train()
        discriminator.train()
        total_g_loss = 0
        total_d_loss = 0
        step_counter = 0
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config.training.num_epochs}")

        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            step_counter += 1 # Increment step counter
            # Train Discriminator
            for _ in range(config.training.n_critic):
                optimizer_d.zero_grad()

                with torch.no_grad():
                    reconstructed_frames = model(current_frames, reference_frames)

                real_outputs = discriminator(current_frames)
                fake_outputs = discriminator(reconstructed_frames.detach())

                d_loss = loss_fn(real_outputs, fake_outputs)

                accelerator.backward(d_loss)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()

                total_d_loss += d_loss.item()

            # Train Generator
            optimizer_g.zero_grad()

            # Style mixing and noise injection
            tc = model.latent_token_encoder(current_frames)
            tr = model.latent_token_encoder(reference_frames)

            noise = torch.randn_like(tc) * noise_magnitude
            tc = tc + noise
            tr = tr + noise

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

          
            reconstructed_frames = model.frame_decoder(aligned_features)
            debug_print(f"Final reconstructed frames shape: {reconstructed_frames.shape}")              
            # lpips_l = lpips_loss(reconstructed_frames, current_frames).mean()
            perceptual_l = perceptual_loss_fn(reconstructed_frames, current_frames)
            adv_l = adversarial_loss(discriminator, reconstructed_frames)

            current_frames.requires_grad_(False)
            reference_frames.requires_grad_(False)
            g_loss = ( config.training.lambda_perceptual * perceptual_l +
                config.training.lambda_adv * adv_l)

            accelerator.backward(g_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer_g.step()

            total_g_loss += g_loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix({"G Loss": f"{g_loss.item():.4f}", "D Loss": f"{d_loss.item():.4f}"})

            wandb.log({
                "batch_g_loss": g_loss.item(),
                "batch_d_loss": d_loss.item(),
                "perceptual_l": perceptual_l.item(),
                "batch": batch_idx + epoch * len(train_dataloader)
            })

            if step_counter % config.training.save_steps == 0:
                sample_path = f"recon_epoch_{epoch+1}.png"
                sample_frames = sample_recon(model, next(iter(train_dataloader)), accelerator, sample_path, 
                                            num_samples=config.logging.sample_size)
                
                # wandb.log({"sample_reconstruction": wandb.Image(sample_path)})

        progress_bar.close()

        avg_g_loss = total_g_loss / len(train_dataloader)
        avg_d_loss = total_d_loss / len(train_dataloader)

        if accelerator.is_main_process:
            accelerator.print(f"Epoch [{epoch+1}/{config.training.num_epochs}], "
                              f"Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

            wandb.log({
                "epoch_g_loss": avg_g_loss,
                "epoch_d_loss": avg_d_loss,
                "epoch": epoch
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

if __name__ == "__main__":
    main()