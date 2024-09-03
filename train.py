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
from lia_resblocks import LiaDiscriminator
from model import IMFModel, debug_print,IMFPatchDiscriminator,MultiScalePatchDiscriminator
from VideoDataset import VideoDataset,gpu_padded_collate
from torchvision.utils import save_image
from helper import log_grad_flow,consistent_sub_sample,count_model_params,normalize,visualize_latent_token, add_gradient_hooks, sample_recon
from torch.optim import AdamW
from omegaconf import OmegaConf
import lpips
from torch.nn.utils import spectral_norm
import torchvision.models as models
from loss import gan_loss_fn,MediaPipeEyeEnhancementLoss
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

from stylegan import EMA
from torch.optim import AdamW, SGD
from transformers import Adafactor
from WebVid10M import WebVid10M



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


class IMFTrainer:
    def __init__(self, config, model, discriminator, train_dataloader, accelerator):
        self.config = config
        self.model = model
        self.discriminator = discriminator
        self.train_dataloader = train_dataloader
        self.accelerator = accelerator
        


        self.gan_loss_type = config.loss.type
        self.perceptual_loss_fn = lpips.LPIPS(net='alex', spatial=True).to(accelerator.device)
        self.pixel_loss_fn = nn.L1Loss()
        # self.eye_loss_fn = MediaPipeEyeEnhancementLoss(eye_weight=1.0).to(accelerator.device)


        self.style_mixing_prob = config.training.style_mixing_prob
        self.noise_magnitude = config.training.noise_magnitude
        self.r1_gamma = config.training.r1_gamma

        self.optimizer_g = AdamW(get_layer_wise_learning_rates(model), lr=2e-4, betas=(0.5, 0.999))
        self.optimizer_d = AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

        # Learning rate schedulers
        # self.scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=100, eta_min=1e-6)
        # self.scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=100, eta_min=1e-6)
        self.scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.5, patience=5, verbose=True)
        self.scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d, mode='min', factor=0.5, patience=5, verbose=True)


        if config.training.use_ema:
            self.ema = EMA(model, decay=config.training.ema_decay)
        else:
            self.ema = None

        self.model, self.discriminator, self.optimizer_g, self.optimizer_d, self.train_dataloader = accelerator.prepare(
            self.model, self.discriminator, self.optimizer_g, self.optimizer_d, self.train_dataloader
        )
        if self.ema:
            self.ema = accelerator.prepare(self.ema)
            self.ema.register()

    def check_exploding_gradients(self, model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    print(f"🔥 Exploding gradients detected in {name}")
                    return True
        return False

    def adjust_learning_rate(self, optimizer, factor=0.1, min_lr=1e-6):
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * factor, min_lr)
        print(f"🔥 Adjusted learning rate. New LR: {optimizer.param_groups[0]['lr']}")


    def train_step(self, x_current, x_reference,global_step):
        if x_current.nelement() == 0:
            print("🔥 Skipping training step due to empty x_current")
            return None, None, None, None, None, None
        

        # Forward Pass
        f_r = self.model.dense_feature_encoder(x_reference)
        t_r = self.model.latent_token_encoder(x_reference)
        t_c = self.model.latent_token_encoder(x_current)

        noise_r = torch.randn_like(t_r) * self.noise_magnitude
        noise_c = torch.randn_like(t_c) * self.noise_magnitude
        t_r = t_r + noise_r
        t_c = t_c + noise_c

        if torch.rand(()).item() < self.style_mixing_prob:
            batch_size = t_c.size(0)
            rand_indices = torch.randperm(batch_size)
            rand_t_c = t_c[rand_indices]
            rand_t_r = t_r[rand_indices]
            mix_mask = torch.rand(batch_size, 1, device=t_c.device) < 0.5
            mix_mask = mix_mask.float()
            mix_t_c = t_c * mix_mask + rand_t_c * (1 - mix_mask)
            mix_t_r = t_r * mix_mask + rand_t_r * (1 - mix_mask)
        else:
            mix_t_c = t_c
            mix_t_r = t_r

        m_c = self.model.latent_token_decoder(mix_t_c)
        m_r = self.model.latent_token_decoder(mix_t_r)

        aligned_features = []
        for i in range(len(self.model.implicit_motion_alignment)):
            f_r_i = f_r[i]
            align_layer = self.model.implicit_motion_alignment[i]
            m_c_i = m_c[i] 
            m_r_i = m_r[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)

        x_reconstructed = self.model.frame_decoder(aligned_features)
        x_reconstructed = normalize(x_reconstructed)


        # eye loss
        # l_eye = self.eye_loss_fn(x_reconstructed, x_current)
          
        if self.config.training.use_subsampling:
            sub_sample_size = (128, 128)  # As mentioned in the paper
            x_current, x_reconstructed = consistent_sub_sample(x_current, x_reconstructed, sub_sample_size)


        if global_step % self.config.logging.sample_every == 0:
            save_image(x_reconstructed, 'x_reconstructed.png',  normalize=True)
            save_image(x_current, 'x_current.png',  normalize=True)
            save_image(x_reference, 'x_reference.png',  normalize=True)

        # Discriminator updates
        d_loss_total = 0
        for _ in range(self.config.training.n_critic):
            self.optimizer_d.zero_grad()
            
            # Real samples
            real_outputs = self.discriminator(x_current)
            d_loss_real = sum(torch.mean(F.relu(1 - output)) for output in real_outputs)
            
            # Fake samples
            fake_outputs = self.discriminator(x_reconstructed.detach())
            d_loss_fake = sum(torch.mean(F.relu(1 + output)) for output in fake_outputs)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake

            # R1 regularization
            if self.config.training.use_r1_reg and global_step % self.config.training.r1_interval == 0:
                x_current.requires_grad = True
                real_outputs = self.discriminator(x_current)
                r1_reg = 0
                for real_output in real_outputs:
                    grad_real = torch.autograd.grad(
                        outputs=real_output.sum(), inputs=x_current, create_graph=True
                    )[0]
                    r1_reg += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                d_loss += self.config.training.r1_gamma * r1_reg

            self.accelerator.backward(d_loss)
            
            if self.check_exploding_gradients(self.discriminator):
                print("🔥 Skipping discriminator update due to exploding gradients")
            else:
                if self.config.training.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.config.training.clip_grad_norm)
                self.optimizer_d.step()
            
            d_loss_total += d_loss.item()


        # Average discriminator loss
        d_loss_avg = d_loss_total / self.config.training.n_critic

        # Generator update
        self.optimizer_g.zero_grad()
        fake_outputs = self.discriminator(x_reconstructed)
        g_loss_gan = sum(-torch.mean(output) for output in fake_outputs)

        l_p = self.pixel_loss_fn(x_reconstructed, x_current).mean()
        l_v = self.perceptual_loss_fn(x_reconstructed, x_current).mean()
        l_eye = self.eye_loss_fn(x_reconstructed, x_current) if self.config.training.use_eye_loss else 0

        g_loss = (self.config.training.lambda_pixel * l_p +
                self.config.training.lambda_perceptual * l_v +
                self.config.training.lambda_adv * g_loss_gan +
                self.config.training.lambda_eye * l_eye)

        self.accelerator.backward(g_loss)

        if self.check_exploding_gradients(self.model):
            print("🔥 Exploding gradients detected. Adjusting learning rate.")
            self.adjust_learning_rate(self.optimizer_g)
            self.adjust_learning_rate(self.optimizer_d)
            self.optimizer_g.zero_grad()
            self.optimizer_d.zero_grad()
        else:
            if self.config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training.clip_grad_norm)
            self.optimizer_g.step()

        if self.ema:
            self.ema.update()

        return d_loss_avg, g_loss.item(), l_p.item(), l_v.item(), g_loss_gan.item(), x_reconstructed

    def train(self, start_epoch=0):
        global_step = start_epoch * len(self.train_dataloader)


        for epoch in range(self.config.training.num_epochs):
            video_repeat = get_video_repeat(epoch, self.config.training.num_epochs, 
                                            self.config.training.initial_video_repeat, 
                                            self.config.training.final_video_repeat)

            self.model.train()
            self.discriminator.train()
            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}")

            epoch_g_loss = 0
            epoch_d_loss = 0
            num_valid_steps = 0
 
            for batch in self.train_dataloader:
                source_frames = batch['frames']
                batch_size, num_frames, channels, height, width = source_frames.shape

                for _ in range(int(video_repeat)):
                    if self.config.training.use_many_xrefs:
                        ref_indices = range(0, num_frames, self.config.training.every_xref_frames)
                    else:
                        ref_indices = [0]

                    for ref_idx in ref_indices:
                        x_reference = source_frames[:, ref_idx]

                        for current_idx in range(num_frames):
                            if current_idx == ref_idx:
                                continue

                            x_current = source_frames[:, current_idx]

                            results = self.train_step(x_current, x_reference, global_step)

                            if results[0] is not None:
                                d_loss, g_loss, l_p, l_v,  g_loss_gan, x_reconstructed = results
                                epoch_g_loss += g_loss
                                epoch_d_loss += d_loss
                                num_valid_steps += 1

                            else:
                                print("Skipping step due to error in train_step")

                   
                            epoch_g_loss += g_loss
                            epoch_d_loss += d_loss

                            if self.accelerator.is_main_process and global_step % self.config.logging.log_every == 0:
                                wandb.log({
                                    "noise_magnitude": self.noise_magnitude,
                                    "g_loss": g_loss,
                                    "d_loss": d_loss,
                                    "pixel_loss": l_p,
                                    "perceptual_loss": l_v,
                                    "gan_loss": g_loss_gan,
                                    "global_step": global_step,
                                    "lr_g": self.optimizer_g.param_groups[0]['lr'],
                                    "lr_d": self.optimizer_d.param_groups[0]['lr']
                                })
                                # Log gradient flow for generator and discriminator
                                log_grad_flow(self.model.named_parameters(),global_step)
                                log_grad_flow(self.discriminator.named_parameters(),global_step)

                            if global_step % self.config.logging.sample_every == 0:
                                sample_path = f"recon_step_{global_step}.png"
                                sample_recon(self.model, (x_reconstructed, x_current, x_reference), self.accelerator, sample_path, 
                                             num_samples=self.config.logging.sample_size)
                                
                            global_step += 1

                             # Checkpoint saving
                            if global_step % self.config.training.save_steps == 0:
                                self.save_checkpoint(epoch)

                # Calculate average losses for the epoch
                if num_valid_steps > 0:
                    avg_g_loss = epoch_g_loss / num_valid_steps
                    avg_d_loss = epoch_d_loss / num_valid_steps

                # Step the schedulers
                self.scheduler_g.step(avg_g_loss)
                self.scheduler_d.step(avg_d_loss)

               


                progress_bar.update(1)
                progress_bar.set_postfix({"G Loss": f"{g_loss:.4f}", "D Loss": f"{d_loss:.4f}"})

            progress_bar.close()
            

        # Final model saving
        self.save_checkpoint(epoch, is_final=True)


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load discriminator state
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        # Load optimizer states
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        
        # Load epoch
        start_epoch = checkpoint['epoch'] + 1
        
        print(f"Loaded checkpoint from epoch {start_epoch - 1}")
        return start_epoch

    def save_checkpoint(self, epoch, is_final=False):
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_discriminator = self.accelerator.unwrap_model(self.discriminator)
        
        if is_final:
            save_path = f"{self.config.checkpoints.dir}/final_model.pth"
            self.accelerator.save(unwrapped_model.state_dict(), save_path)
        else:
            save_path = f"{self.config.checkpoints.dir}/checkpoint.pth"
            self.accelerator.save({
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'discriminator_state_dict': unwrapped_discriminator.state_dict(),
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            }, save_path)

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
        use_resnet_feature=config.model.use_resnet_feature
    )
    add_gradient_hooks(model)

    # discriminator = MultiScalePatchDiscriminator(input_nc=3, ndf=64, n_layers=3, num_D=3)
    discriminator = LiaDiscriminator(size=256,channel_multiplier=1)
    add_gradient_hooks(discriminator)

    # dataset = WebVid10M(video_folder=config.dataset.root_dir)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoDataset(config.dataset.root_dir, 
                                transform=transform, 
                                frame_skip=0, 
                                num_frames=300)

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=gpu_padded_collate 
    )


    trainer = IMFTrainer(config, model, discriminator, dataloader, accelerator)
    # Check if a checkpoint path is provided in the config
    if config.training.load_checkpoint:
        checkpoint_path = config.training.checkpoint_path
        start_epoch = trainer.load_checkpoint(checkpoint_path)
    else:
        start_epoch = 0
    trainer.train()

if __name__ == "__main__":
    main()
