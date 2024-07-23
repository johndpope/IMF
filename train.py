import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from model import IMFModel
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import random
from decord import VideoReader, cpu
from accelerate import Accelerator

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=1):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
        self.frame_pairs = self._generate_frame_pairs()
    
    def _generate_frame_pairs(self):
        frame_pairs = []
        for video_path in self.video_files:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            for i in range(0, total_frames - self.frame_skip):
                frame_pairs.append((video_path, i, i + self.frame_skip))
        return frame_pairs
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        video_path, current_frame_idx, reference_frame_idx = self.frame_pairs[idx]
        vr = VideoReader(video_path, ctx=cpu(0))
        
        current_frame = vr[current_frame_idx].asnumpy()
        reference_frame = vr[reference_frame_idx].asnumpy()
        
        current_frame = torch.from_numpy(current_frame).float().permute(2, 0, 1)
        reference_frame = torch.from_numpy(reference_frame).float().permute(2, 0, 1)
        
        if self.transform:
            current_frame = self.transform(current_frame)
            reference_frame = self.transform(reference_frame)
        
        return current_frame, reference_frame

def sample_recon(model, data, accelerator, output_path, num_samples=8):
    model.eval()
    with torch.no_grad():
        current_frames, reference_frames = data
        current_frames, reference_frames = current_frames[:num_samples], reference_frames[:num_samples]
        
        # Encode frames
        tc = model.latent_token_encoder(current_frames)
        tr = model.latent_token_encoder(reference_frames)
        fr = model.dense_feature_encoder(reference_frames)

        # Reconstruct frames
        reconstructed_frames = model.frame_decoder(model.imf(tc, tr, fr))

        # Prepare original and reconstructed frames for saving
        orig_frames = torch.cat((reference_frames, current_frames), dim=0)
        recon_frames = torch.cat((reference_frames, reconstructed_frames), dim=0)
        frames = torch.cat((orig_frames, recon_frames), dim=0)
        
        # Unnormalize frames
        frames = frames * 0.5 + 0.5
        
        # Save frames as a grid
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        save_image(accelerator.gather(frames), output_path, nrow=num_samples, padding=2, normalize=False)
        accelerator.print(f"Saved sample reconstructions to {output_path}")

def train(config, model, train_dataloader, accelerator, ema_decay=0.999, style_mixing_prob=0.9, r1_gamma=10):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    # Set up exponential moving average of model weights
    ema_model = accelerator.unwrap_model(model).to(accelerator.device)
    ema_model.load_state_dict(accelerator.unwrap_model(model).state_dict())

    mse_loss = nn.MSELoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    start_epoch = 0
    # If a checkpoint exists, restore the latest one
    if os.path.isdir(config.checkpoint_dir):
        checkpoint_list = sorted([f for f in os.listdir(config.checkpoint_dir) if f.endswith('.pth')], reverse=True)
        if checkpoint_list:
            checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_list[0])
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
            accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            accelerator.print(f"Restored from {checkpoint_path}")

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (current_frames, reference_frames) in enumerate(train_dataloader):
            # Forward pass
            reconstructed_frames = model(current_frames, reference_frames)
            
            # Add noise to latent tokens for improved training dynamics
            noise = torch.randn_like(model.latent_token_encoder(current_frames)) * 0.01
            tc = model.latent_token_encoder(current_frames) + noise
            tr = model.latent_token_encoder(reference_frames) + noise

            # Perform style mixing regularization
            if torch.rand(()).item() < style_mixing_prob:
                rand_tc = tc[torch.randperm(tc.size(0))]
                rand_tr = tr[torch.randperm(tr.size(0))]

                mix_tc = [rand_tc if torch.rand(()).item() < 0.5 else tc for _ in range(len(model.imf.implicit_motion_alignment))]
                mix_tr = [rand_tr if torch.rand(()).item() < 0.5 else tr for _ in range(len(model.imf.implicit_motion_alignment))]
            else:
                mix_tc = [tc for _ in range(len(model.imf.implicit_motion_alignment))]
                mix_tr = [tr for _ in range(len(model.imf.implicit_motion_alignment))]

            fr = model.dense_feature_encoder(reference_frames)
            reconstructed_frames = model.frame_decoder(model.imf(mix_tc, mix_tr, fr))

            loss = mse_loss(reconstructed_frames, current_frames)

            # R1 regularization for better training stability  
            if batch_idx % 16 == 0:
                current_frames.requires_grad = True
                reconstructed_frames = model.frame_decoder(model.imf(tc, tr, fr))
                r1_loss = torch.autograd.grad(outputs=reconstructed_frames.sum(), inputs=current_frames, create_graph=True)[0]
                r1_loss = r1_loss.pow(2).reshape(r1_loss.shape[0], -1).sum(1).mean()
                loss = loss + r1_gamma * 0.5 * r1_loss * 16

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            # Update exponential moving average of weights
            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(), accelerator.unwrap_model(model).parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_decay))

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch [{epoch+1}/{config.num_epochs}], Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_{epoch+1}.pth")
            accelerator.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
            accelerator.print(f"Saved checkpoint: {checkpoint_path}")

        if epoch % 10 == 0:
            sample_recon(ema_model, next(iter(train_dataloader)), accelerator, f"recon_epoch_{epoch+1}.png")

    return ema_model

if __name__ == "__main__":

    # Hyperparameters
    latent_dim = 32
    base_channels = 64
    num_layers = 4
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-4

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = IMFModel(latent_dim, base_channels, num_layers)

    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VideoDataset(root_dir="/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666", transform=transform, frame_skip=1)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Train the model
    train(model, dataloader, num_epochs, device, learning_rate, checkpoint_dir='./checkpoints', checkpoint_interval=10)

