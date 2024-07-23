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
from torch.utils.data import IterableDataset
from tqdm.auto import tqdm  # Changed this line
import yaml
#import wandb

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frame_skip=1):
        self.root_dir = root_dir
        self.transform = transform
        self.frame_skip = frame_skip
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
        self.total_frames = self._count_total_frames()
    
    def _count_total_frames(self):
        total = 0
        for video_path in self.video_files:
            vr = VideoReader(video_path, ctx=cpu(0))
            total += max(0, len(vr) - self.frame_skip)
        return total
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        video_idx, frame_idx = self._get_video_and_frame(idx)
        video_path = self.video_files[video_idx]
        vr = VideoReader(video_path, ctx=cpu(0))
        
        current_frame_idx = frame_idx
        reference_frame_idx = frame_idx + self.frame_skip
        
        current_frame = vr[current_frame_idx].asnumpy()
        reference_frame = vr[reference_frame_idx].asnumpy()
        
        current_frame = torch.from_numpy(current_frame).float().permute(2, 0, 1)
        reference_frame = torch.from_numpy(reference_frame).float().permute(2, 0, 1)
        
        if self.transform:
            current_frame = self.transform(current_frame)
            reference_frame = self.transform(reference_frame)
        
        return current_frame, reference_frame
    
    def _get_video_and_frame(self, idx):
        for video_idx, video_path in enumerate(self.video_files):
            vr = VideoReader(video_path, ctx=cpu(0))
            num_frames = max(0, len(vr) - self.frame_skip)
            if idx < num_frames:
                return video_idx, idx
            idx -= num_frames
        raise IndexError("Index out of range")
import torch
from torch.utils.data import IterableDataset
from decord import VideoReader, cpu
import torchvision.transforms as transforms

class SingleVideoIterableDataset(IterableDataset):
    def __init__(self, root_dir, transform=None, frame_skip=1, shuffle=True):
        self.root_dir = root_dir
        self.frame_skip = frame_skip
        self.shuffle = shuffle
        self.video_files = [os.path.join(subdir, file)
                            for subdir, dirs, files in os.walk(root_dir)
                            for file in files if file.endswith('.mp4')]
        random.shuffle(self.video_files)

        # Modify the transform pipeline
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.video_files)
        else:
            per_worker = int(math.ceil(len(self.video_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.video_files))

        for video_idx in range(iter_start, iter_end):
            video_path = self.video_files[video_idx]
            vr = VideoReader(video_path, ctx=cpu(0))
            frame_indices = list(range(len(vr) - self.frame_skip))
            if self.shuffle:
                random.shuffle(frame_indices)

            for frame_idx in frame_indices:
                current_frame_idx = frame_idx
                reference_frame_idx = frame_idx + self.frame_skip

                current_frame = vr[current_frame_idx].as_tensor()
                reference_frame = vr[reference_frame_idx].as_tensor()

                # Ensure the tensors are in the correct format (C, H, W)
                if current_frame.shape[0] != 3:
                    current_frame = current_frame.permute(2, 0, 1)
                if reference_frame.shape[0] != 3:
                    reference_frame = reference_frame.permute(2, 0, 1)

                # Apply transforms
                current_frame = self.transform(current_frame)
                reference_frame = self.transform(reference_frame)

                yield current_frame, reference_frame

    def __len__(self):
        return len(self.video_files) * 1000  # Approximation

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


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
    print("Config:", config)
    print("Training config:", config.get('training', {}))
    
    # Get learning rate with error checking and conversion
    learning_rate = config.get('training', {}).get('learning_rate', None)
    if learning_rate is None:
        raise ValueError("Learning rate not found in config")
    
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        raise ValueError(f"Invalid learning rate: {learning_rate}. Must be a valid number.")
    
    
    print(f"Learning rate: {learning_rate}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
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
            
            # Update progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        progress_bar.close()
        avg_loss = total_loss / len(train_dataloader)
        accelerator.print(f"Epoch [{epoch+1}/{config['training']['num_epochs']}], Average Loss: {avg_loss:.4f}")

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

        if epoch % config['logging']['sample_interval'] == 0:
            sample_recon(ema_model, next(iter(train_dataloader)), accelerator, f"recon_epoch_{epoch+1}.png", 
                         num_samples=config['logging']['sample_size'])
    return ema_model
if __name__ == "__main__":
    # Load configuration
    config = load_config('config.yaml')

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

    # Set up dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    print("Loading VideoDataset...")
    dataset = SingleVideoIterableDataset(
        root_dir=config['dataset']['root_dir'],
        transform=transform,
        frame_skip=config['dataset']['frame_skip']
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        num_workers=4
    )

    print("Training...")
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