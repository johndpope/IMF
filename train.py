import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from model import IMFModel
from torchvision.utils import save_image
import torchvision.transforms as transforms

def sample_recon(model, data, device, output_path, num_samples=8):
    model.eval()
    with torch.no_grad():
        current_frames, reference_frames = data
        current_frames, reference_frames = current_frames[:num_samples].to(device), reference_frames[:num_samples].to(device)
        
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
        save_image(frames, output_path, nrow=num_samples, padding=2, normalize=False)
        print(f"Saved sample reconstructions to {output_path}")


def train(model, dataloader, num_epochs, device, learning_rate=1e-4, ema_decay=0.999, style_mixing_prob=0.9, r1_gamma=10, checkpoint_dir='./checkpoints', checkpoint_interval=10):
    model.train().to(device)

    # Set up exponential moving average of model weights
    ema_model = type(model)().to(device)
    ema_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    mse_loss = nn.MSELoss()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    # If a checkpoint exists, restore the latest one
    if os.path.isdir(checkpoint_dir):
        checkpoint_list = sorted(os.listdir(checkpoint_dir), reverse=True)
        if checkpoint_list:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_list[0])
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Restored from {checkpoint_path}")

    for epoch in range(num_epochs):
            for batch_idx, (current_frames, reference_frames) in enumerate(dataloader):
                current_frames, reference_frames = current_frames.to(device), reference_frames.to(device)

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

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()  

                # Update exponential moving average of weights
                ema_model.requires_grad_(False)
                with torch.no_grad():  
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.copy_(p.lerp(p_ema, ema_decay))
                    for b_ema, b in zip(ema_model.buffers(), model.buffers()):
                        b_ema.copy_(b)
                ema_model.requires_grad_(True)  

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            if epoch % 10 == 0:
                model.eval()
                ema_model.eval()
                sample_recon(ema_model, next(iter(dataloader)), device, f"recon_epoch_{epoch+1}.png")

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
    dataset = VideoDataset("path/to/your/video/frames", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Train the model
    train(model, dataloader, num_epochs, device, learning_rate, checkpoint_dir='./checkpoints', checkpoint_interval=10)

