import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import os
from model import IMFModel


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

    for epoch in range(start_epoch, num_epochs):
        for batch_idx, (current_frames, reference_frames) in enumerate(dataloader):
            current_frames, reference_frames = current_frames.to(device), reference_frames.to(device)

            # ... (Training loop remains the same)

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

