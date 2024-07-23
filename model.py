import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os


class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class LatentTokenEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)
        self.layers = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_layers):
            out_channels = in_channels // 2 if i < num_layers - 1 else in_channels
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

    def forward(self, x):
        x = self.fc(x).view(x.size(0), -1, 4, 4)
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, q, k, v):
        attn_output, _ = self.cross_attention(q, k, v)
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))
        return x

class IMF(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.dense_feature_encoder = DenseFeatureEncoder(base_channels=base_channels, num_layers=num_layers)
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, base_channels=base_channels, num_layers=num_layers)
        self.implicit_motion_alignment = nn.ModuleList([
            ImplicitMotionAlignment(base_channels * (2 ** i)) for i in range(num_layers)
        ])

    def forward(self, x_current, x_reference):
        # Encode reference frame
        f_r = self.dense_feature_encoder(x_reference)
        t_r = self.latent_token_encoder(x_reference)

        # Encode current frame
        t_c = self.latent_token_encoder(x_current)

        # Decode motion features
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)

        # Align features
        aligned_features = []
        for i, (f_r_i, m_r_i, m_c_i, align_layer) in enumerate(zip(f_r, m_r, m_c, self.implicit_motion_alignment)):
            q = m_c_i.flatten(2).permute(2, 0, 1)
            k = m_r_i.flatten(2).permute(2, 0, 1)
            v = f_r_i.flatten(2).permute(2, 0, 1)
            aligned_feature = align_layer(q, k, v)
            aligned_feature = aligned_feature.permute(1, 2, 0).view_as(f_r_i)
            aligned_features.append(aligned_feature)

        return aligned_features

# Example usage
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = IMF().to(device)
#     x_current = torch.randn(1, 3, 256, 256).to(device)
#     x_reference = torch.randn(1, 3, 256, 256).to(device)
#     aligned_features = model(x_current, x_reference)
#     for i, feature in enumerate(aligned_features):
#         print(f"Aligned feature {i} shape:", feature.shape)




class FrameDecoder(nn.Module):
    def __init__(self, base_channels=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = base_channels * (2 ** (num_layers - 1))
        for i in range(num_layers):
            out_channels = in_channels // 2 if i < num_layers - 1 else 3
            self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels) if i < num_layers - 1 else nn.Identity(),
                    nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Tanh()
                )
            )
            in_channels = out_channels

    def forward(self, features):
        x = features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.cat([x, features[-i-2]], dim=1)
        return x

class IMFModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.imf = IMF(latent_dim, base_channels, num_layers)
        self.frame_decoder = FrameDecoder(base_channels, num_layers)

    def forward(self, x_current, x_reference):
        aligned_features = self.imf(x_current, x_reference)
        reconstructed_frame = self.frame_decoder(aligned_features)
        return reconstructed_frame

class VideoDataset(Dataset):
    def __init__(self, video_dir, transform=None):
        self.video_dir = video_dir
        self.frame_files = sorted(os.listdir(video_dir))
        self.transform = transform

    def __len__(self):
        return len(self.frame_files) - 1  # We need pairs of frames

    def __getitem__(self, idx):
        current_frame = self.load_frame(self.frame_files[idx])
        reference_frame = self.load_frame(self.frame_files[idx + 1])
        
        if self.transform:
            current_frame = self.transform(current_frame)
            reference_frame = self.transform(reference_frame)
        
        return current_frame, reference_frame

    def load_frame(self, file_name):
        frame_path = os.path.join(self.video_dir, file_name)
        frame = Image.open(frame_path).convert('RGB')
        return frame

def train(model, dataloader, num_epochs, device, learning_rate=1e-4):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (current_frames, reference_frames) in enumerate(dataloader):
            current_frames, reference_frames = current_frames.to(device), reference_frames.to(device)
            
            optimizer.zero_grad()
            reconstructed_frames = model(current_frames, reference_frames)
            
            loss = mse_loss(reconstructed_frames, current_frames)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
        
        # Save sample reconstructions
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_current, sample_reference = next(iter(dataloader))
                sample_current, sample_reference = sample_current.to(device), sample_reference.to(device)
                sample_reconstructed = model(sample_current, sample_reference)
                save_image(sample_reconstructed, f"reconstruction_epoch_{epoch+1}.png")

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
    train(model, dataloader, num_epochs, device, learning_rate)
