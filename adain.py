import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, style):
        b, c, h, w = x.shape
        style = style.view(b, 2, c, 1, 1)
        
        mean, std = style[:, 0], style[:, 1]
        
        x = F.instance_norm(x)
        x = x * std + mean
        
        print(f"AdaIN output shape: {x.shape}")
        print(f"AdaIN output stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        return x

class AdaINResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.adain1 = AdaIN()
        self.adain2 = AdaIN()
        self.style_mlp = nn.Linear(style_dim, out_channels * 4)  # 2 for each AdaIN (mean and std)
        self.activation = nn.LeakyReLU(0.2)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, style):
        residual = self.skip(x)
        
        style_params = self.style_mlp(style).chunk(2, dim=1)
        
        print(f"AdaINResBlock input shape: {x.shape}")
        
        out = self.conv1(x)
        out = self.adain1(out, style_params[0])
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.adain2(out, style_params[1])
        out = self.activation(out)
        
        out = out + residual
        
        print(f"AdaINResBlock output shape: {out.shape}")
        print(f"AdaINResBlock output stats: min={out.min().item():.4f}, max={out.max().item():.4f}, mean={out.mean().item():.4f}, std={out.std().item():.4f}")
        
        assert not torch.isnan(out).any(), "NaN values detected in AdaINResBlock output"
        assert not torch.isinf(out).any(), "Inf values detected in AdaINResBlock output"
        
        return out

class AdaINLatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, style_dim=32, const_dim=64):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 8, 8))
        self.style_mlp = nn.Linear(latent_dim, style_dim)
        
        self.adain_blocks = nn.ModuleList([
            AdaINResBlock(const_dim, 512, style_dim),
            AdaINResBlock(512, 512, style_dim),
            AdaINResBlock(512, 512, style_dim),
            AdaINResBlock(512, 256, style_dim),
        ])
        
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        ])

    def forward(self, t):
        batch_size = t.shape[0]
        x = self.const.repeat(batch_size, 1, 1, 1)
        style = self.style_mlp(t)
        
        print(f"🎑  AdaINLatentTokenDecoder Initial const shape: {x.shape}")
        print(f"Style shape: {style.shape}")
        
        features = []
        for i, (block, upsample) in enumerate(zip(self.adain_blocks, self.upsamples)):
            x = block(x, style)
            x = upsample(x)
            features.append(x)
            print(f"Feature {i+1} shape: {x.shape}")
        
        return features[::-1]  # Return features in order [m4, m3, m2, m1]