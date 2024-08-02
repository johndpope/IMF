import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange



class AdaptiveLayerNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out

class ImprovedStyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=False, style_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.adain = AdaptiveLayerNorm(out_channels, style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()

    def forward(self, x, style, noise=None):
        x = self.upsample(x)
        x = self.conv(x)
        if noise is None:
            noise = torch.randn_like(x)
        x = x + self.noise_weight * noise
        x = self.adain(x, style)
        x = self.activation(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        residual = x
        
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        out = out + residual
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        
        return out

class ImprovedLatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, const_dim=512, max_features=4):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        self.style_mapping = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim * max_features)
        )
        
        self.style_conv_layers = nn.ModuleList([
            ImprovedStyleConv(const_dim, 512, style_dim=latent_dim),
            ImprovedStyleConv(512, 512, upsample=True, style_dim=latent_dim),
            ImprovedStyleConv(512, 512, style_dim=latent_dim),
            AttentionBlock(512),
            ImprovedStyleConv(512, 512, upsample=True, style_dim=latent_dim),
            ImprovedStyleConv(512, 512, style_dim=latent_dim),
            AttentionBlock(512),
            ImprovedStyleConv(512, 512, upsample=True, style_dim=latent_dim),
            ImprovedStyleConv(512, 256, style_dim=latent_dim),
            AttentionBlock(256),
            ImprovedStyleConv(256, 256, upsample=True, style_dim=latent_dim),
            ImprovedStyleConv(256, 128, style_dim=latent_dim),
            AttentionBlock(128)
        ])

    def forward(self, t):
        print(f"👻 ImprovedLatentTokenDecoder latent t shape: {t.shape}")
        print(f"Input latent t stats: mean={t.mean().item():.4f}, std={t.std().item():.4f}")

        batch_size = t.shape[0]
        x = self.const.repeat(batch_size, 1, 1, 1)
        print(f"Initial const x shape: {x.shape}")
        print(f"Initial const x stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")

        styles = self.style_mapping(t).view(batch_size, -1, t.shape[1])
        print(f"Mapped styles shape: {styles.shape}")
        print(f"Mapped styles stats: mean={styles.mean().item():.4f}, std={styles.std().item():.4f}")

        features = []
        for i, layer in enumerate(self.style_conv_layers):
            if isinstance(layer, ImprovedStyleConv):
                style = styles[:, min(i, styles.shape[1]-1)]
                print(f"Layer {i} (StyleConv) - Input shape: {x.shape}, Style shape: {style.shape}")
                x = layer(x, style)
            else:  # AttentionBlock
                print(f"Layer {i} (AttentionBlock) - Input shape: {x.shape}")
                x = layer(x)
            
            print(f"Layer {i} output shape: {x.shape}")
            print(f"Layer {i} output stats: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
            
            if i in [3, 6, 9, 12]:
                features.append(x)
                print(f"Added feature {len(features)} with shape: {x.shape}")

        print("Final features:")
        for i, feat in enumerate(features[::-1]):
            print(f"Feature {i+1} shape: {feat.shape}")
            print(f"Feature {i+1} stats: mean={feat.mean().item():.4f}, std={feat.std().item():.4f}")

        return features[::-1]  # Return features in order [m4, m3, m2, m1]

