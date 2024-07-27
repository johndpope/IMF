import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
import torchvision.models as models
from memory_profiler import profile

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet model
        resnet = models.resnet50(pretrained=pretrained)
        
        # We'll use the first 4 layers of ResNet
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        features = []
        x = self.layer0(x)
        features.append(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        return features


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
        
        print(f"LatentTokenEncoder initialized:")
        print(f"Input channels: {in_channels}")
        print(f"Latent dimension: {latent_dim}")
        print("Convolutional layers:")
        for i, layer in enumerate(self.conv):
            if isinstance(layer, nn.Conv2d):
                print(f"  Conv2d {i}: in_channels={layer.in_channels}, out_channels={layer.out_channels}, kernel_size={layer.kernel_size}")
        print(f"Final FC layer: in_features={self.fc.in_features}, out_features={self.fc.out_features}")

    def forward(self, x):
        print(f"\nLatentTokenEncoder forward pass:")
        print(f"Input shape: {x.shape}")
        
        for i, layer in enumerate(self.conv):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.AdaptiveAvgPool2d):
                print(f"After layer {i} ({type(layer).__name__}) shape: {x.shape}")
        
        x = x.view(x.size(0), -1)
        print(f"After flattening shape: {x.shape}")
        
        output = self.fc(x)
        print(f"Final output shape: {output.shape}")
        
        return output

class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.const = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        self.fc = nn.Linear(latent_dim, base_channels)
        self.layers = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_layers):
            out_channels = in_channels // 2 if i < num_layers - 1 else in_channels
            self.layers.append(
                StyleConv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, style_dim=base_channels)
            )
            in_channels = out_channels
        
        print(f"LatentTokenDecoder initialized with {num_layers} layers")
        print(f"Constant tensor shape: {self.const.shape}")
        print(f"FC layer: in_features={latent_dim}, out_features={base_channels}")

    def forward(self, x):
        print(f"\nLatentTokenDecoder forward pass:")
        print(f"Input shape: {x.shape}")
        
        style = self.fc(x)
        print(f"After FC layer shape: {style.shape}")
        
        out = self.const.repeat(x.shape[0], 1, 1, 1)
        print(f"Initial out shape: {out.shape}")
        
        features = []
        for i, layer in enumerate(self.layers):
            out = layer(out, style)
            features.append(out)
            print(f"Layer {i} output shape: {out.shape}")
        
        print(f"Final features shapes: {[f.shape for f in features]}")
        return features
    
class StyleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.modulation = nn.Linear(style_dim, in_channels)
        self.demodulation = True
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        print(f"StyleConv2d initialized:")
        print(f"Conv2d: in_channels={in_channels}, out_channels={out_channels}")
        print(f"Modulation: in_features={style_dim}, out_features={in_channels}")

    def forward(self, x, style):
        print(f"\nStyleConv2d forward pass:")
        print(f"Input x shape: {x.shape}")
        print(f"Input style shape: {style.shape}")
        
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, in_channel, 1, 1)
        print(f"Modulated style shape: {style.shape}")
        
        x = x * style
        
        x = self.conv(x)
        print(f"After conv shape: {x.shape}")
        
        if self.demodulation:
            demod = torch.rsqrt(x.pow(2).sum([2, 3], keepdim=True) + 1e-8)
            x = x * demod
        
        x = self.upsample(x)
        print(f"After upsample shape: {x.shape}")
        
        return x
class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim 
        self.motion_dim = motion_dim 
        self.num_heads = num_heads

        self.q_proj = nn.Linear(motion_dim, feature_dim)
        self.k_proj = nn.Linear(motion_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.downsample = nn.AdaptiveAvgPool2d((32, 32))
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
        print(f"ImplicitMotionAlignment initialized:")
        print(f"Feature dimension: {feature_dim}")
        print(f"Motion dimension: {motion_dim}")
        print(f"Number of heads: {num_heads}")

    def forward(self, q, k, v):
        print(f"ImplicitMotionAlignment input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        batch_size, _, h_q, w_q = q.shape
        _, _, h_v, w_v = v.shape

        # Downsample inputs
        q = self.downsample(q)
        k = self.downsample(k)
        v = self.downsample(v)
        print(f"After downsampling - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Reshape and project inputs
        q = self.q_proj(q.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        k = self.k_proj(k.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        v = self.v_proj(v.view(batch_size, self.feature_dim, -1).permute(0, 2, 1))

        # Ensure q, k, and v have the same sequence length
        seq_len = min(q.size(1), k.size(1), v.size(1))
        q = q[:, :seq_len, :]
        k = k[:, :seq_len, :]
        v = v[:, :seq_len, :]

        print(f"Shape of q after reshaping and projection: {q.shape}")
        print(f"Shape of k after reshaping and projection: {k.shape}")
        print(f"Shape of v after reshaping and projection: {v.shape}")

        # Perform cross-attention
        attn_output, _ = self.cross_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        print(f"Shape after cross-attention: {attn_output.shape}")
        
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))
        print(f"Shape after FFN: {x.shape}")

        # Reshape output to match downsampled dimensions
        x = x.transpose(1, 2).view(batch_size, self.feature_dim, 32, 32)
        print(f"Shape after reshaping: {x.shape}")
        
        # Upsample back to original size
        x = F.interpolate(x, size=(h_v, w_v), mode='bilinear', align_corners=False)
        print(f"Final output shape: {x.shape}")

        return x


class IMF(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.dense_feature_encoder = DenseFeatureEncoder(base_channels=base_channels, num_layers=num_layers)
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, base_channels=base_channels, num_layers=num_layers)
        
        # Calculate feature dimensions for each layer
        self.feature_dims = [base_channels * (2 ** i) for i in range(num_layers)]
        
        # Calculate motion dimensions for each layer
        self.motion_dims = [base_channels // (2 ** i) for i in range(num_layers)]
        
        # Create ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            print(f"Creating ImplicitMotionAlignment for layer {i} with feature_dim={feature_dim} and motion_dim={motion_dim}")
            alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim)
            self.implicit_motion_alignment.append(alignment_module)

        
        print("IMF initialized:")
        print(f"Number of layers: {num_layers}")
        print(f"Feature dimensions: {self.feature_dims}")
        print(f"Motion dimensions: {self.motion_dims}")


    def forward(self, x_current, x_reference):
        # Encode reference frame
        f_r = self.dense_feature_encoder(x_reference)
        t_r = self.latent_token_encoder(x_reference)
        print(f"Shape of f_r (encoded reference frame): {[f.shape for f in f_r]}")
        print(f"Shape of t_r (latent tokens from reference frame): {t_r.shape}")

        # Encode current frame
        t_c = self.latent_token_encoder(x_current)
        print(f"Shape of t_c (latent tokens from current frame): {t_c.shape}")

        # Decode motion features
        m_r = self.latent_token_decoder(t_r)
        print(f"Shape of m_r (motion features from reference frame): {[m.shape for m in m_r]}")
        m_c = self.latent_token_decoder(t_c)
        print(f"Shape of m_c (motion features from current frame): {[m.shape for m in m_c]}")

        # Align features
        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            m_r_i = m_r[i]
            m_c_i = m_c[i]
            align_layer = self.implicit_motion_alignment[i]
            
            print(f"Aligning features at layer {i}:")
            print(f"  Shape of f_r_i: {f_r_i.shape}")
            print(f"  Shape of m_r_i: {m_r_i.shape}")
            print(f"  Shape of m_c_i: {m_c_i.shape}")
            
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            print(f"  Shape of aligned_feature: {aligned_feature.shape}")
            aligned_features.append(aligned_feature)

        return aligned_features
    
    def process_tokens(self, t_c, t_r):
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        return m_c, m_r


class ResNetIMF(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.dense_feature_encoder = ResNetFeatureExtractor()
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim)
        
        # Adjust the implicit motion alignment for ResNet feature dimensions
        feature_dims = [64, 256, 512, 1024]  # ResNet50 feature dimensions
        self.implicit_motion_alignment = nn.ModuleList([
            ImplicitMotionAlignment(dim) for dim in feature_dims
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



class FrameDecoder(nn.Module):
    def __init__(self, base_channels=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = base_channels * (2 ** (num_layers - 1))
        for i in range(num_layers):
            out_channels = in_channels // 2 if i < num_layers - 1 else 3
            self.layers.append(
                ResBlock(in_channels, out_channels)
            )
            in_channels = out_channels

    def forward(self, features):
        x = features[-1]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x + features[-i-2]
        return torch.tanh(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return self.upsample(out)

class IMFModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.imf = IMF(latent_dim, base_channels, num_layers)
        self.frame_decoder = FrameDecoder(base_channels, num_layers)

    def forward(self, x_current, x_reference):
        aligned_features = self.imf(x_current, x_reference)
        reconstructed_frame = self.frame_decoder(aligned_features)
        return reconstructed_frame

    @property
    def latent_token_encoder(self):
        return self.imf.latent_token_encoder

    @property
    def dense_feature_encoder(self):
        return self.imf.dense_feature_encoder

    @property
    def latent_token_decoder(self):
        return self.imf.latent_token_decoder
