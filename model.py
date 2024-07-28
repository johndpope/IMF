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



DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)



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
        debug_print(f"LatentTokenEncoder initialized: in_channels={in_channels}, latent_dim={latent_dim}")

    def forward(self, x):
        debug_print(f"LatentTokenEncoder input shape: {x.shape}")
        if isinstance(x, list):
            return [self._forward_single(xi) for xi in x]
        else:
            return self._forward_single(x)

    def _forward_single(self, x):
        x = self.conv(x)
        debug_print(f"After conv shape: {x.shape}")
        x = x.view(x.size(0), -1)
        debug_print(f"After flatten shape: {x.shape}")
        output = self.fc(x)
        debug_print(f"LatentTokenEncoder output shape: {output.shape}")
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
        
        debug_print(f"LatentTokenDecoder initialized with {num_layers} layers")
        debug_print(f"Constant tensor shape: {self.const.shape}")
        debug_print(f"FC layer: in_features={latent_dim}, out_features={base_channels}")

    def forward(self, x):
        debug_print(f"\nLatentTokenDecoder forward pass:")
        debug_print(f"Input shape: {x.shape}")
        
        style = self.fc(x)
        debug_print(f"After FC layer shape: {style.shape}")
        
        out = self.const.repeat(x.shape[0], 1, 1, 1)
        debug_print(f"Initial out shape: {out.shape}")
        
        features = []
        for i, layer in enumerate(self.layers):
            out = layer(out, style)
            features.append(out)
            debug_print(f"Layer {i} output shape: {out.shape}")
        
        debug_print(f"Final features shapes: {[f.shape for f in features]}")
        return features
    
class StyleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, style_dim=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.modulation = nn.Linear(style_dim, in_channels)
        self.demodulation = True
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        debug_print(f"StyleConv2d initialized:")
        debug_print(f"Conv2d: in_channels={in_channels}, out_channels={out_channels}")
        debug_print(f"Modulation: in_features={style_dim}, out_features={in_channels}")

    def forward(self, x, style):
        debug_print(f"\nStyleConv2d forward pass:")
        debug_print(f"Input x shape: {x.shape}")
        debug_print(f"Input style shape: {style.shape}")
        
        batch, in_channel, height, width = x.shape
        style = self.modulation(style).view(batch, in_channel, 1, 1)
        debug_print(f"Modulated style shape: {style.shape}")
        
        x = x * style
        
        x = self.conv(x)
        debug_print(f"After conv shape: {x.shape}")
        
        if self.demodulation:
            demod = torch.rsqrt(x.pow(2).sum([2, 3], keepdim=True) + 1e-8)
            x = x * demod
        
        x = self.upsample(x)
        debug_print(f"After upsample shape: {x.shape}")
        
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

        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )

        debug_print(f"ImplicitMotionAlignment initialized: feature_dim={feature_dim}, motion_dim={motion_dim}")

    def forward(self, q, k, v):
        debug_print(f"ImplicitMotionAlignment input shapes - q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        batch_size, c, h, w = v.shape

        # Reshape and project inputs
        q = self.q_proj(q.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        k = self.k_proj(k.view(batch_size, self.motion_dim, -1).permute(0, 2, 1))
        v = self.v_proj(v.view(batch_size, self.feature_dim, -1).permute(0, 2, 1))

        debug_print(f"After projection - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Ensure q, k, and v have the same sequence length
        seq_len = min(q.size(1), k.size(1), v.size(1))
        q = q[:, :seq_len, :]
        k = k[:, :seq_len, :]
        v = v[:, :seq_len, :]

        debug_print(f"After sequence length adjustment - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Perform cross-attention
        attn_output, _ = self.cross_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)
        debug_print(f"After cross-attention: {attn_output.shape}")
        
        x = self.norm1(q + attn_output)
        x = self.norm2(x + self.ffn(x))
        debug_print(f"After FFN: {x.shape}")

        # Reshape output to match original dimensions
        x = x.transpose(1, 2).contiguous()
        debug_print(f"After transpose: {x.shape}")
        debug_print(f"Attempting to reshape to: {(batch_size, self.feature_dim, h, w)}")
        
        # Check if the reshape is valid
        if x.numel() != batch_size * self.feature_dim * h * w:
            debug_print(f"WARNING: Cannot reshape tensor of size {x.numel()} into shape {(batch_size, self.feature_dim, h, w)}")
            # Adjust the output size to match the input size
            x = F.adaptive_avg_pool2d(x.view(batch_size, self.feature_dim, -1, 1), (h, w))
        else:
            x = x.view(batch_size, self.feature_dim, h, w)
        
        debug_print(f"Final output shape: {x.shape}")

        return x


class IMF(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.dense_feature_encoder = DenseFeatureEncoder(base_channels=base_channels, num_layers=num_layers)
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, base_channels=base_channels, num_layers=num_layers)
        
        self.feature_dims = [base_channels * (2 ** i) for i in range(num_layers)]
        self.motion_dims = [base_channels // (2 ** i) for i in range(num_layers)]
        
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim)
            self.implicit_motion_alignment.append(alignment_module)

        debug_print(f"IMF initialized: num_layers={num_layers}, feature_dims={self.feature_dims}, motion_dims={self.motion_dims}")

    def forward(self, x_current, x_reference):
        debug_print(f"IMF forward - x_current shape: {x_current.shape}")
        debug_print(f"IMF forward - x_reference shape: {x_reference.shape}")
        
        f_r = self.dense_feature_encoder(x_reference)
        debug_print(f"IMF forward - f_r shapes: {[f.shape for f in f_r]}")
        
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        debug_print(f"IMF forward - t_r shape: {t_r.shape}, t_c shape: {t_c.shape}")

        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        debug_print(f"IMF forward - m_r shapes: {[m.shape for m in m_r]}")
        debug_print(f"IMF forward - m_c shapes: {[m.shape for m in m_c]}")

        aligned_features = []
        for i, (f_r_i, m_r_i, m_c_i, align_layer) in enumerate(zip(f_r, m_r, m_c, self.implicit_motion_alignment)):
            debug_print(f"IMF forward - Alignment layer {i}")
            debug_print(f"  f_r_i shape: {f_r_i.shape}")
            debug_print(f"  m_r_i shape: {m_r_i.shape}")
            debug_print(f"  m_c_i shape: {m_c_i.shape}")
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            debug_print(f"  aligned_feature shape: {aligned_feature.shape}")
            aligned_features.append(aligned_feature)

        return aligned_features

    def process_tokens(self, t_c, t_r):
        debug_print(f"process_tokens input types - t_c: {type(t_c)}, t_r: {type(t_r)}")
        
        if isinstance(t_c, list) and isinstance(t_r, list):
            debug_print(f"process_tokens input shapes - t_c: {[tc.shape for tc in t_c]}, t_r: {[tr.shape for tr in t_r]}")
            m_c = [self.latent_token_decoder(tc) for tc in t_c]
            m_r = [self.latent_token_decoder(tr) for tr in t_r]
        else:
            debug_print(f"process_tokens input shapes - t_c: {t_c.shape}, t_r: {t_r.shape}")
            m_c = self.latent_token_decoder(t_c)
            m_r = self.latent_token_decoder(t_r)
        
        if isinstance(m_c[0], list):
            debug_print(f"process_tokens output shapes - m_c: {[[mc_i.shape for mc_i in mc] for mc in m_c]}, m_r: {[[mr_i.shape for mr_i in mr] for mr in m_r]}")
        else:
            debug_print(f"process_tokens output shapes - m_c: {[m.shape for m in m_c]}, m_r: {[m.shape for m in m_r]}")
        
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
        # Ensure input tensors require gradients
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()
        
        aligned_features = self.imf(x_current, x_reference)
        reconstructed_frame = self.frame_decoder(aligned_features)
        
        # Only compute gradients if in training mode
        if self.training:
            grads = torch.autograd.grad(reconstructed_frame.sum(), [x_current, x_reference], retain_graph=True, allow_unused=True)
            print(f"Gradient of output w.r.t. x_current: {grads[0]}")
            print(f"Gradient of output w.r.t. x_reference: {grads[1]}")
        
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