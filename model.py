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
import colored_traceback.auto


DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# keep everything in 1 class to allow copying / pasting into claude / chatgpt



class UpConvResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(channels)
        self.feat_res_block2 = FeatResBlock(channels)

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.feat_res_block1(out)
        out = self.feat_res_block2(out)
        return out

class DownConvResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(channels)
        self.feat_res_block2 = FeatResBlock(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = self.feat_res_block1(out)
        out = self.feat_res_block2(out)
        return out


'''
DenseFeatureEncoder
It starts with a Conv-64-k7-s1-p3 layer, followed by BatchNorm and ReLU, as shown in the first block of the image.
It then uses a series of DownConvResBlocks, which perform downsampling (indicated by ↓2 in the image) while increasing the number of channels:

DownConvResBlock-64
DownConvResBlock-128
DownConvResBlock-256
DownConvResBlock-512
DownConvResBlock-512


It outputs multiple feature maps (f¹ᵣ, f²ᵣ, f³ᵣ, f⁴ᵣ) as shown in the image. These are collected in the features list and returned.

Each DownConvResBlock performs downsampling using a strided convolution, maintains a residual connection, and applies BatchNorm and ReLU activations, which is consistent with typical ResNet architectures.'''
class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.down_blocks = nn.ModuleList([
            DownConvResBlock(64, 64),
            DownConvResBlock(64, 128),
            DownConvResBlock(128, 256),
            DownConvResBlock(256, 512),
            DownConvResBlock(512, 512)
        ])

    def forward(self, x):
        features = []
        
        x = self.initial_conv(x)
        
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < 4:  # Store intermediate features
                features.append(x)
        
        features.append(x)  # Add final feature
        
        return features


class LatentTokenEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.resblock1 = ResBlock(64, 64, upsample=False)
        self.resblock2 = ResBlock(64, 128, upsample=False)
        self.resblock3 = ResBlock(128, 256, upsample=False)
        
        self.resblock4 = nn.Sequential(
            ResBlock(256, 512, upsample=False),
            ResBlock(512, 512, upsample=False),
            ResBlock(512, 512, upsample=False)
        )
        
        self.equal_conv = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        
        x = self.equal_conv(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        t = self.fc_layers(x)
        
        return t
    
class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=False, style_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.style_mod = nn.Linear(style_dim, in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        style = self.style_mod(style).unsqueeze(2).unsqueeze(3)
        x = x * style
        x = self.conv(x)
        x = self.upsample(x)
        return self.activation(x)


'''
LatentTokenDecoder
It starts with a constant input (self.const), as shown at the top of the image.
It uses a series of StyleConv layers, which apply style modulation based on the input latent token t.
Some StyleConv layers perform upsampling (indicated by ↑2 in the image).
The network maintains 512 channels for most of the layers, switching to 256 channels for the final three layers.
It outputs multiple feature maps (m⁴, m³, m², m¹) as shown in the image. These are collected in the features list and returned in reverse order to match the notation in the image.
'''
class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=512, const_dim=512):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        self.style_conv_layers = nn.ModuleList([
            StyleConv(const_dim, 512),
            StyleConv(512, 512, upsample=True),
            StyleConv(512, 512),
            StyleConv(512, 512),
            StyleConv(512, 512, upsample=True),
            StyleConv(512, 512),
            StyleConv(512, 512),
            StyleConv(512, 512, upsample=True),
            StyleConv(512, 512),
            StyleConv(512, 512),
            StyleConv(256, 256, upsample=True),
            StyleConv(256, 256),
            StyleConv(256, 256)
        ])

    def forward(self, t):
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        
        features = []
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            if i in [3, 6, 9, 12]:
                features.append(x)
        
        return features[::-1]  # Return features in order [m4, m3, m2, m1]



class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, num_heads=8, num_transformer_blocks=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(motion_dim, feature_dim)
        self.k_proj = nn.Linear(motion_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        # Separate positional embeddings for queries and keys
        self.p_q = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.p_k = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(feature_dim, num_heads)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads) for _ in range(num_transformer_blocks)
        ])

    def forward(self, m_c, m_r, f_r):
        # Flatten inputs
        m_c = m_c.flatten(2).transpose(1, 2)
        m_r = m_r.flatten(2).transpose(1, 2)
        f_r = f_r.flatten(2).transpose(1, 2)

        # Project inputs and add positional embeddings
        q = self.q_proj(m_c) + self.p_q
        k = self.k_proj(m_r) + self.p_k
        v = self.v_proj(f_r)

        # Cross-attention
        attn_output, _ = self.cross_attention(q, k, v)

        # Transformer blocks
        x = attn_output
        for block in self.transformer_blocks:
            x = block(x)

        # Unflatten output
        output = x.transpose(1, 2).reshape_as(f_r)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)

        return x





'''
FrameDecoder FeatResBlock
It uses a series of UpConvResBlock layers that perform upsampling (indicated by ↑2 in the image).
It incorporates FeatResBlock layers, with 3 blocks for each of the first three levels (512, 512, and 256 channels).
It uses concatenation (Concat in the image) to combine the upsampled features with the processed input features.
The channel dimensions decrease as we go up the network: 512 → 512 → 256 → 128 → 64.
It ends with a final convolutional layer (Conv-3-k3-s1-p1) followed by a Sigmoid activation.
'''

class FrameDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(512, 512),
            UpConvResBlock(512, 512),
            UpConvResBlock(512, 256),
            UpConvResBlock(256, 128),
            UpConvResBlock(128, 64)
        ])
        
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(512) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(512) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(256) for _ in range(3)])
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        f4, f3, f2, f1 = features
        
        x = self.upconv_blocks[0](f4)
        x = torch.cat([x, self.feat_blocks[0](f3)], dim=1)
        
        x = self.upconv_blocks[1](x)
        x = torch.cat([x, self.feat_blocks[1](f2)], dim=1)
        
        x = self.upconv_blocks[2](x)
        x = torch.cat([x, self.feat_blocks[2](f1)], dim=1)
        
        x = self.upconv_blocks[3](x)
        x = self.upconv_blocks[4](x)
        
        x_c = self.final_conv(x)
        
        return x_c

class ResBlock(nn.Module):
    def __init__(self, channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu2(out)
        
        return out

class FeatResBlock(ResBlock):
    def __init__(self, channels):
        super().__init__(channels, downsample=False)
    
'''
DenseFeatureEncoder (EF): Encodes the reference frame into multi-scale features.
LatentTokenEncoder (ET): Encodes both the current and reference frames into latent tokens.
LatentTokenDecoder (IMFD): Decodes the latent tokens into motion features.
ImplicitMotionAlignment (IMFA): Aligns the reference features to the current frame using the motion features.

The forward pass:

Encodes the reference frame using the dense feature encoder.
Encodes both current and reference frames into latent tokens.
Decodes the latent tokens into motion features.
For each scale, aligns the reference features to the current frame using the ImplicitMotionAlignment module.
'''
class IMF(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.dense_feature_encoder = DenseFeatureEncoder()
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, const_dim=base_channels)
        
        # Define feature dimensions for each layer
        self.feature_dims = [64, 128, 256, 512, 512]
        
        # Create ImplicitMotionAlignment modules for each layer
        self.implicit_motion_alignment = nn.ModuleList([
            ImplicitMotionAlignment(
                feature_dim=self.feature_dims[i],
                motion_dim=base_channels // (2 ** i) if i < 4 else base_channels // 8
            ) for i in range(num_layers)
        ])

    def forward(self, x_current, x_reference):
        # Encode reference frame
        f_r = self.dense_feature_encoder(x_reference)
        
        # Encode latent tokens
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        
        # Decode motion features
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        
        # Align features
        aligned_features = []
        for i, (f_r_i, m_r_i, m_c_i, align_layer) in enumerate(zip(f_r, m_r, m_c, self.implicit_motion_alignment)):
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)
        
        return aligned_features



'''
The IMF class is now used as a submodule, which encapsulates the DenseFeatureEncoder, LatentTokenEncoder, LatentTokenDecoder, and ImplicitMotionAlignment modules.
The FrameDecoder is initialized without specifying feature dimensions, as it now expects a list of 4 feature maps as input, which is consistent with the output of the IMF.
In the forward method, we first get the aligned features from the IMF, then pass these to the FrameDecoder to reconstruct the current frame.
We've kept the gradient computation during training, which could be useful for certain training techniques or analyses.
The property decorators for accessing the encoders and decoder are maintained, now accessing them through the IMF submodule.
'''
class IMFModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4):
        super().__init__()
        self.imf = IMF(latent_dim, base_channels, num_layers)
        
        # Update feature_dims to match the output of DenseFeatureEncoder
        feature_dims = [64, 128, 256, 512, 512]
        self.frame_decoder = FrameDecoder()

    def forward(self, x_current, x_reference):
        # Enable gradient computation for input tensors
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()
        
        # Get aligned features from IMF
        aligned_features = self.imf(x_current, x_reference)
        
        # Reconstruct the current frame using the FrameDecoder
        reconstructed_frame = self.frame_decoder(aligned_features)
        
        # Compute gradients during training
        if self.training:
            grads = torch.autograd.grad(reconstructed_frame.sum(), [x_current, x_reference], retain_graph=True, allow_unused=True)
        
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