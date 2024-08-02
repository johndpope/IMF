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
from torch.utils.checkpoint import checkpoint
from performer_pytorch import SelfAttention
import math
import torch.nn.utils.spectral_norm as spectral_norm
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
from dense_encoders import SwinV2FeatureExtractor,SwinV2LatentTokenEncoder
from vit import ImplicitMotionAlignment
from latent_encoder import ImprovedLatentTokenDecoder

# from CIPS import CIPSFrameDecoder
DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class PixelwiseSeparateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SNPixelwiseSeparateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                                 stride=stride, padding=padding, groups=in_channels))
        self.pointwise = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DownConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelwise=False):
        super().__init__()
        Conv2d = PixelwiseSeparateConv if use_pixelwise else nn.Conv2d
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(out_channels, use_pixelwise)
        self.feat_res_block2 = FeatResBlock(out_channels, use_pixelwise)

    def forward(self, x):
        debug_print(f"DownConvResBlock input shape: {x.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        debug_print(f"After conv1, bn1, relu: {out.shape}")
        out = self.avgpool(out)
        debug_print(f"After avgpool: {out.shape}")
        out = self.conv2(out)
        debug_print(f"After conv2: {out.shape}")
        out = self.feat_res_block1(out)
        debug_print(f"After feat_res_block1: {out.shape}")
        out = self.feat_res_block2(out)
        debug_print(f"DownConvResBlock output shape: {out.shape}")
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
        debug_print(f"⚾ DenseFeatureEncoder input shape: {x.shape}")
        features = []
        x = self.initial_conv(x)
        debug_print(f"    After initial conv: {x.shape}")
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            
            if i >= 1:  # Start collecting features from the second block
                debug_print(f"    After down_block {i+1}: {x.shape}")
                features.append(x)
        debug_print(f"    DenseFeatureEncoder output shapes: {[f.shape for f in features]}")
        return features


class LatentTokenEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.resblock1 = ResBlock(64, 64, downsample=False)
        self.resblock2 = ResBlock(64, 128, downsample=True)
        self.resblock3 = ResBlock(128, 256, downsample=True)
        
        self.resblock4 = nn.Sequential(
            ResBlock(256, 512, downsample=True),
            ResBlock(512, 512, downsample=False),
            ResBlock(512, 512, downsample=False)
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
        debug_print(f"💳 LatentTokenEncoder input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        debug_print(f"    After first conv, bn, relu: {x.shape}")
        x = self.resblock1(x)
        debug_print(f"    After resblock1: {x.shape}")
        x = self.resblock2(x)
        debug_print(f"    After resblock2: {x.shape}")
        x = self.resblock3(x)
        debug_print(f"    After resblock3: {x.shape}")
        x = self.resblock4(x)
        debug_print(f"    After resblock4: {x.shape}")
        x = self.equal_conv(x)
        debug_print(f"    After equal_conv: {x.shape}")
        x = self.adaptive_pool(x)
        debug_print(f"    After adaptive_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        debug_print(f"    After flatten: {x.shape}")
        t = self.fc_layers(x)
        debug_print(f"    1xdm=32 LatentTokenEncoder output shape: {t.shape}")
        return t

    
class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=False, style_dim=32):
        super().__init__()
        self.conv = PixelwiseSeparateConv(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.style_mod = nn.Linear(style_dim, in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        debug_print(f"StyleConv input shape: x: {x.shape}, style: {style.shape}")
        style = self.style_mod(style).unsqueeze(2).unsqueeze(3)
        debug_print(f"After style modulation: {style.shape}")
        x = x * style
        debug_print(f"After multiplication: {x.shape}")
        x = self.conv(x)
        debug_print(f"After conv: {x.shape}")
        x = self.upsample(x)
        debug_print(f"After upsample: {x.shape}")
        x = self.activation(x)
        debug_print(f"StyleConv output shape: {x.shape}")
        return x


'''
LatentTokenDecoder
It starts with a constant input (self.const), as shown at the top of the image.
It uses a series of StyleConv layers, which apply style modulation based on the input latent token t.
Some StyleConv layers perform upsampling (indicated by ↑2 in the image).
The network maintains 512 channels for most of the layers, switching to 256 channels for the final three layers.
It outputs multiple feature maps (m⁴, m³, m², m¹) as shown in the image. These are collected in the features list and returned in reverse order to match the notation in the image.




Latent Token Decoder
Transforms compact tokens (tr, tc) into multiple motion features (mlr and mlc) with specific dimensions, where "l" indicates the layer index.

Style Modulation:

Uses a technique from StyleGAN2 called Weight Modulation and Demodulation.
This technique scales convolution weights with the latent token and normalizes them to have a unit standard deviation.
Benefits:

The latent tokens compress multi-scale information, making them comprehensive representations.
Our representation is fully implicit, allowing flexible adjustment of the latent token dimension for different scenarios.
Advantages Over Keypoint-Based Methods:

Unlike keypoint-based methods that use Gaussian heatmaps converted from keypoints, our design scales better and has more capabilities.
Our latent tokens are directly learned by the encoder, rather than being restricted to coordinates with a limited value range.
'''
class LatentTokenDecoder(nn.Module):
    def __init__(self, latent_dim=32, const_dim=64):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        self.style_conv_layers = nn.ModuleList([
            StyleConv(const_dim, 512, style_dim=latent_dim),
            StyleConv(512, 512, upsample=True, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 512, upsample=True, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 512, upsample=True, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 512, style_dim=latent_dim),
            StyleConv(512, 256, upsample=True, style_dim=latent_dim),
            StyleConv(256, 256, style_dim=latent_dim),
            StyleConv(256, 256, style_dim=latent_dim)
        ])

    def forward(self, t):
        debug_print(f"🎑 LatentTokenDecoder input shape: {t.shape}")
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        debug_print(f"    After const: {x.shape}")
        features = []
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            debug_print(f"    After style_conv {i+1}: {x.shape}")
            if i in [3, 6, 9, 12]:
                features.append(x)
        debug_print(f"    LatentTokenDecoder output shapes: {[f.shape for f in features[::-1]]}")
        return features[::-1]  # Return features in order [m4, m3, m2, m1]






class FeatResBlock(nn.Module):
    def __init__(self, channels, use_pixelwise=False):
        super().__init__()
        Conv2d = PixelwiseSeparateConv if use_pixelwise else nn.Conv2d
        self.conv1 = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out

'''
FrameDecoder FeatResBlock
It uses a series of UpConvResBlock layers that perform upsampling (indicated by ↑2 in the image).
It incorporates FeatResBlock layers, with 3 blocks for each of the first three levels (512, 512, and 256 channels).
It uses concatenation (Concat in the image) to combine the upsampled features with the processed input features.
The channel dimensions decrease as we go up the network: 512 → 512 → 256 → 128 → 64.
It ends with a final convolutional layer (Conv-3-k3-s1-p1) followed by a Sigmoid activation.
'''

class FrameDecoder(nn.Module):
    def __init__(self, feature_dims=[256, 512, 1024, 1024]):
        super().__init__()
        
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(feature_dims[3], feature_dims[2]),
            UpConvResBlock(feature_dims[2] * 2, feature_dims[1]),
            UpConvResBlock(feature_dims[1] * 2, feature_dims[0]),
            UpConvResBlock(feature_dims[0] * 2, feature_dims[0] // 2),
            UpConvResBlock(feature_dims[0] // 2, feature_dims[0] // 4)
        ])
        
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(feature_dims[2]) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(feature_dims[1]) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(feature_dims[0]) for _ in range(3)])
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_dims[0] // 4, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        debug_print(f"🎒 FrameDecoder input shapes: {[f.shape for f in features]}")

        x = features[-1]  # Start with the smallest feature map
        debug_print(f"    Initial x shape: {x.shape}")
        
        for i in range(len(self.upconv_blocks)):
            debug_print(f"\n    Processing upconv_block {i+1}")
            x = self.upconv_blocks[i](x)
            debug_print(f"    After upconv_block {i+1}: {x.shape}")
            
            if i < len(self.feat_blocks):
                debug_print(f"    Processing feat_block {i+1}")
                feat_input = features[-(i+2)]
                debug_print(f"    feat_block {i+1} input shape: {feat_input.shape}")
                feat = self.feat_blocks[i](feat_input)
                debug_print(f"    feat_block {i+1} output shape: {feat.shape}")
                
                debug_print(f"    Concatenating: x {x.shape} and feat {feat.shape}")
                x = torch.cat([x, feat], dim=1)
                debug_print(f"    After concatenation: {x.shape}")
        
        debug_print("\n    Applying final convolution")
        x = self.final_conv(x)
        debug_print(f"    FrameDecoder final output shape: {x.shape}")

        return x

class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixelwise=True):
        super().__init__()
        Conv2d = PixelwiseSeparateConv if use_pixelwise else nn.Conv2d
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block = FeatResBlock(out_channels, use_pixelwise)
        
        self.residual_conv = Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        
        # Add a 1x1 convolution for the residual connection if channel sizes differ
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.upsample(x)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        out = self.feat_res_block(out)
        return out


'''
The upsample parameter is replaced with downsample to match the diagram.
The first convolution now has a stride of 2 when downsampling.
The shortcut connection now uses a 3x3 convolution with stride 2 when downsampling, instead of a 1x1 convolution.
ReLU activations are applied both after adding the residual and at the end of the block.
The FeatResBlock is now a subclass of ResBlock with downsample=False, as it doesn't change the spatial dimensions.
'''
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        debug_print(f"ResBlock input shape: {x.shape}")
        debug_print(f"ResBlock parameters: in_channels={self.in_channels}, out_channels={self.out_channels}, downsample={self.downsample}")

        residual = self.shortcut(x)
        debug_print(f"After shortcut: {residual.shape}")
        
        out = self.conv1(x)
        debug_print(f"After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)
        debug_print(f"After bn1 and relu1: {out.shape}")
        
        out = self.conv2(out)
        debug_print(f"After conv2: {out.shape}")
        out = self.bn2(out)
        debug_print(f"After bn2: {out.shape}")
        
        out += residual
        debug_print(f"After adding residual: {out.shape}")
        
        out = self.relu2(out)
        debug_print(f"ResBlock output shape: {out.shape}")
        
        return out


class TokenManipulationNetwork(nn.Module):
    def __init__(self, token_dim, condition_dim, hidden_dim=256):
        super().__init__()
        self.token_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim)
        )

    def forward(self, token, condition):
        token_encoded = self.token_encoder(token)
        condition_encoded = self.condition_encoder(condition)
        combined = torch.cat([token_encoded, condition_encoded], dim=-1)
        edited_token = self.decoder(combined)
        return edited_token



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
class IMFModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4, condition_dim=None):
        super().__init__()
        # self.dense_feature_encoder = DenseFeatureEncoder() #- original from paper - there's no features in it
        # self.dense_feature_encoder = ViTFeatureExtractor() - google

        self.dense_feature_encoder = SwinV2FeatureExtractor()
        # self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_encoder = SwinV2LatentTokenEncoder(latent_dim=latent_dim)

        self.latent_token_decoder = ImprovedLatentTokenDecoder(latent_dim=latent_dim, const_dim=base_channels)
        
        # Update these to match SwinV2 output
        self.feature_dims = [256, 512, 1024, 1024]
        
        # Adjust these to match your LatentTokenDecoder output
        self.motion_dims = [128, 256, 512, 512]  # Example values, adjust as needed
        
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            self.implicit_motion_alignment.append(ImplicitMotionAlignment(
                feature_dim=feature_dim,
                motion_dim=motion_dim,
                depth=2,
                heads=8,
                dim_head=64,
                mlp_dim=1024
            ))
        
        self.frame_decoder = FrameDecoder()
        
        # Add token manipulation network if condition_dim is provided
        if condition_dim is not None:
            self.token_manipulation = TokenManipulationNetwork(latent_dim, condition_dim)
        else:
            self.token_manipulation = None

    def forward(self, x_current, x_reference, condition=None):
        print(f"🚀 IMFModel forward pass:")
        print(f"  Input shapes: x_current: {x_current.shape}, x_reference: {x_reference.shape}")

        f_r = self.dense_feature_encoder(x_reference)
        print(f"  Dense feature encoder output shapes: {[f.shape for f in f_r]}")

        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        print(f"  Latent token encoder outputs: t_r: {t_r.shape}, t_c: {t_c.shape}")

        if condition is not None and self.token_manipulation is not None:
            t_c = self.token_manipulation(t_c, condition)
            print(f"  After token manipulation: t_c: {t_c.shape}")

        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        print(f"  Latent token decoder outputs:")
        print(f"    m_r shapes: {[m.shape for m in m_r]}")
        print(f"    m_c shapes: {[m.shape for m in m_c]}")

        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            m_r_i = m_r[i]
            m_c_i = m_c[i]
            align_layer = self.implicit_motion_alignment[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)
            print(f"  Aligned feature {i+1} shape: {aligned_feature.shape}")

        reconstructed_frame = self.frame_decoder(aligned_features)
        print(f"  Reconstructed frame shape: {reconstructed_frame.shape}")

        return reconstructed_frame

    def process_tokens(self, t_c, t_r):
        if isinstance(t_c, list) and isinstance(t_r, list):
            m_c = [self.latent_token_decoder(tc) for tc in t_c]
            m_r = [self.latent_token_decoder(tr) for tr in t_r]
        else:
            m_c = self.latent_token_decoder(t_c)
            m_r = self.latent_token_decoder(t_r)
        
        return m_c, m_r



class SNConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(SNConv2d, self).__init__(*args, **kwargs)
        self = spectral_norm(self)

    def forward(self, input):
        return super(SNConv2d, self).forward(input)
'''
PatchDiscriminator
This implementation of the PatchDiscriminator class follows the architecture described in the supplementary material of the IMF paper. Here are the key features:

Multi-scale: The discriminator operates on two scales. The first scale (scale1) processes the input at its original resolution, while the second scale (scale2) processes a downsampled version of the input.
Spectral Normalization: We use spectral normalization on all convolutional layers to stabilize training, as indicated by the "SN" in the paper's diagram.
Architecture: Each scale follows the structure described in the paper:

4x4 convolutions with stride 2
LeakyReLU activation (α=0.2)
Instance Normalization (except for the first and last layers)
Channel progression: 64 -> 128 -> 256 -> 512 -> 1


Output: The forward method returns a list containing the outputs from both scales.
Weight Initialization: A helper function init_weights is provided to initialize the weights of the network, which can be applied using the apply method.
'''
class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(PatchDiscriminator, self).__init__()
        
        self.scale1 = nn.Sequential(
            SNConv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0)
        )
        
        self.scale2 = nn.Sequential(
            SNConv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            SNConv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        debug_print(f"PatchDiscriminator input shape: {x.shape}")

        # Scale 1
        output1 = x
        for i, layer in enumerate(self.scale1):
            output1 = layer(output1)
            debug_print(f"Scale 1 - Layer {i} output shape: {output1.shape}")

        # Scale 2
        x_downsampled = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        debug_print(f"Scale 2 - Downsampled input shape: {x_downsampled.shape}")
        
        output2 = x_downsampled
        for i, layer in enumerate(self.scale2):
            output2 = layer(output2)
            debug_print(f"Scale 2 - Layer {i} output shape: {output2.shape}")

        debug_print(f"PatchDiscriminator final output shapes: {output1.shape}, {output2.shape}")
        return [output1, output2]

# Helper function to initialize weights
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)