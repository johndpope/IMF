import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
# from vit_scaled import ImplicitMotionAlignment #- SLOW but reduces memory 2x/3x
from vit import ImplicitMotionAlignment
# from vit_xformers import ImplicitMotionAlignment
from stylegan import EqualConv2d,EqualLinear
from pixelwise import PixelwiseSeparateConv, SNPixelwiseSeparateConv
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import List, Union,Tuple
# from common import DownConvResBlock,UpConvResBlock
import colored_traceback.auto # makes terminal show color coded output when crash

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

use_pixelwise = True
# keep everything in 1 class to allow copying / pasting into claude / chatgpt
class EfficientFeatureExtractor(nn.Module):
    def __init__(self, output_channels=[24, 40, 112, 320]):
        super().__init__()
        # Load pre-trained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
        
        # Define where to extract features
        self.feature_indices = [2, 3, 5, 7]  # Corresponds to different stages in EfficientNet
        
        # Add 1x1 convolutions to adjust channel dimensions if needed
        self.adjustments = nn.ModuleList([
            nn.Conv2d(self.backbone[i][-1].out_channels, out_channels, kernel_size=1)
            for i, out_channels in zip(self.feature_indices, output_channels)
        ])

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.feature_indices:
                adj_index = self.feature_indices.index(i)
                features.append(self.adjustments[adj_index](x))
        return features



class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, output_channels=[128, 256, 512, 512]):
        super().__init__()
        debug_print(f"Initializing ResNetFeatureExtractor with output_channels: {output_channels}")
        # Load a pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # We'll use the first 4 layers of ResNet
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # Add additional convolutional layers to adjust channel dimensions
        self.adjust1 = nn.Conv2d(256, output_channels[0], kernel_size=1)
        self.adjust2 = nn.Conv2d(512, output_channels[1], kernel_size=1)
        self.adjust3 = nn.Conv2d(1024, output_channels[2], kernel_size=1)
        self.adjust4 = nn.Conv2d(2048, output_channels[3], kernel_size=1)

        # for param in self.resnet.parameters(): Shoots up vram by 10gb
        #     param.requires_grad = False

    def forward(self, x):
        debug_print(f"👟 ResNetFeatureExtractor input shape: {x.shape}")
        features = []
        
        x = self.layer0(x)
        debug_print(f"After layer0: {x.shape}")
        
        x = self.layer1(x)
        debug_print(f"After layer1: {x.shape}")
        feature1 = self.adjust1(x)
        debug_print(f"After adjust1: {feature1.shape}")
        features.append(feature1)
        
        x = self.layer2(x)
        debug_print(f"After layer2: {x.shape}")
        feature2 = self.adjust2(x)
        debug_print(f"After adjust2: {feature2.shape}")
        features.append(feature2)
        
        x = self.layer3(x)
        debug_print(f"After layer3: {x.shape}")
        feature3 = self.adjust3(x)
        debug_print(f"After adjust3: {feature3.shape}")
        features.append(feature3)
        
        x = self.layer4(x)
        debug_print(f"After layer4: {x.shape}")
        feature4 = self.adjust4(x)
        debug_print(f"After adjust4: {feature4.shape}")
        features.append(feature4)
        
        debug_print(f"ResNetFeatureExtractor output: {len(features)} features")
        for i, feat in enumerate(features):
            debug_print(f"  Feature {i+1} shape: {feat.shape}")
        
        return features
      
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


class DownConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(out_channels)
        self.feat_res_block2 = FeatResBlock(out_channels)

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




class DenseFeatureEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int = 3,
                 initial_channels: int = 64,
                 down_channels: List[int] = [64, 128, 256, 512, 512],
                 initial_kernel_size: int = 7,
                 use_batch_norm: bool = True,
                 activation: nn.Module = nn.ReLU(inplace=True),
                 feature_start_index: int = 1):
        super().__init__()
        
        self.feature_start_index = feature_start_index
        
        # Initial convolution
        initial_conv_layers = [
            nn.Conv2d(in_channels, initial_channels, kernel_size=initial_kernel_size, 
                      stride=1, padding=initial_kernel_size//2)
        ]
        if use_batch_norm:
            initial_conv_layers.append(nn.BatchNorm2d(initial_channels))
        initial_conv_layers.append(activation)
        
        self.initial_conv = nn.Sequential(*initial_conv_layers)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        in_ch = initial_channels
        for out_ch in down_channels:
            self.down_blocks.append(DownConvResBlock(in_ch, out_ch))
            in_ch = out_ch

    def forward(self, x):
        debug_print(f"⚾ DenseFeatureEncoder input shape: {x.shape}")
        features = []
        x = self.initial_conv(x)
        debug_print(f"    After initial conv: {x.shape}")
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            debug_print(f"    After down_block {i+1}: {x.shape}")
            if i >= self.feature_start_index:
                features.append(x)
        debug_print(f"    DenseFeatureEncoder output shapes: {[f.shape for f in features]}")
        return features

'''
The upsample parameter is replaced with downsample to match the diagram.
The first convolution now has a stride of 2 when downsampling.
The shortcut connection now uses a 3x3 convolution with stride 2 when downsampling, instead of a 1x1 convolution.
ReLU activations are applied both after adding the residual and at the end of the block.
The FeatResBlock is now a subclass of ResBlock with downsample=False, as it doesn't change the spatial dimensions.
'''




class LatentTokenEncoder(nn.Module):
    def __init__(self, 
                 dm: int = 32, 
                 input_channels: int = 3,
                 initial_channels: int = 64,
                 res_channels: List[int] = [64, 128, 256, 512, 512, 512],
                 linear_layers: int = 4,
                 use_pixelwise: bool = True,
                 activation: nn.Module = nn.LeakyReLU(0.2)):
        super(LatentTokenEncoder, self).__init__()
        
        Conv2d = PixelwiseSeparateConv if use_pixelwise else nn.Conv2d
        self.activation = activation

        # Initial convolution
        self.conv1 = Conv2d(input_channels, initial_channels, kernel_size=3, stride=1, padding=1)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in res_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels, downsample=True))
            in_channels = out_channels
        
        # Equal convolution
        self.equalconv = EqualConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # Linear layers
        self.linear_layers = nn.ModuleList()
        for _ in range(linear_layers):
            self.linear_layers.append(EqualLinear(in_channels, in_channels))
        
        # Final linear layer
        self.final_linear = EqualLinear(in_channels, dm)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        x = self.equalconv(x)
        
        # Global average pooling
        x = x.mean([2, 3])
        
        for linear in self.linear_layers:
            x = self.activation(linear(x))
        
        x = self.final_linear(x)
        
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

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = style.view(batch, 1, in_channel, 1, 1)
        
        # Weight modulation
        weight = self.weight.unsqueeze(0) * style
        
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.weight.size(0), 1, 1, 1)

        weight = weight.view(
            batch * self.weight.size(0), in_channel, self.weight.size(2), self.weight.size(3)
        )

        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.weight.size(0), height, width)

        return out

class StyledConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, upsample=False, demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, demodulate=demodulate)
        self.style = nn.Linear(style_dim, in_channels)
        self.upsample = upsample
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, latent):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        style = self.style(latent)
        x = self.conv(x, style)
        x = self.activation(x)
        return x

class LatentTokenDecoder(nn.Module):
    def __init__(self, 
                 latent_dim: int = 32,
                 const_dim: int = 32,
                 init_size: Tuple[int, int] = (4, 4),
                 channel_config: List[int] = [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 256],
                 upsample_indices: List[int] = [1, 4, 7, 10],
                 output_indices: List[int] = [3, 6, 9, 12]):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, *init_size))
        self.output_indices = output_indices
        
        self.style_conv_layers = nn.ModuleList()
        in_channels = const_dim
        for i, out_channels in enumerate(channel_config):
            self.style_conv_layers.append(
                StyledConv(in_channels, out_channels, 3, latent_dim, 
                           upsample=(i in upsample_indices))
            )
            in_channels = out_channels

    def forward(self, t):
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        outputs = []
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            if i in self.output_indices:
                outputs.append(x)
        return tuple(reversed(outputs))  # Return in order m4, m3, m2, m1

    

'''
FrameDecoder FeatResBlock
It uses a series of UpConvResBlock layers that perform upsampling (indicated by ↑2 in the image).
It incorporates FeatResBlock layers, with 3 blocks for each of the first three levels (512, 512, and 256 channels).
It uses concatenation (Concat in the image) to combine the upsampled features with the processed input features.
The channel dimensions decrease as we go up the network: 512 → 512 → 256 → 128 → 64.
It ends with a final convolutional layer (Conv-3-k3-s1-p1) followed by a Sigmoid activation.
'''
class FrameDecoder(nn.Module):
    def __init__(self, upconv_channels=None, feat_channels=None, final_channels=3):
        super().__init__()
        
        if upconv_channels is None:
            upconv_channels = [(512, 512), (1024, 512), (768, 256), (384, 128), (128, 64)]
        if feat_channels is None:
            feat_channels = [512, 256, 128]
        
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(in_ch, out_ch) for in_ch, out_ch in upconv_channels
        ])
        
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(ch) for _ in range(3)])
            for ch in feat_channels
        ])
        
        self.final_conv = nn.Sequential(
            PixelwiseSeparateConv(upconv_channels[-1][1], final_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        debug_print(f"🎒 FrameDecoder input shapes: {[f.shape for f in features]}")

        reshaped_features = self._reshape_features(features)
        debug_print(f"Reshaped features: {[f.shape for f in reshaped_features]}")

        x = reshaped_features[-1]  # Start with the smallest feature map
        debug_print(f"    Initial x shape: {x.shape}")
        
        for i, upconv_block in enumerate(self.upconv_blocks):
            debug_print(f"\n    Processing upconv_block {i+1}")
            x = upconv_block(x)
            debug_print(f"    After upconv_block {i+1}: {x.shape}")
            
            if i < len(self.feat_blocks):
                debug_print(f"    Processing feat_block {i+1}")
                feat_input = reshaped_features[-(i+2)]
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

    def _reshape_features(self, features):
        reshaped_features = []
        for feat in features:
            if len(feat.shape) == 3:  # (batch, hw, channels)
                b, hw, c = feat.shape
                h = w = int(math.sqrt(hw))
                reshaped_feat = feat.permute(0, 2, 1).view(b, c, h, w)
            else:  # Already in (batch, channels, height, width) format
                reshaped_feat = feat
            reshaped_features.append(reshaped_feat)
        return reshaped_features
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
        Conv2d = PixelwiseSeparateConv if use_pixelwise else nn.Conv2d
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) # this used to have stride = 2
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
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4, noise_level=0.1, style_mix_prob=0.5):
        super().__init__()
        

        self.latent_token_encoder = LatentTokenEncoder(dm=latent_dim) 
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim)

        self.motion_dims = [256, 512, 512, 512]  # queries / keys
        self.feature_dims = [128, 256, 512, 512]  # values
        self.decoder_dims = self.motion_dims[-1]
        # self.dense_feature_encoder = ResNetFeatureExtractor(output_channels=self.feature_dims)
        
        self.dense_feature_encoder = EfficientFeatureExtractor(output_channels=self.feature_dims)
        
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            self.implicit_motion_alignment.append(ImplicitMotionAlignment(
                feature_dim=feature_dim,
                motion_dim=motion_dim,
                heads=8,
                dim_head=64
            ))
        
        self.frame_decoder = FrameDecoder()
        self.noise_level = noise_level
        self.style_mix_prob = style_mix_prob

        # StyleGAN2-like additions
        self.mapping_network = MappingNetwork(latent_dim, latent_dim, depth=8)
        self.noise_injection = NoiseInjection()

    def add_noise(self, tensor):
        return tensor + torch.randn_like(tensor) * self.noise_level

    def style_mixing(self, t_c, t_r):
        device = t_c.device
        if random.random() < self.style_mix_prob:
            batch_size = t_c.size(0)
            perm = torch.randperm(batch_size, device=device)
            t_c_mixed = t_c[perm]
            t_r_mixed = t_r[perm]
            mix_mask = (torch.rand(batch_size, 1, device=device) < 0.5).to(device)
            t_c = torch.where(mix_mask, t_c, t_c_mixed)
            t_r = torch.where(mix_mask, t_r, t_r_mixed)
        return t_c, t_r

    def forward(self, x_current, x_reference):
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()

        # Dense feature encoding
        f_r = self.dense_feature_encoder(x_reference)

        # Latent token encoding
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)

        # StyleGAN2-like mapping network
        t_r = self.mapping_network(t_r)
        t_c = self.mapping_network(t_c)

        # Add noise to latent tokens
        t_r = self.add_noise(t_r)
        t_c = self.add_noise(t_c)

        # Apply style mixing
        t_c, t_r = self.style_mixing(t_c, t_r)

        # Latent token decoding
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)

        # Implicit motion alignment with noise injection
        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            m_r_i = self.noise_injection(m_r[i])
            m_c_i = self.noise_injection(m_c[i])
            align_layer = self.implicit_motion_alignment[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)

        # Frame decoding
        reconstructed_frame = self.frame_decoder(aligned_features)

        return reconstructed_frame, {
            'dense_features': f_r,
            'latent_tokens': (t_c, t_r),
            'motion_features': (m_c, m_r),
            'aligned_features': aligned_features
        }

    def set_noise_level(self, noise_level):
        self.noise_level = noise_level

    def set_style_mix_prob(self, style_mix_prob):
        self.style_mix_prob = style_mix_prob

    def process_tokens(self, t_c, t_r):
        if isinstance(t_c, list) and isinstance(t_r, list):
            m_c = [self.latent_token_decoder(tc) for tc in t_c]
            m_r = [self.latent_token_decoder(tr) for tr in t_r]
        else:
            m_c = self.latent_token_decoder(t_c)
            m_r = self.latent_token_decoder(t_r)
        
        return m_c, m_r

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim, w_dim, depth):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.extend([
                nn.Linear(latent_dim if i == 0 else w_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width).to(x.device)
        return x + noise



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
    def __init__(self, input_nc=3, ndf=80):
        super(PatchDiscriminator, self).__init__()
        
        self.scale1 = nn.Sequential(
            SNPixelwiseSeparateConv(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SNPixelwiseSeparateConv(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            SNPixelwiseSeparateConv(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            SNPixelwiseSeparateConv(ndf * 4, 1, kernel_size=1, stride=1, padding=0)
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
            SNConv2d(ndf * 4, 1, kernel_size=1, stride=1, padding=0)
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



        