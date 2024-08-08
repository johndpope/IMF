import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
from vit import ImplicitMotionAlignment
from stylegan import EqualConv2d,EqualLinear
# from common import DownConvResBlock,UpConvResBlock
import colored_traceback.auto # makes terminal show color coded output when crash

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        debug_print(*args, **kwargs)

# keep everything in 1 class to allow copying / pasting into claude / chatgpt
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_channels,pretrained=True, freeze=True):
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

        for param in self.resnet.parameters():
            param.requires_grad = False

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block = FeatResBlock(out_channels)
        
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
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out



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
            debug_print(f"    After down_block {i+1}: {x.shape}")
            if i >= 1:  # Start collecting features from the second block
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
    def __init__(self, input_channels=3, dm=32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        
        self.activation = nn.LeakyReLU(0.2)

        self.res1 = ResBlock(64, 64, downsample=True)  # ResBlock-64, ↓2
        self.res2 = ResBlock(64, 128, downsample=True)  # ResBlock-128, ↓2
        self.res3 = ResBlock(128, 256, downsample=True)  # ResBlock-256, ↓2
        
        # 3× ResBlock-512, ↓2
        self.res4 = ResBlock(256, 512, downsample=True)
        self.res5 = ResBlock(512, 512, downsample=True)
        self.res6 = ResBlock(512, 512, downsample=True)
        
        self.equalconv = EqualConv2d(512, 512, kernel_size=3, stride=1, padding=1)

        
        # 4× EqualLinear-512
        self.linear1 = EqualLinear(512, 512)
        self.linear2 = EqualLinear(512, 512)
        self.linear3 = EqualLinear(512, 512)
        self.linear4 = EqualLinear(512, 512)
        
        self.final_linear = EqualLinear(512, dm) # EqualLinear-dm

    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        
        x = self.equalconv(x)
        
        # Global average pooling instead of flattening
        x = x.mean([2, 3])
        
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))
        
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
    def __init__(self, latent_dim=32, const_dim=32):
        super().__init__()
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))
        
        self.style_conv_layers = nn.ModuleList([
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim)
        ])

    def forward(self, t):
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        m1, m2, m3, m4 = None, None, None, None
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            if i == 3:
                m1 = x
            elif i == 6:
                m2 = x
            elif i == 9:
                m3 = x
            elif i == 12:
                m4 = x
        return m4, m3, m2, m1 
    

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
        self.upconv_512_1 = UpConvResBlock(512, 512)
        self.feat_res_512 = nn.Sequential(*[FeatResBlock(512) for _ in range(3)])
        self.upconv_512_2 = UpConvResBlock(1024, 512) # implied from diagram as we concat 512 + 512
        self.feat_res_512_2 = nn.Sequential(*[FeatResBlock(512) for _ in range(3)])
        self.upconv_256 = UpConvResBlock(1024, 256) # looks harsh downgrade - is diagram wrong 🤷
        self.feat_res_256 = nn.Sequential(*[FeatResBlock(256) for _ in range(3)])
        self.upconv_128 = UpConvResBlock(512, 128)
        self.upconv_64 = UpConvResBlock(128, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, features):
        f_c1, f_c2,f_c3, f_c4  = features
        debug_print(f"Input f_c4 shape: {f_c4.shape}") #  [1, 512, 8, 8]
        
        spatial_dims = [f.shape[-1] for f in [f_c4, f_c3, f_c2, f_c1]]
        assert spatial_dims == sorted(spatial_dims), f"Spatial dimensions must be in ascending order. Got: {spatial_dims}"
        # First UpConvResBlock
        x = self.upconv_512_1(f_c4)
        debug_print(f"After 1st UpConvResBlock-512, ↑2 shape: {x.shape}") # [1, 512, 16, 16]
        
        # First FeatResBlock and concatenation
        f3 = self.feat_res_512(f_c3)
        x = torch.cat([x, f3], dim=1)

        # Second UpConvResBlock
        x = self.upconv_512_2(x)
        debug_print(f"After 2nd UpConvResBlock-512, ↑2 shape: {x.shape}") #[1, 512, 32, 32]
        
        # Second FeatResBlock and concatenation
        f2 = self.feat_res_512_2(f_c2)
        x = torch.cat([x, f2], dim=1)
        debug_print(f"After concat with f_c2 shape: {x.shape}") # [1, 1024, 32, 32]
        
        # Third UpConvResBlock
        x = self.upconv_256(x)
        debug_print(f"After UpConvResBlock-256, ↑2 shape: {x.shape}") # [1, 256, 64, 64]
        
        # Third FeatResBlock and concatenation
        f1 = self.feat_res_256(f_c1)
        x = torch.cat([x, f1], dim=1)
        debug_print(f"After concat with f_c1 shape: {x.shape}") # [1, 512, 64, 64]
        
        # Final UpConvResBlocks
        x = self.upconv_128(x)
        debug_print(f"After UpConvResBlock-128, ↑2 shape: {x.shape}")
        
        x = self.upconv_64(x)
        debug_print(f"After UpConvResBlock-64, ↑2 shape: {x.shape}")
        
        # Final convolution and activation
        x = self.final_conv(x)
        x = self.sigmoid(x)
        debug_print(f"Final output shape: {x.shape}")
        
        return x


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
        # debug_print(f"ResBlock input shape: {x.shape}")
        # debug_print(f"ResBlock parameters: in_channels={self.in_channels}, out_channels={self.out_channels}, downsample={self.downsample}")

        residual = self.shortcut(x)
        # debug_print(f"After shortcut: {residual.shape}")
        
        out = self.conv1(x)
        # debug_print(f"After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)
        # debug_print(f"After bn1 and relu1: {out.shape}")
        
        out = self.conv2(out)
        # debug_print(f"After conv2: {out.shape}")
        out = self.bn2(out)
        # debug_print(f"After bn2: {out.shape}")
        
        out += residual
        # debug_print(f"After adding residual: {out.shape}")
        
        out = self.relu2(out)
        # debug_print(f"ResBlock output shape: {out.shape}")
        
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



class StyleMixer(nn.Module):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, t_c, t_r,style_mix_prob):
        if self.training and torch.rand(1).item() < self.prob:
            batch_size = t_c.size(0)
            perm = torch.randperm(batch_size)
            mix_mask = (torch.rand(batch_size, 1, device=t_c.device) < 0.5).float()
            t_c = t_c * mix_mask + t_c[perm] * (1 - mix_mask)
            t_r = t_r * mix_mask + t_r[perm] * (1 - mix_mask)
        return t_c, t_r

    def set_prob(self, prob):
        self.prob = prob
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
    def __init__(self, latent_dim=32,  num_layers=4):
        super().__init__()
        
        self.motion_dims = [256, 512, 512, 512]  # Output of LatentTokenDecoder - "512 channels for most of the layers, switching to 256 channels for the final three layer" (m⁴, m³, m², m¹) 
        self.feature_dims = [256, 512, 512, 512]  # This should match the output of DenseFeatureEncoder


        self.feature_extractor = ResNetFeatureExtractor(output_channels=self.feature_dims)
        self.latent_token_encoder = LatentTokenEncoder() 
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim)



        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            self.implicit_motion_alignment.append(ImplicitMotionAlignment(
                feature_dim=feature_dim,
                motion_dim=motion_dim,
                heads=8,
                dim_head=64,
                mlp_dim=1024
            ))



        self.frame_decoder = FrameDecoder()

        # StyleGAN2-like additions
        self.mapping_network = MappingNetwork(latent_dim, latent_dim, depth=8)
        self.noise_injection = NoiseInjection()
        self.style_mixer = StyleMixer()
   

    def forward(self, x_current, x_reference, style_mix_prob=0.9, noise_magnitude=0.1):
        return self._forward_impl(x_current, x_reference, style_mix_prob, noise_magnitude, is_inference=False)

    def inference(self, x_current, x_reference):
        return self._forward_impl(x_current, x_reference, style_mix_prob=0, noise_magnitude=0, is_inference=True)

    def _forward_impl(self, x_current, x_reference, style_mix_prob, noise_magnitude, is_inference):
        # Dense feature encoding
        f_r = self.feature_extractor(x_reference)
        

        # Latent token encoding
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)

        # StyleGAN2-like mapping network
        t_r = self.mapping_network(t_r)
        t_c = self.mapping_network(t_c)

        if not is_inference:
            # Add noise to latent tokens
            t_r = self.noise_injection(t_r, noise_magnitude * torch.randn_like(t_r))
            t_c = self.noise_injection(t_c, noise_magnitude * torch.randn_like(t_c))

            # Apply style mixing
            t_c, t_r = self.style_mixer(t_c, t_r, style_mix_prob)

        # Latent token decoding
        m_r = checkpoint(self.latent_token_decoder, t_r)
        m_c = checkpoint(self.latent_token_decoder, t_c)
        for r in m_r:
            debug_print(f"m_r:{r.shape}")

        for c in m_c:
            debug_print(f"m_c:{c.shape}")

        # Implicit motion alignment with noise injection
        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            if not is_inference:
                m_r_i = self.noise_injection(m_r[i], noise=noise_magnitude * torch.randn_like(m_r[i]))
                m_c_i = self.noise_injection(m_c[i], noise=noise_magnitude * torch.randn_like(m_c[i]))
            else:
                m_r_i = m_r[i]
                m_c_i = m_c[i]
            align_layer = self.implicit_motion_alignment[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)

        # Frame decoding
        reconstructed_frame = self.frame_decoder(aligned_features)

        if is_inference:
            return reconstructed_frame
        else:
            return reconstructed_frame, {
                'dense_features': f_r,
                'latent_tokens': (t_c, t_r),
                'motion_features': (m_c, m_r),
                'aligned_features': aligned_features
            }





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



        