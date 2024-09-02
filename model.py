import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
# from vit_scaled import ImplicitMotionAlignment #- SLOW but reduces memory 2x/3x
# from vit_mlgffn import ImplicitMotionAlignment
# from vit_xformers import ImplicitMotionAlignment
from vit import ImplicitMotionAlignment
from resblocks import  FeatResBlock,UpConvResBlock,DownConvResBlock
from lia_resblocks import StyledConv,EqualConv2d,EqualLinear,ResBlock # these are correct https://github.com/hologerry/IMF/issues/4  "You can refer to this repo https://github.com/wyhsirius/LIA/ for StyleGAN2 related code, such as Encoder, Decoder."


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import math
# from common import DownConvResBlock,UpConvResBlock
import colored_traceback.auto # makes terminal show color coded output when crash
import random

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


# keep everything in 1 class to allow copying / pasting into claude / chatgpt


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
    def __init__(self, in_channels=3, output_channels=[128, 256, 512, 512], initial_channels=64):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        
        # Add an initial down block that doesn't change the number of channels
        self.down_blocks.append(DownConvResBlock(current_channels, current_channels))
        
        # Add down blocks for each specified output channel
        for out_channels in output_channels:
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

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
    def __init__(self, initial_channels=64, output_channels=[128, 256, 512, 512], dm=32):
        super(LatentTokenEncoder, self).__init__()

        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        # Dynamically create ResBlocks
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels,out_channels))
            in_channels = out_channels

        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)

        # 4× EqualLinear layers
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        
        self.final_linear = EqualLinear(output_channels[-1], dm)

    def forward(self, x):
        debug_print(f"LatentTokenEncoder input shape: {x.shape}")

        x = self.activation(self.conv1(x))
        debug_print(f"After initial conv and activation: {x.shape}")
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            debug_print(f"After res_block {i+1}: {x.shape}")
        
        x = self.equalconv(x)
        debug_print(f"After equalconv: {x.shape}")
        
        # Global average pooling
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        debug_print(f"After global average pooling: {x.shape}")
        
        for i, linear_layer in enumerate(self.linear_layers):
            x = self.activation(linear_layer(x))
            debug_print(f"After linear layer {i+1}: {x.shape}")
        
        x = self.final_linear(x)
        debug_print(f"Final output: {x.shape}")
        
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
    def __init__(self,gradient_scale=1):
        super().__init__()
        
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(512, 512),
            UpConvResBlock(1024, 512),
            UpConvResBlock(768, 256),
            UpConvResBlock(384, 128),
            UpConvResBlock(128, 64)
        ])
        
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(512) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(256) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(128) for _ in range(3)])
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(3),
            nn.Sigmoid()
        )
        
     

    def forward(self, features):
        debug_print(f"🎒 FrameDecoder input shapes")
        for f in features:
            debug_print(f"f:{f.shape}")
        # Reshape features
        reshaped_features = []
        for feat in features:
            if len(feat.shape) == 3:  # (batch, hw, channels)
                b, hw, c = feat.shape
                h = w = int(math.sqrt(hw))
                reshaped_feat = feat.permute(0, 2, 1).view(b, c, h, w)
            else:  # Already in (batch, channels, height, width) format
                reshaped_feat = feat
            reshaped_features.append(reshaped_feat)

        debug_print(f"Reshaped features: {[f.shape for f in reshaped_features]}")

        x = reshaped_features[-1]  # Start with the smallest feature map
        debug_print(f"    Initial x shape: {x.shape}")
        
        for i in range(len(self.upconv_blocks)):
            debug_print(f"\n    Processing upconv_block {i+1}")
            x = self.upconv_blocks[i](x)
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
    def __init__(self,use_resnet_feature=False,use_mlgffn=False,use_skip=False,use_enhanced_generator=False, latent_dim=32, base_channels=64, num_layers=4, noise_level=0.1, style_mix_prob=0.5):
        super().__init__()
        

        self.latent_token_encoder = LatentTokenEncoder(
            initial_channels=64,
            output_channels=[256, 256, 512, 512,512, 512],
            dm=32
        ) 
        
        self.latent_token_decoder = LatentTokenDecoder() # seems ok -  this should be the values for cross attention
        self.feature_dims = [128, 256, 512, 512] # 128 should be 32 channels.
        self.spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8)]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels= self.feature_dims)

        self.motion_dims = [256, 512, 512, 512]
        
        # Initialize a list of ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            spatial_dim = self.spatial_dims[i]
            alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim,spatial_dim=spatial_dim, )
            self.implicit_motion_alignment.append(alignment_module)
       

        self.frame_decoder = FrameDecoder()  
        self.noise_level = noise_level
        self.style_mix_prob = style_mix_prob

        # StyleGAN2-like additions
        self.mapping_network = MappingNetwork(latent_dim, latent_dim, depth=8)
        self.noise_injection = NoiseInjection()

        # self.apply(self.frame_decoder)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)       

    def add_noise(self, tensor):
        return tensor + torch.randn_like(tensor) * self.noise_level

    def style_mixing(self, t_c, t_r):
        device = t_c.device
        if random.random() < self.style_mix_prob:
            batch_size, token_dim = t_c.size()
            mix_mask = (torch.rand(batch_size, token_dim, device=device) < 0.5).float()
            t_c_mixed = t_c * mix_mask + t_r * (1 - mix_mask)
            t_r_mixed = t_r * mix_mask + t_c * (1 - mix_mask)
            return t_c_mixed, t_r_mixed
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
class IMFPatchDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(IMFPatchDiscriminator, self).__init__()
        
          
        self.scale1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0))
        )
        
        self.scale2 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=1, stride=1, padding=0))
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



        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_instance_norm=True):
        super(ConvBlock, self).__init__()
        layers = [
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if use_instance_norm:
            layers.insert(1, nn.InstanceNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3):
        super(PatchDiscriminator, self).__init__()
        sequence = [ConvBlock(input_nc, ndf, use_instance_norm=False)]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [ConvBlock(ndf * nf_mult_prev, ndf * nf_mult)]

        sequence += [
            ConvBlock(ndf * nf_mult, ndf * nf_mult, stride=1),
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=3):
        super(MultiScalePatchDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers

        for i in range(num_D):
            subnetD = PatchDiscriminator(input_nc, ndf, n_layers)
            setattr(self, f'scale_{i}', subnetD)

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        result = []
        input_downsampled = input
        for i in range(self.num_D):
            model = getattr(self, f'scale_{i}')
            result.append(self.singleD_forward(model, input_downsampled))
            if i != self.num_D - 1:
                input_downsampled = F.avg_pool2d(input_downsampled, kernel_size=3, stride=2, padding=1, count_include_pad=False)
        return result

    def get_scale_params(self):
        return [getattr(self, f'scale_{i}').parameters() for i in range(self.num_D)]

        
        