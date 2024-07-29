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

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# keep everything in 1 class to allow copying / pasting into claude / chatgpt



class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(out_channels)
        self.feat_res_block2 = FeatResBlock(out_channels)

    def forward(self, x):
        # print(f"UpConvResBlock input shape: {x.shape}")
        out = self.upsample(x)
        # print(f"After upsample: {out.shape}")
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        # print(f"After conv1, bn1, relu: {out.shape}")
        out = self.conv2(out)
        # print(f"After conv2: {out.shape}")
        out = self.feat_res_block1(out)
        # print(f"After feat_res_block1: {out.shape}")
        out = self.feat_res_block2(out)
        # print(f"UpConvResBlock output shape: {out.shape}")
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
        # print(f"DownConvResBlock input shape: {x.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print(f"After conv1, bn1, relu: {out.shape}")
        out = self.avgpool(out)
        # print(f"After avgpool: {out.shape}")
        out = self.conv2(out)
        # print(f"After conv2: {out.shape}")
        out = self.feat_res_block1(out)
        # print(f"After feat_res_block1: {out.shape}")
        out = self.feat_res_block2(out)
        # print(f"DownConvResBlock output shape: {out.shape}")
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
        print(f"⚾ DenseFeatureEncoder input shape: {x.shape}")
        features = []
        x = self.initial_conv(x)
        # print(f"After initial conv: {x.shape}")
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            # print(f"After down_block {i+1}: {x.shape}")
            if i < 4:
                features.append(x)
        features.append(x)
        # print(f"DenseFeatureEncoder output shapes: {[f.shape for f in features]}")
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
        # print(f"LatentTokenEncoder input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(f"After first conv, bn, relu: {x.shape}")
        x = self.resblock1(x)
        # print(f"After resblock1: {x.shape}")
        x = self.resblock2(x)
        # print(f"After resblock2: {x.shape}")
        x = self.resblock3(x)
        # print(f"After resblock3: {x.shape}")
        x = self.resblock4(x)
        # print(f"After resblock4: {x.shape}")
        x = self.equal_conv(x)
        # print(f"After equal_conv: {x.shape}")
        x = self.adaptive_pool(x)
        # print(f"After adaptive_pool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After flatten: {x.shape}")
        t = self.fc_layers(x)
        # print(f"LatentTokenEncoder output shape: {t.shape}")
        return t

    
class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=False, style_dim=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.style_mod = nn.Linear(style_dim, in_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, style):
        # print(f"StyleConv input shape: x: {x.shape}, style: {style.shape}")
        style = self.style_mod(style).unsqueeze(2).unsqueeze(3)
        # print(f"After style modulation: {style.shape}")
        x = x * style
        # print(f"After multiplication: {x.shape}")
        x = self.conv(x)
        # print(f"After conv: {x.shape}")
        x = self.upsample(x)
        # print(f"After upsample: {x.shape}")
        x = self.activation(x)
        # print(f"StyleConv output shape: {x.shape}")
        return x


'''
LatentTokenDecoder
It starts with a constant input (self.const), as shown at the top of the image.
It uses a series of StyleConv layers, which apply style modulation based on the input latent token t.
Some StyleConv layers perform upsampling (indicated by ↑2 in the image).
The network maintains 512 channels for most of the layers, switching to 256 channels for the final three layers.
It outputs multiple feature maps (m⁴, m³, m², m¹) as shown in the image. These are collected in the features list and returned in reverse order to match the notation in the image.
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
        print(f"🎑 LatentTokenDecoder input shape: {t.shape}")
        x = self.const.repeat(t.shape[0], 1, 1, 1)
        print(f"After const: {x.shape}")
        features = []
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            print(f"After style_conv {i+1}: {x.shape}")
            if i in [3, 6, 9, 12]:
                features.append(x)
        print(f"LatentTokenDecoder output shapes: {[f.shape for f in features[::-1]]}")
        return features[::-1]  # Return features in order [m4, m3, m2, m1]


class MemoryEfficientAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim * 3, dim * 3, bias=False)  # Changed input dimension
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # Diagnostics: print input shape
        print(f"Input shape: {x.shape}")

        # Apply qkv transformation
        qkv = self.qkv(x)
        # Diagnostics: print shape after qkv transformation
        print(f"Shape after qkv transformation: {qkv.shape}")

        # Reshape qkv and permute dimensions
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // 3 // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Diagnostics: print shapes of q, k, v
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Diagnostics: print shape of attention scores
        print(f"attn shape: {attn.shape}")

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C // 3)
        # Diagnostics: print shape after applying attention
        print(f"Shape after attention application: {x.shape}")

        # Project the output
        x = self.proj(x)
        # Diagnostics: print final output shape
        print(f"Final output shape: {x.shape}")

        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MemoryEfficientAttention(dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        def custom_forward(x):
            x = x + self.attention(self.layer_norm1(x))
            x = x + self.feed_forward(self.layer_norm2(x))
            return x
        return checkpoint(custom_forward, x)

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, num_heads=8, num_transformer_blocks=4, chunk_size=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(motion_dim, feature_dim)
        self.k_proj = nn.Linear(motion_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.p_q = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.p_k = nn.Parameter(torch.randn(1, 1, feature_dim))

        self.attention = MemoryEfficientAttention(feature_dim, num_heads)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads) for _ in range(num_transformer_blocks)
        ])

    def forward(self, m_c, m_r, f_r):
        print(f"m_c shape: {m_c.shape}")
        print(f"m_r shape: {m_r.shape}")
        print(f"f_r shape: {f_r.shape}")

        b, c_m, h_m, w_m = m_c.shape
        _, c_f, h_f, w_f = f_r.shape
        m_c = m_c.view(b, self.motion_dim, -1).permute(0, 2, 1)
        m_r = m_r.view(b, self.motion_dim, -1).permute(0, 2, 1)
        f_r = f_r.view(b, self.feature_dim, -1).permute(0, 2, 1)

        print(f"Reshaped m_c shape: {m_c.shape}")
        print(f"Reshaped m_r shape: {m_r.shape}")
        print(f"Reshaped f_r shape: {f_r.shape}")

        q = self.q_proj(m_c) + self.p_q
        k = self.k_proj(m_r) + self.p_k
        v = self.v_proj(f_r)

        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")

        chunks = []
        for i in range(0, q.size(1), self.chunk_size):
            q_chunk = q[:, i:i+self.chunk_size]
            k_chunk = k[:, i:i+self.chunk_size]
            v_chunk = v[:, i:i+self.chunk_size]
            
            print(f"Chunk {i // self.chunk_size}:")
            print(f"q_chunk shape: {q_chunk.shape}")
            print(f"k_chunk shape: {k_chunk.shape}")
            print(f"v_chunk shape: {v_chunk.shape}")
            
            if v_chunk.size(1) == 0:
                print("Skipping empty v_chunk")
                continue
            elif v_chunk.size(1) < q_chunk.size(1):
                v_chunk = v_chunk.repeat(1, (q_chunk.size(1) + v_chunk.size(1) - 1) // v_chunk.size(1), 1)[:, :q_chunk.size(1)]
            
            attn_input = torch.cat([q_chunk, k_chunk, v_chunk], dim=-1)
            print(f"attn_input shape: {attn_input.shape}")
            
            attn_output = self.attention(attn_input)
            print(f"attn_output shape: {attn_output.shape}")
            
            chunks.append(attn_output)
        
        x = torch.cat(chunks, dim=1)
        print(f"Shape after concatenation: {x.shape}")

        for block in self.transformer_blocks:
            x = block(x)

        output = x.permute(0, 2, 1).view(b, self.feature_dim, h_f, w_f)
        print(f"Final output shape: {output.shape}")

        return output



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
        print(f"FrameDecoder input shapes: {[f.shape for f in features]}")

        f4, f3, f2, f1 = features
        
        x = self.upconv_blocks[0](f4)
        print(f"After first upconv: {x.shape}")

        x = torch.cat([x, self.feat_blocks[0](f3)], dim=1)
        print(f"After first concat: {x.shape}")

        x = self.upconv_blocks[1](x)
        x = torch.cat([x, self.feat_blocks[1](f2)], dim=1)
        
        x = self.upconv_blocks[2](x)
        x = torch.cat([x, self.feat_blocks[2](f1)], dim=1)
        
        x = self.upconv_blocks[3](x)
        x = self.upconv_blocks[4](x)
        
        x_c = self.final_conv(x)
        print(f"FrameDecoder output shape: {x_c.shape}")

        return x_c


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
        # print(f"ResBlock input shape: {x.shape}")
        # print(f"ResBlock parameters: in_channels={self.in_channels}, out_channels={self.out_channels}, downsample={self.downsample}")

        residual = self.shortcut(x)
        # print(f"After shortcut: {residual.shape}")
        
        out = self.conv1(x)
        # print(f"After conv1: {out.shape}")
        out = self.bn1(out)
        out = self.relu1(out)
        # print(f"After bn1 and relu1: {out.shape}")
        
        out = self.conv2(out)
        # print(f"After conv2: {out.shape}")
        out = self.bn2(out)
        # print(f"After bn2: {out.shape}")
        
        out += residual
        # print(f"After adding residual: {out.shape}")
        
        out = self.relu2(out)
        # print(f"ResBlock output shape: {out.shape}")
        
        return out

class FeatResBlock(ResBlock):
    def __init__(self, channels):
        super().__init__(channels, channels, downsample=False)
    
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
        self.feature_dims = [64, 128, 256, 512]
        
        # Create ImplicitMotionAlignment modules for each layer
        self.implicit_motion_alignment = nn.ModuleList([
            ImplicitMotionAlignment(
                feature_dim=self.feature_dims[i],
                motion_dim=256 if i == 3 else 512
            ) for i in range(num_layers)
        ])


    def forward(self, x_current, x_reference):
        print(f"IMF input shapes: x_current: {x_current.shape}, x_reference: {x_reference.shape}")

        # Encode reference frame
        f_r = self.dense_feature_encoder(x_reference)
        
        # Encode latent tokens
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        print(f"Encoded tokens shapes: t_r: {t_r.shape}, t_c: {t_c.shape}")

        # Decode motion features
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        
        # Align features
        aligned_features = []

        # Get the number of layers we're working with
        num_layers = len(self.implicit_motion_alignment)

        # Iterate through each layer
        for i in range(num_layers):
            # Extract the corresponding features and alignment layer for this iteration
            f_r_i = f_r[i]
            m_r_i = m_r[i]
            m_c_i = m_c[i]
            align_layer = self.implicit_motion_alignment[i]
            
            # Print input shapes for debugging
            print(f"Layer {i+1} input shapes:")
            print(f"  f_r_i shape: {f_r_i.shape}")
            print(f"  m_r_i shape: {m_r_i.shape}")
            print(f"  m_c_i shape: {m_c_i.shape}")
            
            # Perform the alignment
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            
            # Add the aligned feature to our list
            aligned_features.append(aligned_feature)
            
            # Print the shape of the aligned feature
            print(f"Aligned feature {i+1} shape: {aligned_feature.shape}")
            print("--------------------")

        # At this point, aligned_features contains all the aligned features

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
        self.dense_feature_encoder = DenseFeatureEncoder()
        self.latent_token_encoder = LatentTokenEncoder(latent_dim=latent_dim)
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim, const_dim=base_channels)
        
        self.feature_dims = [64, 128, 256, 512, 512]
        
        self.implicit_motion_alignment = nn.ModuleList([
            ImplicitMotionAlignment(
                feature_dim=self.feature_dims[i],
                motion_dim=base_channels // (2 ** i) if i < 4 else base_channels // 8
            ) for i in range(num_layers)
        ])
        
        self.frame_decoder = FrameDecoder()

    def forward(self, x_current, x_reference):
        # Enable gradient computation for input tensors
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()
        
        # Dense feature encoding
        f_r = self.dense_feature_encoder(x_reference)
        
        # Latent token encoding
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)
        
        # Latent token decoding
        m_r = self.latent_token_decoder(t_r)
        m_c = self.latent_token_decoder(t_c)
        
        # Implicit motion alignment
        aligned_features = []
        # Get the number of layers we're working with
        num_layers = len(self.implicit_motion_alignment)

        print(f"🎣 IMF - Number of layers: {num_layers}")
        print(f"f_r shapes: {[f.shape for f in f_r]}")
        print(f"m_r shapes: {[m.shape for m in m_r]}")
        print(f"m_c shapes: {[m.shape for m in m_c]}")

        # Iterate through each layer
        for i in range(num_layers):
            # Extract the corresponding features and alignment layer for this iteration
            f_r_i = f_r[i]
            m_r_i = m_r[i]
            m_c_i = m_c[i]
            align_layer = self.implicit_motion_alignment[i]
            
            print(f"\nProcessing layer {i+1}:")
            print(f"  f_r_i shape: {f_r_i.shape}")
            print(f"  m_r_i shape: {m_r_i.shape}")
            print(f"  m_c_i shape: {m_c_i.shape}")
            
            # Perform the alignment
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            
            # Add the aligned feature to our list
            aligned_features.append(aligned_feature)
            
            print(f"  Aligned feature shape: {aligned_feature.shape}")

        print("\nFinal aligned features shapes:")
        for i, feat in enumerate(aligned_features):
            print(f"  Layer {i+1}: {feat.shape}")
        
        # Frame decoding
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


class SNConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(SNConv2d, self).__init__(*args, **kwargs)
        self.conv = nn.utils.spectral_norm(self)

    def forward(self, input):
        return self.conv(input)
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
        print(f"PatchDiscriminator input shape: {x.shape}")

        # Scale 1
        output1 = x
        for i, layer in enumerate(self.scale1):
            output1 = layer(output1)
            print(f"Scale 1 - Layer {i} output shape: {output1.shape}")

        # Scale 2
        x_downsampled = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        print(f"Scale 2 - Downsampled input shape: {x_downsampled.shape}")
        
        output2 = x_downsampled
        for i, layer in enumerate(self.scale2):
            output2 = layer(output2)
            print(f"Scale 2 - Layer {i} output shape: {output2.shape}")

        print(f"PatchDiscriminator final output shapes: {output1.shape}, {output2.shape}")
        return [output1, output2]

# Helper function to initialize weights
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)