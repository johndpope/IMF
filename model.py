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


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"

        self.query_linear = nn.Linear(query_dim, query_dim)
        self.key_linear = nn.Linear(key_dim, query_dim)
        self.value_linear = nn.Linear(value_dim, query_dim)
        self.output_linear = nn.Linear(query_dim, value_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        print(f"🍕 Input query shape: {query.shape}")
        print(f"Input key shape: {key.shape}")
        print(f"Input value shape: {value.shape}")

        batch_size = query.size(0)

        # Linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        print(f"Query shape after linear projection: {query.shape}")
        print(f"Key shape after linear projection: {key.shape}")
        print(f"Value shape after linear projection: {value.shape}")

        # Split into heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        print(f"Query shape after splitting into heads: {query.shape}")
        print(f"Key shape after splitting into heads: {key.shape}")
        print(f"Value shape after splitting into heads: {value.shape}")

        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, value)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        # Project back to value_dim
        output = self.output_linear(context)

        print(f"Final output shape: {output.shape}")

        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Initialize the multi-head attention layer
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        def custom_forward(x):
            # Apply layer normalization before attention
            norm_x = self.layer_norm1(x)
            
            # Perform self-attention
            attn_output, _ = self.attention(norm_x, norm_x, norm_x)
            
            # Add the residual connection
            x = x + attn_output
            
            # Apply layer normalization before feed-forward network
            norm_x = self.layer_norm2(x)
            
            # Pass through feed-forward network
            ff_output = self.feed_forward(norm_x)
            
            # Add the residual connection
            x = x + ff_output
            
            return x
        
        return checkpoint(custom_forward, x)



class PositionalEncoding(nn.Module):
    def __init__(self, max_len=5000):
        super().__init__()
        self.max_len = max_len

    def forward(self, x):
        print(f"PositionalEncoding input shape: {x.shape}")
        
        batch_size, seq_len, d_model = x.size()
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        print(f"Generated positional encoding shape: {pe.shape}")
        return x + pe.to(x.device)
'''
ImplicitMotionAlignment

Cross-Attention Module
The module consists of two parts: the cross-attention module and the transformer. 
The cross-attention module is implemented using the scaled dot-product cross-attention mechanism. 
This module takes as input the motion features ml_c and ml_r as the queries (Q) and keys (K), respectively, 
and the appearance features fl_r as the values (V).

Flattening and Positional Embeddings
Before computing the attention weights, the input features are first flattened to a 1D vector. 
Then, positional embeddings Pq and Pk are added to the queries and keys, respectively. 
This is done to capture the spatial relationships between the features.

Attention Weights Computation
The attention weights are computed by taking the dot product of the queries with the keys, dividing each by √dk 
(where dk is the dimensionality of the keys), and applying a softmax function to obtain the weights on the values.

Output-Aligned Values
The output-aligned values V' are computed through matrix multiplication of the attention weights with the values.

Transformer Blocks
The output-aligned values V' are further refined using multi-head self-attention and feedforward network-based 
transformer blocks. This is done to capture the complex relationships between the features and to produce 
the final appearance features fl_c of the current frame.
'''
class ImplicitMotionAlignment(nn.Module):
    def __init__(self, motion_dim, feature_dim, num_heads=8):
        super(ImplicitMotionAlignment, self).__init__()
        query_dim = motion_dim
        key_dim = motion_dim
        value_dim = feature_dim
        
        self.cross_attention = CrossAttention(query_dim, key_dim, value_dim, num_heads)
        self.pq = PositionalEncoding(query_dim)
        self.pk = PositionalEncoding(key_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads) for _ in range(4)
        ])

    def forward(self, ml_c, ml_r, fl_r):
        print("🎮 ImplicitMotionAlignment input shapes:")
        print(f"  ml_c: {ml_c.shape}")
        print(f"  ml_r: {ml_r.shape}")
        print(f"  fl_r: {fl_r.shape}")

        # Flatten the inputs
        b, c, h, w = ml_c.shape
        ml_c = ml_c.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        ml_r = ml_r.view(b, c, -1).permute(0, 2, 1)  # (b, h*w, c)
        fl_r = fl_r.view(b, fl_r.size(1), -1).permute(0, 2, 1)  # (b, h*w, c)

        print("After flattening:")
        print(f"  ml_c: {ml_c.shape}")
        print(f"  ml_r: {ml_r.shape}")
        print(f"  fl_r: {fl_r.shape}")

        # Add positional embeddings to queries (ml_c) and keys (ml_r)
        ml_c = self.pq(ml_c)
        ml_r = self.pk(ml_r)

        print("After adding positional embeddings:")
        print(f"  ml_c: {ml_c.shape}")
        print(f"  ml_r: {ml_r.shape}")

        # Cross-attention
        v_prime = self.cross_attention(ml_c, ml_r, fl_r)
        print(f"After cross-attention: {v_prime.shape}")

        # Transformer blocks
        for i, transformer_block in enumerate(self.transformer_blocks):
            v_prime = transformer_block(v_prime)
            print(f"After transformer block {i+1}: {v_prime.shape}")

        # Reshape back to 2D
        fl_c = v_prime.permute(0, 2, 1).view(b, -1, h, w)
        print(f"Final output shape: {fl_c.shape}")

        return fl_c



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