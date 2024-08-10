import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedFrameDecoder(nn.Module):
    def __init__(self, input_dims=[512, 512, 256, 128], output_dim=3, use_attention=True):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Dynamic creation of upconv and feat blocks
        self.upconv_blocks = nn.ModuleList()
        self.feat_blocks = nn.ModuleList()
        
        for i in range(len(input_dims) - 1):
            in_dim = input_dims[i] * 2 if i > 0 else input_dims[i]  # Double the input channels for concatenation
            out_dim = input_dims[i+1]
            self.upconv_blocks.append(UpConvResBlock(in_dim, out_dim))
            self.feat_blocks.append(nn.Sequential(*[FeatResBlock(out_dim) for _ in range(3)]))
        
        # Add attention layers if specified
        if use_attention:
            self.attention_layers = nn.ModuleList([
                SelfAttention(dim) for dim in input_dims[1:]
            ])
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_dims[-1] * 2, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Learnable positional encoding
        self.pos_encoding = PositionalEncoding2D(input_dims[0])

    def forward(self, features):
        print(f"EnhancedFrameDecoder input features shapes: {[f.shape for f in features]}")
        
        # Reshape and reverse features list
        reshaped_features = self.reshape_features(features)[::-1]
        print(f"Reshaped features shapes: {[f.shape for f in reshaped_features]}")
        
        x = reshaped_features[0]  # Start with the smallest feature map
        print(f"Initial x shape: {x.shape}")
        
        x = self.pos_encoding(x)  # Add positional encoding
        print(f"After positional encoding, x shape: {x.shape}")
        
        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x)
            print(f"After upconv {i}, x shape: {x.shape}")
            
            if i + 1 < len(reshaped_features):
                feat = self.feat_blocks[i](reshaped_features[i+1])
                print(f"Feature {i} shape: {feat.shape}")
                
                if self.use_attention:
                    feat = self.attention_layers[i](feat)
                    print(f"After attention {i}, feat shape: {feat.shape}")
                
                x = torch.cat([x, feat], dim=1)
                print(f"After concatenation {i}, x shape: {x.shape}")
        
        x = self.final_conv(x)
        print(f"Final output shape: {x.shape}")
        return x

    def reshape_features(self, features):
        reshaped = []
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:  # (batch, hw, channels)
                b, hw, c = feat.shape
                h = w = int(math.sqrt(hw))
                reshaped_feat = feat.permute(0, 2, 1).view(b, c, h, w)
            else:  # Already in (batch, channels, height, width) format
                reshaped_feat = feat
            reshaped.append(reshaped_feat)
            print(f"Reshaped feature {i} shape: {reshaped_feat.shape}")
        return reshaped

class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        print(f"UpConvResBlock input shape: {x.shape}")
        x = self.upsample(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        print(f"UpConvResBlock output shape: {x.shape}")
        return x

class FeatResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        print(f"FeatResBlock input shape: {x.shape}")
        residual = self.conv(x)
        print(f"FeatResBlock residual shape: {residual.shape}")
        out = F.relu(x + residual)
        print(f"FeatResBlock output shape: {out.shape}")
        return out

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        print(f"SelfAttention input shape: {x.shape}")
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        print(f"SelfAttention output shape: {out.shape}")
        return out

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.org_channels = channels
        channels = int(math.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        print(f"PositionalEncoding2D input shape: {tensor.shape}")
        _, _, h, w = tensor.shape
        pos_x, pos_y = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='ij')
        pos_x = pos_x.to(tensor.device).float()
        pos_y = pos_y.to(tensor.device).float()
        
        sin_inp_x = pos_x.unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        sin_inp_y = pos_y.unsqueeze(-1) @ self.inv_freq.unsqueeze(0)
        
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(0)
        
        emb = torch.cat((emb_x, emb_y), dim=-1).permute(0, 3, 1, 2)
        out = tensor + emb[:, :self.org_channels, :, :]
        print(f"PositionalEncoding2D output shape: {out.shape}")
        return out