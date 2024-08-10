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
            in_dim = input_dims[i]
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
            nn.Conv2d(input_dims[-1], output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        # Learnable positional encoding
        self.pos_encoding = PositionalEncoding2D(input_dims[0])

    def forward(self, features):
        # Reshape and reverse features list
        reshaped_features = self.reshape_features(features)[::-1]
        
        x = reshaped_features[0]  # Start with the smallest feature map
        x = self.pos_encoding(x)  # Add positional encoding
        
        for i, (upconv, feat_block) in enumerate(zip(self.upconv_blocks, self.feat_blocks)):
            x = upconv(x)
            feat = feat_block(reshaped_features[i+1])
            
            if self.use_attention:
                feat = self.attention_layers[i](feat)
            
            x = torch.cat([x, feat], dim=1)
        
        x = self.final_conv(x)
        return x

    def reshape_features(self, features):
        reshaped = []
        for feat in features:
            if len(feat.shape) == 3:  # (batch, hw, channels)
                b, hw, c = feat.shape
                h = w = int(math.sqrt(hw))
                reshaped_feat = feat.permute(0, 2, 1).view(b, c, h, w)
            else:  # Already in (batch, channels, height, width) format
                reshaped_feat = feat
            reshaped.append(reshaped_feat)
        return reshaped

class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        return self.upsample(self.conv(x))

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
        return F.relu(x + self.conv(x))

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
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
        tensor = tensor.permute(0, 2, 3, 1)
        _, h, w, _ = tensor.shape
        pos_x, pos_y = torch.meshgrid(torch.arange(w), torch.arange(h))
        pos_x = pos_x.to(tensor.device).float()
        pos_y = pos_y.to(tensor.device).float()
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((h, w, self.channels * 2)).to(tensor.device)
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:] = emb_y
        emb = emb[None, :, :, :self.org_channels].permute(0, 3, 1, 2)
        return tensor.permute(0, 3, 1, 2) + emb