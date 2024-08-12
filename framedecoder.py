import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from resblock import UpConvResBlock,FeatResBlock

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
class EnhancedFrameDecoder(nn.Module):
    def __init__(self, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        
        # Adjusted channel counts to handle concatenated features
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(512, 512),
            UpConvResBlock(1024, 512),  # 512 + 512 = 1024 input channels
            UpConvResBlock(768, 256),   # 512 + 256 = 768 input channels
            UpConvResBlock(384, 128),   # 256 + 128 = 384 input channels
            UpConvResBlock(256, 64)     # 128 + 128 = 256 input channels
        ])
        
        self.feat_blocks = nn.ModuleList([
            nn.Sequential(*[FeatResBlock(512) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(256) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(128) for _ in range(3)]),
            nn.Sequential(*[FeatResBlock(128) for _ in range(3)])  # Changed from 64 to 128
        ])
        
        if use_attention:
            self.attention_layers = nn.ModuleList([
                SelfAttention(512),
                SelfAttention(256),
                SelfAttention(128),
                SelfAttention(128)  # Changed from 64 to 128
            ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
        self.pos_encoding = PositionalEncoding2D(512)

    def forward(self, features):
        debug_print(f"🎲 EnhancedFrameDecoder input features shapes: {[f.shape for f in features]}")
        
        x = features[-1]  # Start with the smallest feature map
        debug_print(f"Initial x shape: {x.shape}")
        
        x = self.pos_encoding(x)  # Add positional encoding
        debug_print(f"After positional encoding, x shape: {x.shape}")
        
        for i in range(len(self.upconv_blocks)):
            debug_print(f"\nProcessing upconv_block {i+1}")
            x = self.upconv_blocks[i](x)
            debug_print(f"After upconv_block {i+1}: {x.shape}")
            
            if i < len(features) - 1:  # Process all encoder features except the last one
                debug_print(f"Processing feat_block {i+1}")
                feat_input = features[-(i+2)]
                debug_print(f"feat_block {i+1} input shape: {feat_input.shape}")
                feat = self.feat_blocks[i](feat_input)
                debug_print(f"feat_block {i+1} output shape: {feat.shape}")
                
                if self.use_attention:
                    feat = self.attention_layers[i](feat)
                    debug_print(f"After attention {i+1}, feat shape: {feat.shape}")
                
                debug_print(f"Concatenating: x {x.shape} and feat {feat.shape}")
                x = torch.cat([x, feat], dim=1)
                debug_print(f"After concatenation: {x.shape}")
        
        debug_print("\nApplying final convolution")
        x = self.final_conv(x)
        debug_print(f"EnhancedFrameDecoder final output shape: {x.shape}")

        return x
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        debug_print(f"SelfAttention input shape: {x.shape}")
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        debug_print(f"SelfAttention output shape: {out.shape}")
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
        debug_print(f"PositionalEncoding2D input shape: {tensor.shape}")
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
        debug_print(f"PositionalEncoding2D output shape: {out.shape}")
        return out