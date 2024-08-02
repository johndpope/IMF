import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0)]

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H*W).permute(2, 0, 1)
        
        x_norm = self.norm1(x_reshaped)
        att_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + att_output

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        output = x_reshaped.permute(1, 2, 0).view(B, C, H, W)
        return output

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, depth=2, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(feature_dim, motion_dim, heads, dim_head)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])

    def forward(self, ml_c, ml_r, fl_r):
        # Cross-attention module
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)

        # Transformer blocks
        for block in self.transformer_blocks:
            V_prime = block(V_prime)

        return V_prime


# class CrossAttentionModule(nn.Module):
#     def __init__(self, feature_dim, motion_dim, heads, dim_head):
#         super().__init__()
#         self.heads = heads
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5

#         print(f"Initializing CrossAttentionModule with:")
#         print(f"  feature_dim: {feature_dim}")
#         print(f"  motion_dim: {motion_dim}")
#         print(f"  heads: {heads}")
#         print(f"  dim_head: {dim_head}")

#         # Adjust projection layers to match input dimensions
#         self.motion_proj = nn.Linear(motion_dim, 256)
#         self.feature_proj = nn.Linear(feature_dim, 256)

#         # Keep the rest of the layers as they were
#         self.to_q = nn.Linear(256, heads * dim_head)
#         self.to_k = nn.Linear(256, heads * dim_head)
#         self.to_v = nn.Linear(256, heads * dim_head)
#         self.to_out = nn.Linear(heads * dim_head, feature_dim)

#         print(f"Linear layer shapes:")
#         print(f"  motion_proj: in={motion_dim}, out=256")
#         print(f"  feature_proj: in={feature_dim}, out=256")
#         print(f"  to_q: in=256, out={heads * dim_head}")
#         print(f"  to_k: in=256, out={heads * dim_head}")
#         print(f"  to_v: in=256, out={heads * dim_head}")
#         print(f"  to_out: in={heads * dim_head}, out={feature_dim}")

#         self.pos_encoding_q = PositionalEncoding(256)
#         self.pos_encoding_k = PositionalEncoding(256)

#     def forward(self, ml_c, ml_r, fl_r):
#         B, C_m, H_m, W_m = ml_c.shape
#         _, C_f, H_f, W_f = fl_r.shape

#         print(f"\n💀 CrossAttentionModule forward pass:")
#         print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
#         print(f"Dimensions: B={B}, C_m={C_m}, H_m={H_m}, W_m={W_m}, C_f={C_f}, H_f={H_f}, W_f={W_f}")

#         # Flatten inputs
#         ml_c = ml_c.view(B, C_m, H_m*W_m).permute(0, 2, 1)
#         ml_r = ml_r.view(B, C_m, H_m*W_m).permute(0, 2, 1)
#         fl_r = F.interpolate(fl_r, size=(H_m, W_m), mode='bilinear', align_corners=False)
#         fl_r = fl_r.view(B, C_f, H_m*W_m).permute(0, 2, 1)

#         print(f"Flattened shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")

#         # Project inputs to 256 dimensions
#         ml_c = self.motion_proj(ml_c)
#         ml_r = self.motion_proj(ml_r)
#         fl_r = self.feature_proj(fl_r)

#         print(f"After projection: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")

#         # Generate positional encodings
#         p_q = self.pos_encoding_q(ml_c)
#         p_k = self.pos_encoding_k(ml_r)

#         print(f"Positional encoding shapes: p_q: {p_q.shape}, p_k: {p_k.shape}")

#         # Ensure positional encodings match the input dimensions
#         p_q = p_q[:ml_c.size(1), :, :]
#         p_k = p_k[:ml_r.size(1), :, :]

#         ml_c = ml_c + p_q
#         ml_r = ml_r + p_k

#         print(f"After adding positional encodings: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}")

#         # Compute Q, K, V
#         q = self.to_q(ml_c)
#         k = self.to_k(ml_r)
#         v = self.to_v(fl_r)

#         print(f"After linear projections: q: {q.shape}, k: {k.shape}, v: {v.shape}")

#         q = q.view(B, H_m*W_m, self.heads, self.dim_head)
#         k = k.view(B, H_m*W_m, self.heads, self.dim_head)
#         v = v.view(B, H_m*W_m, self.heads, self.dim_head)

#         print(f"After reshaping: q: {q.shape}, k: {k.shape}, v: {v.shape}")

#         # Transpose for attention computation
#         q = q.transpose(1, 2)
#         k = k.transpose(1, 2)
#         v = v.transpose(1, 2)

#         print(f"After transposing: q: {q.shape}, k: {k.shape}, v: {v.shape}")

#         # Compute attention weights and output
#         attention_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
#         print(f"Attention weights shape: {attention_weights.shape}")

#         V_prime = torch.matmul(attention_weights, v)
#         print(f"V_prime shape after attention: {V_prime.shape}")

#         V_prime = V_prime.transpose(1, 2).contiguous().view(B, H_m*W_m, self.heads * self.dim_head)
#         print(f"V_prime shape after reshaping: {V_prime.shape}")

#         V_prime = self.to_out(V_prime)
#         print(f"V_prime shape after final linear layer: {V_prime.shape}")

#         output = V_prime.permute(0, 2, 1).view(B, C_f, H_m, W_m)
#         print(f"Final output shape: {output.shape}")

#         return output

class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, motion_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        # Adjust dimensions based on actual input
        self.to_q = nn.Linear(motion_dim, heads * dim_head)
        self.to_k = nn.Linear(motion_dim, heads * dim_head)
        self.to_v = nn.Linear(feature_dim, heads * dim_head)
        self.to_out = nn.Linear(heads * dim_head, feature_dim)

        # Update positional encoding to match motion_dim
        self.pos_encoding_q = PositionalEncoding(motion_dim)
        self.pos_encoding_k = PositionalEncoding(motion_dim)

    def forward(self, ml_c, ml_r, fl_r):
        B, C_m, H_m, W_m = ml_c.shape
        _, C_f, H_f, W_f = fl_r.shape

        # Flatten and interpolate inputs
        ml_c = ml_c.view(B, C_m, H_m*W_m).permute(0, 2, 1)
        ml_r = ml_r.view(B, C_m, H_m*W_m).permute(0, 2, 1)
        fl_r = F.interpolate(fl_r, size=(H_m, W_m), mode='bilinear', align_corners=False)
        fl_r = fl_r.view(B, C_f, H_m*W_m).permute(0, 2, 1)

        # Add positional encodings
        # print(f"ml_c{ml_c.shape}")
        # print(f"ml_r{ml_r.shape}")
        p_k =  self.pos_encoding_q(ml_c)
        p_q = self.pos_encoding_k(ml_r)
        # print(f"p_k{ml_c.shape}")
        # print(f"p_q{ml_r.shape}")

        ml_c = ml_c + p_k
        ml_r = ml_r + p_q

        # Compute Q, K, V
        q = self.to_q(ml_c).view(B, H_m*W_m, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(ml_r).view(B, H_m*W_m, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(fl_r).view(B, H_m*W_m, self.heads, self.dim_head).transpose(1, 2)

        # Compute attention
        attention_weights = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        V_prime = torch.matmul(attention_weights, v)

        # Reshape and project output
        V_prime = V_prime.transpose(1, 2).contiguous().view(B, H_m*W_m, self.heads * self.dim_head)
        V_prime = self.to_out(V_prime)

        # Reshape to match input spatial dimensions
        output = V_prime.permute(0, 2, 1).view(B, C_f, H_m, W_m)

        return output