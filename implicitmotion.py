import torch
import torch.nn as nn
from einops import rearrange
import matha
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py

'''
ImplicitMotionAlignment
    The module consists of two parts: the cross-attention module and the transformer. 

Cross-Attention Module
    The cross-attention module is implemented using the scaled dot-product cross-attention mechanism. 
    This module takes as input the motion features ml_c and ml_r as the queries (Q) and keys (K), respectively, 
    and the appearance features fl_r as the values (V).

    Flattening and Positional Embeddings
    Before computing the attention weights, the input features are first flattened to a 1D vector. BxCx(HxW)
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
    def __init__(self, feature_dim, num_layers, num_heads, mlp_dim, image_size=256, patch_size=16):
        super().__init__()
        
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        
        h, w = self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]
        num_patches = h * w
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * feature_dim, feature_dim)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        self.cross_attention = CrossAttention(feature_dim, num_heads)
        self.transformer = TransformerBlock(feature_dim, num_layers, num_heads, feature_dim // num_heads, mlp_dim)
        
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, m_c, m_r, fl_r):
        # Process appearance features
        x = self.to_patch_embedding(fl_r)
        b, n, _ = x.shape
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        # Process motion features
        m_c = self.to_patch_embedding(m_c)
        m_r = self.to_patch_embedding(m_r)
        
        # Apply cross-attention
        x = self.cross_attention(x, m_c, m_r)
        
        # Apply transformer
        x = self.transformer(x)
        x = x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, m_c, m_r):
        b, n, _, h = *x.shape, self.num_heads
        
        q = self.to_q(x)
        k = self.to_k(m_c)
        v = self.to_v(m_r)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# official paper code 
# class CrossAttention(nn.Module):
#     def __init__(self,feature_dim):
#         super().__init__()

#         self.q_pos_embedding = posemb_sincos_2d(
#             h = 256 ,
#             w = 256 ,
#             dim = feature_dim,
#         ) 

#         self.k_pos_embedding = posemb_sincos_2d(
#             h = 256 ,
#             w = 256 ,
#             dim = feature_dim,
#         ) 

#     # motion current m_c, motion reference m_r, appearance feature f_r =  queries, keys, values
#     def forward(self, queries, keys, values):
#         # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
#         q = torch.flatten(queries, start_dim=2).transpose(-1, -2)
#         q = q + self.q_pos_embedding  # (b, dim_spatial, dim_qk)

#         # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
#         k = torch.flatten(keys, start_dim=2).transpose(-1, -2)
#         k = k + self.k_pos_embedding  # (b, dim_spatial, dim_qk)
#         # (b, dim_v, h, w) -> (b, dim_v, dim_spatial) -> (b, dim_spatial, dim_v)
#         v = torch.flatten(values, start_dim=2).transpose(-1, -2)

#         # (b, dim_spatial, dim_qk) * (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_spatial)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)  # (b, dim_spatial, dim_spatial)  softmax

#         # (b, dim_spatial, dim_spatial) * (b, dim_spatial, dim_v) -> (b, dim_spatial, dim_v)
#         out = torch.matmul(attn, v)
#         return out



# class TransformerBlock(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 MultiHeadAttention(dim, heads = heads, dim_head = dim_head),
#                 FeedForward(dim, mlp_dim)
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return self.norm(x)


# class MultiHeadAttention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         self.norm = nn.LayerNorm(dim)

#         self.attend = nn.Softmax(dim = -1)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
#         self.to_out = nn.Linear(inner_dim, dim, bias = False)

#     def forward(self, x):
#         x = self.norm(x)

#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim=None):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#     def forward(self, x):
#         return self.net(x)