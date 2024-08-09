import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]

class ScaledDotProductAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)

        # Use PyTorch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        return self.to_out(attn_output)

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, depth=4, heads=8, dim_head=64, dropout=0.1):
        super().__init__()
        self.cross_attention = CrossAttentionModule(feature_dim, motion_dim, heads, dim_head, dropout)
        self.attention_blocks = nn.ModuleList([
            ScaledDotProductAttentionBlock(feature_dim, heads, dim_head, dropout)
            for _ in range(depth)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(depth)])

    def forward(self, ml_c, ml_r, fl_r, return_embeddings=False):
        embeddings = []
        
        # Cross-attention module
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        if return_embeddings:
            embeddings.append(("After Cross-Attention", V_prime.detach().cpu()))

        # Attention blocks
        B, C, H, W = V_prime.shape
        V_prime = V_prime.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        
        for i, (attn, norm) in enumerate(zip(self.attention_blocks, self.norms)):
            V_prime = V_prime + attn(norm(V_prime))
            if return_embeddings:
                embeddings.append((f"After Attention Block {i}", V_prime.detach().cpu()))
        
        V_prime = V_prime.permute(0, 2, 1).view(B, C, H, W)
        
        if return_embeddings:
            return V_prime, embeddings
        else:
            return V_prime
    @staticmethod
    def visualize_embeddings(embeddings, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (title, emb) in enumerate(embeddings):
            if i >= len(axes):
                break
            
            emb = emb.contiguous().cpu().reshape(-1, emb.shape[-1])
            
            if emb.shape[0] > 10000:
                indices = np.random.choice(emb.shape[0], 10000, replace=False)
                emb = emb[indices]

            tsne = TSNE(n_components=2, random_state=42)
            emb_2d = tsne.fit_transform(emb.numpy())

            axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, motion_dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(motion_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(motion_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(feature_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, feature_dim)
        self.dropout = nn.Dropout(dropout)

        self.pos_encoding_q = PositionalEncoding(motion_dim)
        self.pos_encoding_k = PositionalEncoding(motion_dim)

    def forward(self, ml_c, ml_r, fl_r):
        B, C_m, H, W = ml_c.shape
        _, C_f, _, _ = fl_r.shape

        # Flatten inputs
        ml_c = ml_c.view(B, C_m, H*W).permute(0, 2, 1)  # (B, H*W, C_m)
        ml_r = ml_r.view(B, C_m, H*W).permute(0, 2, 1)  # (B, H*W, C_m)
        fl_r = fl_r.view(B, C_f, H*W).permute(0, 2, 1)  # (B, H*W, C_f)

        # Generate and add positional encodings
        ml_c = ml_c + self.pos_encoding_q(ml_c)
        ml_r = ml_r + self.pos_encoding_k(ml_r)

        # Compute Q, K, V
        q = self.to_q(ml_c).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(ml_r).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(fl_r).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H*W, self.heads * self.dim_head)
        output = self.to_out(attn_output)
        output = output.permute(0, 2, 1).view(B, C_f, H, W)
        return output

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    dropout = 0.1

    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    model = ImplicitMotionAlignment(feature_dim, motion_dim, depth, heads, dim_head, dropout).to(device)

    with torch.no_grad():
        output, embeddings = model(ml_c, ml_r, fl_r)

    model.visualize_embeddings(embeddings, "pytorch_scaled_attention_embeddings_visualization.png")