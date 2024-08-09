import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from torch.utils.checkpoint import checkpoint


# https://medium.com/pytorch/training-compact-transformers-from-scratch-in-30-minutes-with-pytorch-ff5c21668ed5
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
    def __init__(self, motion_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim=motion_dim, heads=heads, dim_head=dim_head)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(motion_dim, heads, dim_head, mlp_dim)
            for _ in range(depth)
        ])

    def forward(self, ml_c, ml_r, fl_r):
        print(f"ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        
        for i, transformer in enumerate(self.transformer_blocks):
            V_prime = transformer(V_prime)
            print(f"Layer {i+1} output shape: {V_prime.shape}")
        
        return V_prime




    @staticmethod
    def visualize_embeddings(embeddings, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (title, emb) in enumerate(embeddings):
            if i >= len(axes):
                break
            
            # Ensure the tensor is contiguous, on CPU, and reshape
            emb = emb.contiguous().cpu().reshape(-1, emb.shape[-1])
            
            # Select a subset of embeddings if too large
            if emb.shape[0] > 10000:
                indices = np.random.choice(emb.shape[0], 10000, replace=False)
                emb = emb[indices]

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            emb_2d = tsne.fit_transform(emb.numpy())

            # Plot
            axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


class CrossAttentionModule(nn.Module):
    def __init__(self, dim=128, heads=8, dim_head=64):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, heads * dim_head)
        self.to_k = nn.Linear(dim, heads * dim_head)
        self.to_v = nn.Linear(dim, heads * dim_head)
        self.to_out = nn.Linear(heads * dim_head, dim)

    def forward(self, ml_c, ml_r, fl_r):
        print(f"🌻 CrossAttentionModule Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        
        B, C, H, W = fl_r.shape
        
        # Flatten and transpose inputs
        ml_c = ml_c.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        ml_r = ml_r.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        fl_r = fl_r.view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
        
        print(f"After flattening: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")

        # Compute Q, K, V
        q = self.to_q(ml_c).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(ml_r).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(fl_r).view(B, H*W, self.heads, self.dim_head).transpose(1, 2)
        
        print(f"Q: {q.shape}, K: {k.shape}, V: {v.shape}")

        # Compute attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        print(f"Attention shape: {attn.shape}")

        # Apply attention to values
        out = torch.matmul(attn, v)
        print(f"Output after attention: {out.shape}")
        
        # Reshape and project back to original dimensions
        out = out.transpose(1, 2).contiguous().view(B, H*W, self.heads * self.dim_head)
        out = self.to_out(out)
        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        print(f"Final output shape: {out.shape}")

        return out

# Test the module
if __name__ == "__main__":
    B, C, H, W = 2, 128, 64, 64
    ml_c = torch.randn(B, C, H, W)
    ml_r = torch.randn(B, C, H, W)
    fl_r = torch.randn(B, C, H, W)

    module = CrossAttentionModule()
    output = module(ml_c, ml_r, fl_r)

# Example usage
if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    # Example dimensions
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024



    # Create random input tensors and move to device
    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    # Initialize the ImplicitMotionAlignment module and move to device
    model = ImplicitMotionAlignment(feature_dim, motion_dim, depth, heads, dim_head, mlp_dim).to(device)

    # Forward pass
    with torch.no_grad():
        output, embeddings = model(ml_c, ml_r, fl_r)

    #print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    #print(f"Output shape: {output.shape}")

    # Visualize embeddings
    model.visualize_embeddings(embeddings, "embeddings_visualization.png")
    #print("Embedding visualization saved as 'embeddings_visualization.png'")