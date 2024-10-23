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


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H*W).permute(0, 2, 1)  # Change to (B, H*W, C)
        
        x_norm = self.norm1(x_reshaped)
        att_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + att_output

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        output = x_reshaped.permute(0, 2, 1).view(B, C, H, W)
        return output


class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim,spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim * spatial_dim, dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        

    def forward(self, ml_c, ml_r, fl_r):
        # embeddings = []
        print(f"self.spatial_dim:{self.spatial_dim}")
        print(f"self.feature_dim:{self.feature_dim}")
        print(f"self.motion_dim:{self.motion_dim}")
        # Cross-attention module
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        # embeddings.append(("After Cross-Attention", V_prime.detach().cpu()))
        print(f"ImplicitMotionAlignment: After cross-attention, V_prime.shape = {V_prime.shape}")

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            V_prime = block(V_prime)
            # embeddings.append((f"After Transformer Block {i}", V_prime.detach().cpu()))
            print(f"ImplicitMotionAlignment: After transformer block {i}, V_prime.shape = {V_prime.shape}")

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
    def __init__(self, 
        dim_spatial=4096,
        dim_qk=256,
        dim_v=256
        ):
     
        super().__init__()
        self.dim_head = dim_qk
        self.scale = dim_qk ** -0.5
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.attend = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        q = torch.flatten(queries, start_dim=2).transpose(-1, -2)
        q = q + self.q_pos_embedding

        k = torch.flatten(keys, start_dim=2).transpose(-1, -2)
        k = k + self.k_pos_embedding

        v = torch.flatten(values, start_dim=2).transpose(-1, -2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = torch.reshape(out.transpose(-1, -2), values.shape)

        return out


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

    print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    print(f"Output shape: {output.shape}")

    # Visualize embeddings
    # model.visualize_embeddings(embeddings, "embeddings_visualization.png")
    #print("Embedding visualization saved as 'embeddings_visualization.png'")