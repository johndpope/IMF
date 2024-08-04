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
        self.attention = nn.MultiheadAttention(dim, heads)
        # ffn
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)


    def forward(self, x):
        #print(f"TransformerBlock input shape: {x.shape}")
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H*W).permute(2, 0, 1)
        
        x_norm = self.norm1(x_reshaped)
        att_output, _ = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + att_output
        #print(f"TransformerBlock: After attention, x_reshaped.shape = {x_reshaped.shape}")

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output
        #print(f"TransformerBlock: After feedforward, x_reshaped.shape = {x_reshaped.shape}")

        output = x_reshaped.permute(1, 2, 0).view(B, C, H, W)
        #print(f"TransformerBlock output shape: {output.shape}")
        return output

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(feature_dim, motion_dim, heads, dim_head)
        
        # x4
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim),
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim),
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim),
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim)
        ])

    def forward(self, ml_c, ml_r, fl_r):
        embeddings = []
        
        # Cross-attention module
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        embeddings.append(("After Cross-Attention", V_prime.detach().cpu()))
        #print(f"ImplicitMotionAlignment: After cross-attention, V_prime.shape = {V_prime.shape}")

        # Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            V_prime = block(V_prime)
            embeddings.append((f"After Transformer Block {i}", V_prime.detach().cpu()))
            #print(f"ImplicitMotionAlignment: After transformer block {i}, V_prime.shape = {V_prime.shape}")

        return V_prime #, embeddings


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
    def __init__(self, feature_dim, motion_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(motion_dim, heads * dim_head)
        self.to_k = nn.Linear(motion_dim, heads * dim_head)
        self.to_v = nn.Linear(feature_dim, heads * dim_head)
        self.to_out = nn.Linear(heads * dim_head, feature_dim)

        self.pos_encoding_q = PositionalEncoding(motion_dim)
        self.pos_encoding_k = PositionalEncoding(motion_dim)

    def forward(self, ml_c, ml_r, fl_r):
        print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")

        B, C_m, H, W = ml_c.shape
        _, C_f, _, _ = fl_r.shape
        dim_spatial = H * W

        # Flatten spatial dimensions and transpose
        q = torch.flatten(ml_c, start_dim=2).transpose(-1, -2)
        k = torch.flatten(ml_r, start_dim=2).transpose(-1, -2)
        v = torch.flatten(fl_r, start_dim=2).transpose(-1, -2)
        print(f"After flattening: q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Add positional encodings
        q = q + self.pos_encoding_q(q)
        k = k + self.pos_encoding_k(k)
        print(f"After adding positional encoding: q: {q.shape}, k: {k.shape}")

        # Linear projections
        q = self.to_q(q).view(B, dim_spatial, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(k).view(B, dim_spatial, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(v).view(B, dim_spatial, self.heads, self.dim_head).transpose(1, 2)
        print(f"After linear projections: q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Compute attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1)
        print(f"Attention shape: {attn.shape}")

        # Apply attention to values
        out = torch.matmul(attn, v)
        print(f"After applying attention: {out.shape}")

        # Reshape and project back to original feature dimension
        out = out.transpose(1, 2).contiguous().view(B, dim_spatial, self.heads * self.dim_head)
        out = self.to_out(out)
        print(f"After reshaping and final projection: {out.shape}")

        # Reshape back to original spatial dimensions
        out = out.transpose(-1, -2).view(B, C_f, H, W)
        print(f"Final output shape: {out.shape}")

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

    #print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    #print(f"Output shape: {output.shape}")

    # Visualize embeddings
    model.visualize_embeddings(embeddings, "embeddings_visualization.png")
    #print("Embedding visualization saved as 'embeddings_visualization.png'")