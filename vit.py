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
    def __init__(self, feature_dim, motion_dim,spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[0], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        
    def forward(self, ml_c, ml_r, fl_r):
        #print(f"ImplicitMotionAlignment input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        #print(f"After cross-attention shape: {V_prime.shape}")
        for i, block in enumerate(self.transformer_blocks):
            V_prime = block(V_prime)
            #print(f"After transformer block {i} shape: {V_prime.shape}")
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

        ##print("CrossAttentionModule:",dim_spatial)
        ##print("dim_qk:",dim_qk)
        ##print("dim_v:",dim_v)
        

        # Separate positional encodings for queries and keys
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.attend = nn.Softmax(dim=-1)
    def forward(self, queries, keys, values):
        #print(f"CrossAttentionModule input shapes: queries: {queries.shape}, keys: {keys.shape}, values: {values.shape}")

        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        q = torch.flatten(queries, start_dim=2).transpose(-1, -2)
        q = q + self.q_pos_embedding  # (b, dim_spatial, dim_qk)

        # in paper, key dim_spatial may be different from query dim_spatial
        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        k = torch.flatten(keys, start_dim=2).transpose(-1, -2)
        k = k + self.k_pos_embedding  # (b, dim_spatial, dim_qk)
        # (b, dim_v, h, w) -> (b, dim_v, dim_spatial) -> (b, dim_spatial, dim_v)
        v = torch.flatten(values, start_dim=2).transpose(-1, -2)

        # # (b, dim_spatial, dim_qk) * (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_spatial)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)  # (b, dim_spatial, dim_spatial)

        # (b, dim_spatial, dim_spatial) * (b, dim_spatial, dim_v) -> (b, dim_spatial, dim_v)
        out = torch.matmul(attn, v)

        # Or the torch version fast attention
        # out = F.scaled_dot_product_attention(q, k, v)
        out = torch.reshape(out.transpose(-1, -2), values.shape)  # (b, dim_spatial, dim_v) -> (b, dim_v, h, w)
        #print(f"CrossAttentionModule output shape: {out.shape}")

        return out


# Example usage
if __name__ == "test":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##print(f"Using device: {device}")

    # Example dimensions
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024
    spatial_dim = (64,64)


    # Create random input tensors and move to device
    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    # Initialize the ImplicitMotionAlignment module and move to device
    model = ImplicitMotionAlignment(feature_dim, motion_dim,spatial_dim, depth, heads, dim_head, mlp_dim).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(ml_c, ml_r, fl_r)

    #print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    #print(f"Output shape: {output.shape}")

    # Visualize embeddings
    # model.visualize_embeddings(embeddings, "embeddings_visualization.png")
    ##print("Embedding visualization saved as 'embeddings_visualization.png'")

if __name__ == "__main__":

    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024
    spatial_dim = (64,64)

    ml_c = torch.randn(B, C_m, H, W)
    ml_r = torch.randn(B, C_m, H, W)
    fl_r = torch.randn(B, C_f, H, W)

    pytorch_model = ImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim)
    pytorch_output = pytorch_model(ml_c, ml_r, fl_r)    
    #print("output:",pytorch_output.shape)