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


class MoHAttention(nn.Module):
    def __init__(self, dim, num_heads, shared_heads=1, routed_heads=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shared_heads = shared_heads
        self.routed_heads = routed_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

        self.router = nn.Linear(dim, num_heads - shared_heads)
        self.shared_router = nn.Linear(dim, 2)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

    def forward(self, x):
        B, N, C = x.shape
        print(f"Input shape: {x.shape}")

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        print(f"q, k, v shape: {q.shape}")

        q_norm = F.normalize(q, dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature)

        attn = (q_norm_scaled @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        print(f"Attention shape: {attn.shape}")

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print(f"After attention shape: {x.shape}")

        logits = self.router(x)
        gates = F.softmax(logits, dim=-1)
        print(f"Gates shape: {gates.shape}")

        _, indices = torch.topk(gates, k=self.routed_heads, dim=-1)
        mask = F.one_hot(indices, num_classes=self.num_heads-self.shared_heads).sum(dim=-2)
        print(f"Mask shape: {mask.shape}")

        routed_head_gates = gates * mask
        routed_head_gates = routed_head_gates * self.routed_heads
        print(f"Routed head gates shape: {routed_head_gates.shape}")

        shared_head_weight = self.shared_router(x)
        shared_head_gates = F.softmax(shared_head_weight, dim=-1) * self.shared_heads
        print(f"Shared head gates shape: {shared_head_gates.shape}")

        weight_0 = self.shared_router(x)
        weight_0 = F.softmax(weight_0, dim=-1) * 2
        print(f"Weight_0 shape: {weight_0.shape}")

        shared_head_gates = torch.einsum("bn,bnc->bnc", weight_0[..., 0], shared_head_gates)
        routed_head_gates = torch.einsum("bn,bnc->bnc", weight_0[..., 1], routed_head_gates)

        masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=-1)
        print(f"Masked gates shape: {masked_gates.shape}")
        print(f"x shape before einsum: {x.shape}")

        x = torch.einsum("bnc,bnd->bnd", masked_gates, x)
        print(f"x shape after einsum: {x.shape}")

        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, shared_heads=1, routed_heads=3):
        super().__init__()
        self.attention = MoHAttention(dim, num_heads, shared_heads, routed_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # B, H*W, C
        
        x_norm = self.norm1(x_flat)
        att_output = self.attention(x_norm)
        x_flat = x_flat + att_output

        ff_output = self.mlp(self.norm2(x_flat))
        x_flat = x_flat + ff_output

        output = x_flat.permute(0, 2, 1).view(B, C, H, W)
        return output

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, mlp_dim=1024, shared_heads=1, routed_heads=3):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[1], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, mlp_dim, shared_heads, routed_heads) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim

    def forward(self, ml_c, ml_r, fl_r):
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        
        for block in self.transformer_blocks:
            V_prime = block(V_prime)

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

        #print("CrossAttentionModule:",dim_spatial)
        #print("dim_qk:",dim_qk)
        #print("dim_v:",dim_v)
        

        # Separate positional encodings for queries and keys
        self.q_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, dim_spatial, dim_qk))
        self.attend = nn.Softmax(dim=-1)
    def forward(self, queries, keys, values):
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

        return out


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    mlp_dim = 1024
    shared_heads = 1
    routed_heads = 3

    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    model = ImplicitMotionAlignment(feature_dim, motion_dim, (H, W), depth, heads, mlp_dim, shared_heads, routed_heads).to(device)

    with torch.no_grad():
        output = model(ml_c, ml_r, fl_r)

    print(f"Input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    print(f"Output shape: {output.shape}")