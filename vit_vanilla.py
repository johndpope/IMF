import torch
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        debug_print(f"PatchEmbedding input shape: {x.shape}")
        x = self.projection(x)
        debug_print(f"PatchEmbedding after projection shape: {x.shape}")
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=x.shape[0])
        x = torch.cat([cls_tokens, x], dim=1)
        debug_print(f"PatchEmbedding output shape: {x.shape}")
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    
    def forward(self, x):
        debug_print(f"MultiHeadAttention input shape: {x.shape}")
        out = self.attention(x, x, x)[0]
        debug_print(f"MultiHeadAttention output shape: {out.shape}")
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
    
    def forward(self, x):
        debug_print(f"FeedForward input shape: {x.shape}")
        out = self.net(x)
        debug_print(f"FeedForward output shape: {out.shape}")
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = FeedForward(embed_dim, mlp_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        debug_print(f"TransformerBlock input shape: {x.shape}")
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        debug_print(f"TransformerBlock output shape: {x.shape}")
        return x

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.embed_dim = feature_dim
        self.spatial_dim = spatial_dim

        # Calculate patch size based on input spatial dimensions
        self.patch_size = 256 // spatial_dim[0]  # Assuming 256x256 input
        
        self.patch_embedding = PatchEmbedding(motion_dim, self.embed_dim, self.patch_size)
        
        num_patches = (spatial_dim[0] * spatial_dim[1]) // (self.patch_size ** 2)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.embed_dim))
        
        self.transformer = nn.Sequential(*[TransformerBlock(self.embed_dim, heads, mlp_dim) for _ in range(depth)])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, feature_dim)
        )

    def forward(self, ml_c, ml_r, fl_r):
        print(f"VisionTransformer input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        embeddings = []
        
        x = self.patch_embedding(ml_c)
        print(f"After patch embedding shape: {x.shape}")
        x += self.pos_embedding
        print(f"After adding positional embedding shape: {x.shape}")
        embeddings.append(("After Patch Embedding", x.detach().cpu()))
        
        for i, block in enumerate(self.transformer):
            x = block(x)
            print(f"After transformer block {i} shape: {x.shape}")
            embeddings.append((f"After Transformer Block {i}", x.detach().cpu()))
        
        # Remove the CLS token and reshape to original spatial dimensions
        x = x[:, 1:, :]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.spatial_dim[0], w=self.spatial_dim[1])
        print(f"After rearranging to spatial dimensions: {x.shape}")
        
        # Apply MLP head to each spatial location
        x = self.mlp_head(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        print(f"Final output shape: {x.shape}")
        
        return x, embeddings


    @staticmethod
    def visualize_embeddings(embeddings, save_path):
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import numpy as np

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

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_print(f"Using device: {device}")
    
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024

    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    model = ImplicitMotionAlignment(feature_dim, motion_dim, (H, W), depth, heads, dim_head, mlp_dim).to(device)

    debug_print("Model architecture:")
    debug_print(model)

    debug_print("\nStarting forward pass...")
    with torch.no_grad():
        output, embeddings = model(ml_c, ml_r, fl_r)

    debug_print(f"\nInput shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
    debug_print(f"Output shape: {output.shape}")

    model.visualize_embeddings(embeddings, "embeddings_visualization.png")
    debug_print("Embedding visualization saved as 'embeddings_visualization.png'")