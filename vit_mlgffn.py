import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns


DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.attention_weights = None

    def forward(self, x):
        B, C, H, W = x.shape
        print(f"WindowAttention input shape: {x.shape}")

        # Unfold input
        x_unfolded = F.unfold(x, kernel_size=self.window_size, stride=self.window_size)
        print(f"After unfolding: {x_unfolded.shape}")
        
        x_unfolded = x_unfolded.view(B, C, self.window_size[0], self.window_size[1], -1).permute(0, 4, 1, 2, 3)
        print(f"After reshaping and permuting: {x_unfolded.shape}")
        
        # Apply QKV
        qkv = self.qkv(x_unfolded.view(-1, C, self.window_size[0], self.window_size[1]))
        print(f"After QKV: {qkv.shape}")
        
        q, k, v = qkv.chunk(3, dim=1)
        print(f"Q, K, V shapes: {q.shape}, {k.shape}, {v.shape}")

        # Reshape and compute attention
        q = q.view(-1, self.num_heads, C // self.num_heads, self.window_size[0] * self.window_size[1]).transpose(-1, -2)
        k = k.view(-1, self.num_heads, C // self.num_heads, self.window_size[0] * self.window_size[1])
        v = v.view(-1, self.num_heads, C // self.num_heads, self.window_size[0] * self.window_size[1]).transpose(-1, -2)
        print(f"Reshaped Q, K, V shapes: {q.shape}, {k.shape}, {v.shape}")

        attn = (q @ k) * self.scale
        print(f"Attention shape before softmax: {attn.shape}")
        
        attn = attn.softmax(dim=-1)
        print(f"Attention shape after softmax: {attn.shape}")

        # Store attention weights for visualization
        self.attention_weights = attn.detach()

        x = (attn @ v).transpose(-1, -2).contiguous()
        print(f"Shape after attention application: {x.shape}")
        
        x = x.view(-1, C, self.window_size[0], self.window_size[1])
        print(f"Shape after reshaping: {x.shape}")
        
        x = self.proj(x)
        print(f"Shape after projection: {x.shape}")

        # Fold output
        x = x.view(B, -1, C * self.window_size[0] * self.window_size[1]).transpose(1, 2)
        print(f"Shape before folding: {x.shape}")
        
        x = F.fold(x, output_size=(H, W), kernel_size=self.window_size, stride=self.window_size)
        print(f"WindowAttention output shape: {x.shape}")
        
        return x


    def get_attention_weights(self, x):
        # Run a forward pass to compute attention weights
        self.forward(x)
        return self.attention_weights

def visualize_window_attention(model, input_tensor):
    B, C, H, W = input_tensor.shape
    
    with torch.no_grad():
        attention_weights = model.get_attention_weights(input_tensor)
    
    # Average attention weights across all heads
    avg_attention = attention_weights.mean(dim=1)
    
    # Reshape attention weights to match the input shape
    num_windows = (H // model.window_size[0]) * (W // model.window_size[1])
    avg_attention = avg_attention.view(B, num_windows, model.window_size[0] * model.window_size[1], -1)
    
    # Create a figure with subplots for each window
    fig, axes = plt.subplots(H // model.window_size[0], W // model.window_size[1], figsize=(20, 20))
    fig.suptitle('Average Window Attention Weights', fontsize=16)

    for i, ax in enumerate(axes.flat):
        sns.heatmap(avg_attention[0, i].cpu().numpy(), ax=ax, cmap='viridis', cbar=False)
        ax.set_title(f'Window {i+1}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        print(f"GlobalAttention input shape: {x.shape}")

        # Apply QKV
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape and compute attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(-1, -2)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(-1, -2).contiguous()
        x = x.view(B, C, H, W)
        x = self.proj(x)

        print(f"GlobalAttention output shape: {x.shape}")
        return x

class SpatialAwareSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        
        self.window_attn = WindowAttention(dim, self.window_size, num_heads)
        self.global_attn = GlobalAttention(dim, num_heads)

    def forward(self, x):
        print(f"SpatialAwareSelfAttention input shape: {x.shape}")
        
        # Window attention
        x_windows = self.window_attn(x)
        print(f"SpatialAwareSelfAttention after window attention: {x_windows.shape}")
        
        # Global attention
        x_global = self.global_attn(x)
        print(f"SpatialAwareSelfAttention after global attention: {x_global.shape}")
        
        # Combine local and global attention
        x = x_windows + x_global
        
        print(f"SpatialAwareSelfAttention output shape: {x.shape}")
        return x






class ChannelAwareSelfAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        debug_print(f"ChannelAwareSelfAttention input shape: {x.shape}")
        y = F.adaptive_avg_pool2d(x, (1, 1))
        y = self.fc(y)
        debug_print(f"ChannelAwareSelfAttention after fc: {y.shape}")
        output = x * y
        debug_print(f"ChannelAwareSelfAttention output shape: {output.shape}")
        return output

class MLGFFN(nn.Module):
    def __init__(self, dim, expansion_factor=4):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.dwconv3x3 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1, groups=hidden_dim // 2)
        self.dwconv5x5 = nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=5, padding=2, groups=hidden_dim // 2)
        self.fc2 = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        debug_print(f"MLGFFN input shape: {x.shape}")
        x = self.fc1(x)
        debug_print(f"MLGFFN after fc1: {x.shape}")
        
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        debug_print(f"MLGFFN after split: {x1.shape}, {x2.shape}")
        x1 = self.act(self.dwconv3x3(x1))
        x2 = self.act(self.dwconv5x5(x2))
        debug_print(f"MLGFFN after convolutions: {x1.shape}, {x2.shape}")
        
        x_local = torch.cat([x1, x2], dim=1)
        x_global = F.adaptive_avg_pool2d(x, (1, 1)).expand_as(x)
        debug_print(f"MLGFFN local and global: {x_local.shape}, {x_global.shape}")
        
        x = torch.cat([x_local, x_global], dim=1)
        x = self.fc2(x)
        debug_print(f"MLGFFN output shape: {x.shape}")
        return x

class HSCATB(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(dim)
        self.attn = SpatialAwareSelfAttention(dim, num_heads, window_size)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.casa = ChannelAwareSelfAttention(dim)
        self.norm3 = nn.InstanceNorm2d(dim)
        self.mlgffn = MLGFFN(dim, mlp_ratio)

    def forward(self, x):
        print(f"HSCATB input shape: {x.shape}")
        x = x + self.attn(self.norm1(x))
        print(f"HSCATB after spatial attention: {x.shape}")
        x = x + self.casa(self.norm2(x))
        print(f"HSCATB after channel attention: {x.shape}")
        x = x + self.mlgffn(self.norm3(x))
        print(f"HSCATB output shape: {x.shape}")
        return x

class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, depth=2, num_heads=8, window_size=4, mlp_ratio=4):
        super().__init__()
        self.cross_attention = CrossAttentionModule(feature_dim, motion_dim, num_heads)
        self.hscatb_blocks = nn.ModuleList([
            HSCATB(feature_dim, num_heads, window_size, mlp_ratio) for _ in range(depth)
        ])

    def forward(self, ml_c, ml_r, fl_r):
        debug_print(f"ImprovedImplicitMotionAlignment input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        debug_print(f"ImprovedImplicitMotionAlignment after cross attention: {V_prime.shape}")
        
        for i, block in enumerate(self.hscatb_blocks):
            V_prime = block(V_prime)
            debug_print(f"ImprovedImplicitMotionAlignment after HSCATB block {i+1}: {V_prime.shape}")

        debug_print(f"ImprovedImplicitMotionAlignment output shape: {V_prime.shape}")
        return V_prime

class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, motion_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Conv2d(motion_dim, feature_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(motion_dim, feature_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, ml_c, ml_r, fl_r):
        debug_print(f"CrossAttentionModule input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        B, _, H, W = ml_c.shape

        q = self.to_q(ml_c).view(B, self.num_heads, self.head_dim, H*W).permute(0, 1, 3, 2)
        k = self.to_k(ml_r).view(B, self.num_heads, self.head_dim, H*W).permute(0, 1, 3, 2)
        v = self.to_v(fl_r).view(B, self.num_heads, self.head_dim, H*W).permute(0, 1, 3, 2)
        debug_print(f"CrossAttentionModule q,k,v shapes: {q.shape}, {k.shape}, {v.shape}")

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        debug_print(f"CrossAttentionModule attention shape: {attn.shape}")

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.feature_dim, H, W)
        out = self.to_out(out)
        debug_print(f"CrossAttentionModule output shape: {out.shape}")
        return out

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    num_heads = 8
    window_size = 8
    mlp_ratio = 4

    ml_c = torch.randn(B, C_m, H, W).to(device)
    ml_r = torch.randn(B, C_m, H, W).to(device)
    fl_r = torch.randn(B, C_f, H, W).to(device)

    model = ImplicitMotionAlignment(feature_dim, motion_dim, depth, num_heads, window_size, mlp_ratio).to(device)

    with torch.no_grad():
        output = model(ml_c, ml_r, fl_r)

    print(f"\nFinal output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")


    # Initialize model and input
    window_size = (8, 8)  # Adjust as needed
    model = WindowAttention(dim=256, window_size=window_size, num_heads=8).to(device)
    input_tensor = torch.randn(1, 256, 64, 64).to(device)

    # Visualize attention
    visualize_window_attention(model, input_tensor)