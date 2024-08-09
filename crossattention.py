import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
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
        B, L, C = ml_c.shape
        
        q = self.to_q(ml_c).view(B, L, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(ml_r).view(B, L, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(fl_r).view(B, L, self.heads, self.dim_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.heads * self.dim_head)
        out = self.to_out(out)
        
        return out

class CrossAttentionModule(nn.Module):
    def __init__(self, dims=[128, 256, 512, 512], heads=8, dim_head=64):
        super().__init__()
        self.dims = dims
        self.attention_layers = nn.ModuleList([
            CrossAttentionLayer(dim, heads, dim_head) for dim in dims
        ])

    def forward(self, ml_c, ml_r, fl_r):
        outputs = []
        for i, layer in enumerate(self.attention_layers):
            print(f"🌻 Layer {i+1} Input shapes: ml_c: {ml_c[i].shape}, ml_r: {ml_r[i].shape}, fl_r: {fl_r[i].shape}")
            
            B, C, H, W = fl_r[i].shape
            
            # Flatten and transpose inputs
            ml_c_flat = ml_c[i].view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
            ml_r_flat = ml_r[i].view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
            fl_r_flat = fl_r[i].view(B, C, -1).transpose(1, 2)  # (B, H*W, C)
            
            print(f"After flattening: ml_c: {ml_c_flat.shape}, ml_r: {ml_r_flat.shape}, fl_r: {fl_r_flat.shape}")

            out = layer(ml_c_flat, ml_r_flat, fl_r_flat)
            out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            
            print(f"Layer {i+1} output shape: {out.shape}")
            outputs.append(out)

        return outputs

# Test the module
if __name__ == "__main__":
    dims = [128, 256, 512, 512]
    B = 2
    ml_c = [torch.randn(B, dim, 64 // (2**i), 64 // (2**i)) for i, dim in enumerate(dims)]
    ml_r = [torch.randn(B, dim, 64 // (2**i), 64 // (2**i)) for i, dim in enumerate(dims)]
    fl_r = [torch.randn(B, dim, 64 // (2**i), 64 // (2**i)) for i, dim in enumerate(dims)]

    module = CrossAttentionModule(dims)
    outputs = module(ml_c, ml_r, fl_r)

    for i, out in enumerate(outputs):
        print(f"Final output shape for layer {i+1}: {out.shape}")