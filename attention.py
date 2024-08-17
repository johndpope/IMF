import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(
        self,
        dim_spatial=4096,
        dim_qk=256,
        dim_v=256,
        **kwargs,
    ):
        super().__init__()

        self.scale = dim_qk**-0.5

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


if __name__ == "__main__":
    # (b, c, h, w)
    queries = torch.randn(1, 128, 64, 64)
    keys = torch.randn(1, 128, 64, 64)
    values = torch.randn(1, 256, 64, 64)


    attn = Attention(dim_spatial=64 * 64, dim_qk=128, dim_v=256)
    out = attn(queries, keys, values)
    print(out.shape)


