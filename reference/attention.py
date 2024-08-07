
# --------------------------------------------------------
# Adopted from Swin Transformer
# Modified by Krushi Patel
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange, Reduce

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows





def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.query_size = self.window_size
        self.key_size = self.window_size[0] * 2
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0

        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        
        attn = attn + relative_position_bias.unsqueeze(0)

      
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        #return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'
        return f'dim={self.dim}, num_heads={self.num_heads}'
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class GlobalAttention(nn.Module):
    r""" MOA - multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, input_resolution,num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.query_size = self.window_size[0]
       
        self.key_size = self.window_size[0] + 2
        h,w = input_resolution
        self.seq_len = h//self.query_size
    
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.reduction = 32
        self.pre_conv = nn.Conv2d(dim, int(dim//self.reduction), 1)
     
      
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.seq_len - 1) * (2 * self.seq_len - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        #print(self.relative_position_bias_table.shape)
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.seq_len)
        coords_w = torch.arange(self.seq_len)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
     
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2

        relative_coords[:, :, 0] += self.seq_len - 1  # shift to start from 0

        relative_coords[:, :, 1] += self.seq_len - 1
        relative_coords[:, :, 0] *= 2 * self.seq_len - 1
       
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
      
        self.register_buffer("relative_position_index", relative_position_index)
        
       

        self.queryembedding = Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = self.query_size, p2 = self. query_size)

        self.keyembedding = nn.Unfold(kernel_size=(self.key_size, self.key_size), stride = 14, padding=1)
   
        self.query_dim = int(dim//self.reduction) * self.query_size * self.query_size
        self.key_dim = int(dim//self.reduction) * self.key_size * self.key_size
                
        self.q = nn.Linear(self.query_dim, self.dim,bias=qkv_bias)
        self.kv = nn.Linear(self.key_dim, 2*self.dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        #B, H, W, C = x.shape
        B,_, C = x.shape  
          
        x = x.reshape(-1, C, H, W)    
        x = self.pre_conv(x)
        query = self.queryembedding(x).view(B,-1,self.query_dim)
        query = self.q(query)
        B,N,C = query.size()
        
        q = query.reshape(B,N,self.num_heads, C//self.num_heads).permute(0,2,1,3)
        key = self.keyembedding(x).view(B,-1,self.key_dim)
        kv = self.kv(key).reshape(B,N,2,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        k = kv[0]
        v = kv[1]
        
        
        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.seq_len * self.seq_len, self.seq_len * self.seq_len, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

       
        attn = attn + relative_position_bias.unsqueeze(0)
      
        attn = self.softmax(attn)
        
        attn = self.attn_drop(attn)
     
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) 
     
        x = self.proj(x)
      
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class LocalTransformerBlock(nn.Module):
    r""" Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
   
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
           
            self.window_size = min(self.input_resolution)
   
       
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)



    def forward(self, x):
        H, W = self.input_resolution
     
        B, L, C = x.shape
      
        assert L == H * W, "input feature has wrong size"
       
        shortcut = x
        x = self.norm1(x)
        
        x = x.view(B, H, W, C)
     

    
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C 
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C     
        attn_windows = self.attn(x_windows)  # nW*B, window_size*window_size, C    
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)


       

 
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, drop_path_global=0., use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
       
        self.drop_path_gl = DropPath(drop_path_global) if drop_path_global > 0. else nn.Identity()
        # build blocks
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
           if min(self.input_resolution) >= self.window_size:
                 self.glb_attn = GlobalAttention(dim, to_2tuple(window_size), self.input_resolution, num_heads = num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                 self.post_conv = nn.Conv2d(dim, dim, 3, padding=1)
                 self.norm1 = norm_layer(dim)
                 self.norm2 = norm_layer(dim)
                
           else:
                 self.post_conv = None
                 self.glb_attn = None
                 self.norm1 = None
                 self.norm2 = None
                
           self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
           
            
        else:
            self.downsample = None
            
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                
        if self.downsample is not None:
        
           if min(self.input_resolution) >= self.window_size:
                shortcut = x
                x = self.norm1(x)
                H, W = self.input_resolution
                B,_,C = x.size()
         
                no_window = int(H*W/self.window_size**2)   
                local_attn = x.view(B,no_window,self.window_size, self.window_size,C)
             
                glb_attn = self.glb_attn(x, H, W)
                glb_attn = glb_attn.view(B,no_window,1,1,C)
                x = torch.add(local_attn, glb_attn).view(B,C,H,W)
              

                x = shortcut.view(B,C,H,W) + self.drop_path_gl(x)
                x = self.norm2(x.view(B,H*W,C))
                post_conv = self.drop_path_gl(self.post_conv(x.view(B,C,H,W))).view(B, H*W, C)
                x = x + post_conv
                
           x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class MOATransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_global = [x.item() for x in torch.linspace(0, 0.2, len(depths)-1)]
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               drop_path_global = (dpr_global[i_layer]) if (i_layer < self.num_layers -1) else 0,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
 
        for layer in self.layers:
            x = layer(x)
           
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

# --------------------------------------------------------
# Adopted from Swin Transformer
# Modified by Krushi Patel
# print(sum(p.numel() for p in model.parameters() if p.requires_grad), 'parameters')
# --------------------------------------------------------

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = MOATransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=14,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    output=model(input)
    print(output.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F



class DoubleAttention(nn.Module):

    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)

        return tmpZ 


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    a2 = DoubleAttention(512,128,128,True)
    output=a2(input)
    print(output.shape)

    
import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    kernel_size=input.shape[2]
    cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
    output=cbam(input)
    print(output.shape)

    
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim=seg_dim

        self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)

        self.reweighting=MLP(dim,dim//4,dim*3)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,x) :
        B,H,W,C=x.shape

        c_embed=self.mlp_c(x)

        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,2,1,4).reshape(B,self.seg_dim,W,H*S)
        h_embed=self.mlp_h(h_embed).reshape(B,self.seg_dim,W,H,S).permute(0,3,2,1,4).reshape(B,H,W,C)

        w_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,1,2,4).reshape(B,self.seg_dim,H,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,W,S).permute(0,2,3,1,4).reshape(B,H,W,C)

        weight=(c_embed+h_embed+w_embed).permute(0,3,1,2).flatten(2).mean(2)
        weight=self.reweighting(weight).reshape(B,C,3).permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2)

        x=c_embed*weight[0]+w_embed*weight[1]+h_embed*weight[2]

        x=self.proj_drop(self.proj(x))

        return x



if __name__ == '__main__':
    input=torch.randn(64,8,8,512)
    seg_dim=8
    vip=WeightedPermuteMLP(512,seg_dim)
    out=vip(input)
    print(out.shape)
    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model,1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input) #(bs,nq,1)
        weight_i = torch.softmax(i, dim=1) #bs,nq,1
        context_score = weight_i * self.fc_k(input) #bs,nq,d_model
        context_vector = torch.sum(context_score,dim=1,keepdim=True) #bs,1,d_model
        v = self.fc_v(input) * context_vector #bs,nq,d_model
        out = self.fc_o(v) #bs,nq,d_model

        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = ShuffleAttention(channel=512,G=8)
    output=se(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from model.attention.SelfAttention import ScaledDotProductAttention
from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention

class PositionAttentionModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=ScaledDotProductAttention(d_model,d_k=d_model,d_v=d_model,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1).permute(0,2,1) #bs,h*w,c
        y=self.pa(y,y,y) #bs,h*w,c
        return y


class ChannelAttentionModule(nn.Module):
    
    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.cnn=nn.Conv2d(d_model,d_model,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.pa=SimplifiedScaledDotProductAttention(H*W,h=1)
    
    def forward(self,x):
        bs,c,h,w=x.shape
        y=self.cnn(x)
        y=y.view(bs,c,-1) #bs,c,h*w
        y=self.pa(y,y,y) #bs,c,h*w
        return y




class DAModule(nn.Module):

    def __init__(self,d_model=512,kernel_size=3,H=7,W=7):
        super().__init__()
        self.position_attention_module=PositionAttentionModule(d_model=512,kernel_size=3,H=7,W=7)
        self.channel_attention_module=ChannelAttentionModule(d_model=512,kernel_size=3,H=7,W=7)
    
    def forward(self,input):
        bs,c,h,w=input.shape
        p_out=self.position_attention_module(input)
        c_out=self.channel_attention_module(input)
        p_out=p_out.permute(0,2,1).view(bs,c,h,w)
        c_out=c_out.view(bs,c,h,w)
        return p_out+c_out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
    print(danet(input).shape)

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple

class LocalAttention(nn.Module):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop):
        
        super().__init__()

        window_size = to_2tuple(window_size)

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        assert dim % heads == 0
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)
        self.window_size = window_size
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.01)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):

        B, C, H, W = x.size()
        r1, r2 = H // self.window_size[0], W // self.window_size[1]
        
        x_total = einops.rearrange(x, 'b c (r1 h1) (r2 w1) -> b (r1 r2) (h1 w1) c', h1=self.window_size[0], w1=self.window_size[1]) # B x Nr x Ws x C
        
        x_total = einops.rearrange(x_total, 'b m n c -> (b m) n c')

        qkv = self.proj_qkv(x_total) # B' x N x 3C
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q * self.scale
        q, k, v = [einops.rearrange(t, 'b n (h c1) -> b h n c1', h=self.heads) for t in [q, k, v]]
        attn = torch.einsum('b h m c, b h n c -> b h m n', q, k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_bias = relative_position_bias
        attn = attn + attn_bias.unsqueeze(0)

        if mask is not None:
            # attn =(b * nW) h w w
            # mask =nW ww ww
            nW, ww, _ = mask.size()
            attn = einops.rearrange(attn, '(b n) h w1 w2 -> b n h w1 w2', n=nW, h=self.heads, w1=ww, w2=ww) + mask.reshape(1, nW, 1, ww, ww)
            attn = einops.rearrange(attn, 'b n h w1 w2 -> (b n) h w1 w2')
        attn = self.attn_drop(attn.softmax(dim=3))

        x = torch.einsum('b h m n, b h n c -> b h m c', attn, v)
        x = einops.rearrange(x, 'b h n c1 -> b n (h c1)')
        x = self.proj_drop(self.proj_out(x)) # B' x N x C
        x = einops.rearrange(x, '(b r1 r2) (h1 w1) c -> b c (r1 h1) (r2 w1)', r1=r1, r2=r2, h1=self.window_size[0], w1=self.window_size[1]) # B x C x H x W
        
        return x, None, None

class ShiftWindowAttention(LocalAttention):

    def __init__(self, dim, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size):
        
        super().__init__(dim, heads, window_size, attn_drop, proj_drop)

        self.fmap_size = to_2tuple(fmap_size)
        self.shift_size = shift_size

        assert 0 < self.shift_size < min(self.window_size), "wrong shift size."

        img_mask = torch.zeros(*self.fmap_size)  # H W
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1
        mask_windows = einops.rearrange(img_mask, '(r1 h1) (r2 w1) -> (r1 r2) (h1 w1)', h1=self.window_size[0],w1=self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW ww ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        self.register_buffer("attn_mask", attn_mask)
      
    def forward(self, x):

        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        sw_x, _, _ = super().forward(shifted_x, self.attn_mask)
        x = torch.roll(sw_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        return x, None, None
    

class DAttentionBaseline(nn.Module):

    def __init__(
        self, q_size, kv_size, n_heads, n_head_channels, n_groups,
        attn_drop, proj_drop, stride, 
        offset_range_factor, use_pe, dwc_pe,
        no_off, fixed_pe, stage_idx
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        
        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, kk//2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
    
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk
        
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
            
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
            
        if self.no_off:
            offset = offset.fill(0.0)
            
        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
            
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        
        if self.use_pe:
            
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        
        y = self.proj_drop(self.proj_out(out))
        
        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)

class TransformerMLP(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.chunk = nn.Sequential()
        self.chunk.add_module('linear1', nn.Linear(self.dim1, self.dim2))
        self.chunk.add_module('act', nn.GELU())
        self.chunk.add_module('drop1', nn.Dropout(drop, inplace=True))
        self.chunk.add_module('linear2', nn.Linear(self.dim2, self.dim1))
        self.chunk.add_module('drop2', nn.Dropout(drop, inplace=True))
    
    def forward(self, x):

        _, _, H, W = x.size()
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        x = self.chunk(x)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class TransformerMLPWithConv(nn.Module):

    def __init__(self, channels, expansion, drop):
        
        super().__init__()
        
        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0) 
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)
    
    def forward(self, x):
        
        x = self.drop1(self.act(self.dwc(self.linear1(x))))
        x = self.drop2(self.linear2(x))
        
        return x

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec={stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        for d in range(self.depths):

            x0 = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)

        return x, positions, references

class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], 
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i])
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        self.cls_norm = LayerNormProxy(dims[-1]) 
        self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.reset_parameters()
    
    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    def forward(self, x):
        
        x = self.patch_proj(x)
        positions = []
        references = []
        for i in range(4):
            x, pos, ref = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
        x = self.cls_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.cls_head(x)
        
        return x, positions, references

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DAT(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        expansion=4,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 7, 7] ,
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    )
    output=model(input)
    print(output[0].shape)
from torch import nn, sqrt
import torch
import sys
from math import sqrt
sys.path.append('.')
from model.conv.MBConv import MBConvBlock
from model.attention.SelfAttention import ScaledDotProductAttention

class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        y=self.mlp0(self.s0(x))
        y=self.maxpool2d(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2d(y)
        #stage2
        y=self.mlp2(self.s2(y))
        y=self.maxpool2d(y)
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)
        #stage4
        y=self.mlp4(self.s4(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1))
        N=y.shape[-1]
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))

        return y

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    coatnet=CoAtNet(3,224)
    y=coatnet(x)
    print(y.shape)
    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self,in_ch,out_ch,k):
        super().__init__()
        if(k==1):
            self.depth_conv=nn.Identity()
        else:
            self.depth_conv=nn.Conv1d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                groups=in_ch,
                padding=k//2
                )
        self.pointwise_conv=nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            groups=1
        )
    def forward(self,x):
        out=self.pointwise_conv(self.depth_conv(x))
        return out
    


class MUSEAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):


        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.conv1=Depth_Pointwise_Conv1d(h * d_v, d_model,1)
        self.conv3=Depth_Pointwise_Conv1d(h * d_v, d_model,3)
        self.conv5=Depth_Pointwise_Conv1d(h * d_v, d_model,5)
        self.dy_paras=nn.Parameter(torch.ones(3))
        self.softmax=nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        #Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2=v.permute(0,1,3,2).contiguous().view(b_s,-1,nk) #bs,dim,n
        self.dy_paras=nn.Parameter(self.softmax(self.dy_paras))
        out2=self.dy_paras[0]*self.conv1(v2)+self.dy_paras[1]*self.conv3(v2)+self.dy_paras[2]*self.conv5(v2)
        out2=out2.permute(0,2,1) #bs.n.dim

        out=out+out2
        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=sa(input,input,input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class AFT_FULL(nn.Module):

    def __init__(self, d_model,n=49,simple=False):

        super(AFT_FULL, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model,d_model)
        if(simple):
            self.position_biases=torch.zeros((n,n))
        else:
            self.position_biases=nn.Parameter(torch.ones((n,n)))
        self.d_model = d_model
        self.n=n
        self.sigmoid=nn.Sigmoid()

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):

        bs, n,dim = input.shape

        q = self.fc_q(input) #bs,n,dim
        k = self.fc_k(input).view(1,bs,n,dim) #1,bs,n,dim
        v = self.fc_v(input).view(1,bs,n,dim) #1,bs,n,dim
        
        numerator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1))*v,dim=2) #n,bs,dim
        denominator=torch.sum(torch.exp(k+self.position_biases.view(n,1,-1,1)),dim=2) #n,bs,dim

        out=(numerator/denominator) #n,bs,dim
        out=self.sigmoid(q)*(out.permute(1,0,2)) #bs,n,dim

        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    aft_full = AFT_FULL(d_model=512, n=49)
    output=aft_full(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ResidualAttention(nn.Module):

    def __init__(self, channel=512 , num_class=1000,la=0.2):
        super().__init__()
        self.la=la
        self.fc=nn.Conv2d(in_channels=channel,out_channels=num_class,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        b,c,h,w=x.shape
        y_raw=self.fc(x).flatten(2) #b,num_class,hxw
        y_avg=torch.mean(y_raw,dim=2) #b,num_class
        y_max=torch.max(y_raw,dim=2)[0] #b,num_class
        score=y_avg+self.la*y_max
        return score

        


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
    output=resatt(input)
    print(output.shape)

    
import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    triplet = TripletAttention()
    output=triplet(input)
    print(output.shape)
    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class PSA(nn.Module):

    def __init__(self, channel=512,reduction=4,S=4):
        super().__init__()
        self.S=S

        self.convs=[]
        for i in range(S):
            self.convs.append(nn.Conv2d(channel//S,channel//S,kernel_size=2*(i+1)+1,padding=i+1))

        self.se_blocks=[]
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel//S, channel // (S*reduction),kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S*reduction), channel//S,kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        
        self.softmax=nn.Softmax(dim=1)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        #Step1:SPC module
        SPC_out=x.view(b,self.S,c//self.S,h,w) #bs,s,ci,h,w
        for idx,conv in enumerate(self.convs):
            SPC_out[:,idx,:,:,:]=conv(SPC_out[:,idx,:,:,:])

        #Step2:SE weight
        se_out=[]
        for idx,se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:,idx,:,:,:]))
        SE_out=torch.stack(se_out,dim=1)
        SE_out=SE_out.expand_as(SPC_out)

        #Step3:Softmax
        softmax_out=self.softmax(SE_out)

        #Step4:SPA
        PSA_out=SPC_out*softmax_out
        PSA_out=PSA_out.view(b,-1,h,w)

        return PSA_out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    psa = PSA(channel=512,reduction=8)
    output=psa(input)
    a=output.view(-1).sum()
    a.backward()
    print(output.shape)

    
from torch import nn
import torch
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.fn=fn
    def forward(self,x,**kwargs):
        return self.fn(self.ln(x),**kwargs)

class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim,dropout) :
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self,dim,heads,head_dim,dropout):
        super().__init__()
        inner_dim=heads*head_dim
        project_out=not(heads==1 and head_dim==dim)

        self.heads=heads
        self.scale=head_dim**-0.5

        self.attend=nn.Softmax(dim=-1)
        self.to_qkv=nn.Linear(dim,inner_dim*3,bias=False)
        
        self.to_out=nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self,x):
        qkv=self.to_qkv(x).chunk(3,dim=-1)
        q,k,v=map(lambda t:rearrange(t,'b p n (h d) -> b p h n d',h=self.heads),qkv)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        attn=self.attend(dots)
        out=torch.matmul(attn,v)
        out=rearrange(out,'b p h n d -> b p n (h d)')
        return self.to_out(out)





class Transformer(nn.Module):
    def __init__(self,dim,depth,heads,head_dim,mlp_dim,dropout=0.):
        super().__init__()
        self.layers=nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim,Attention(dim,heads,head_dim,dropout)),
                PreNorm(dim,FeedForward(dim,mlp_dim,dropout))
            ]))


    def forward(self,x):
        out=x
        for att,ffn in self.layers:
            out=out+att(out)
            out=out+ffn(out)
        return out

class MobileViTAttention(nn.Module):
    def __init__(self,in_channel=3,dim=512,kernel_size=3,patch_size=7):
        super().__init__()
        self.ph,self.pw=patch_size,patch_size
        self.conv1=nn.Conv2d(in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)
        self.conv2=nn.Conv2d(in_channel,dim,kernel_size=1)

        self.trans=Transformer(dim=dim,depth=3,heads=8,head_dim=64,mlp_dim=1024)

        self.conv3=nn.Conv2d(dim,in_channel,kernel_size=1)
        self.conv4=nn.Conv2d(2*in_channel,in_channel,kernel_size=kernel_size,padding=kernel_size//2)

    def forward(self,x):
        y=x.clone() #bs,c,h,w

        ## Local Representation
        y=self.conv2(self.conv1(x)) #bs,dim,h,w

        ## Global Representation
        _,_,h,w=y.shape
        y=rearrange(y,'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim',ph=self.ph,pw=self.pw) #bs,h,w,dim
        y=self.trans(y)
        y=rearrange(y,'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)',ph=self.ph,pw=self.pw,nh=h//self.ph,nw=w//self.pw) #bs,dim,h,w

        ## Fusion
        y=self.conv3(y) #bs,dim,h,w
        y=torch.cat([x,y],1) #bs,2*dim,h,w
        y=self.conv4(y) #bs,c,h,w

        return y


if __name__ == '__main__':
    m=MobileViTAttention()
    input=torch.randn(1,3,49,49)
    output=m(input)
    print(output.shape)
    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))
excluded_nodes = [] #'Admm','ViewBackward', 'ExpandBackward', 'CloneBackward', 'UnsafeViewBackward'
collapsed_nodes = {}

def make_dot(var, params=None, show_attrs=False, max_attr_chars=50):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes represent the variables that require grad. Orange nodes represent
    saved tensors. Green nodes represent the output tensors. Dark green nodes
    represent the base tensors of output views.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.5',
        fontname='Helvetica',
        fillcolor='#BBDEFB',
        fontcolor='#0D47A1',
    )

    edge_attr = dict(
        style='bold',
        color='#00C853',
        fontname='Helvetica',
    )
    dot = Digraph(node_attr=node_attr, edge_attr=edge_attr, graph_attr=dict(size="12,12"))
    
    seen = set()
    var_color = '#4FC3F7'  # Light Blue 300
    saved_tensor_color = '#FFD54F'  # Amber 300
    output_tensor_color = '#AED581'  # Light Green 300
    base_tensor_color = '#C8E6C9'  # Green 100
    dark_base_tensor_color = '#1B5E20'  # Green 900

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor=output_tensor_color)
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor=var_color)
            else:
                node_name = str(type(var).__name__)
                if 'Backward' in node_name:
                    if node_name not in collapsed_nodes:
                        collapsed_nodes[node_name] = str(id(var))
                        dot.node(str(id(var)), 'Backward', fillcolor='lightgray')
                    else:
                        dot.edge(str(id(var)), collapsed_nodes[node_name], style='dashed', color='gray')
                else:
                    dot.node(str(id(var)), node_name)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor=output_tensor_color)
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor=var_color)
                else:
                    node_name = str(type(var).__name__)
                    if not any(excluded_node in node_name for excluded_node in excluded_nodes):
                        dot.node(str(id(var)), node_name)
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)
    def add_base_tensor(var, color=base_tensor_color):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if var.grad_fn:
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color=dark_base_tensor_color)
            dot.edge(str(id(var._base)), str(id(var)), style="dotted", color=dark_base_tensor_color)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,H=7,W=7,ratio=3,apply_transform=True):

        super(EMSA, self).__init__()
        self.H=H
        self.W=W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.ratio=ratio
        if(self.ratio>1):
            self.sr=nn.Sequential()
            self.sr_conv=nn.Conv2d(d_model,d_model,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=d_model)
            self.sr_ln=nn.LayerNorm(d_model)

        self.apply_transform=apply_transform and h>1
        if(self.apply_transform):
            self.transform=nn.Sequential()
            self.transform.add_module('conv',nn.Conv2d(h,h,kernel_size=1,stride=1))
            self.transform.add_module('softmax',nn.Softmax(-1))
            self.transform.add_module('in',nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq ,c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if(self.ratio>1):
            x=queries.permute(0,2,1).view(b_s,c,self.H,self.W) #bs,c,H,W
            x=self.sr_conv(x) #bs,c,h,w
            x=x.contiguous().view(b_s,c,-1).permute(0,2,1) #bs,n',c
            x=self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if(self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att) # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1) # (b_s, h, nq, n')


        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



if __name__ == '__main__':
    input = torch.randn(50, 64, 512)
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
    output = emsa(input, input, input)

    # Visualize the graph
    dot = make_dot(output, params=dict(emsa.named_parameters()))
    # Attempt to modify the canvas size
    dot.graph_attr['size'] = "120,120"  # Set the size as width,height in inches

   
    dot.render('emsa_graph', format='svg')  # Save the graph as PNG
    print(output.shape)
    
import torch
from torch import nn
import math
from timm.models.layers import DropPath, to_2tuple

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class GFNet(nn.Module):
    def __init__(self, embed_dim=384, img_size=224, patch_size=16, mlp_ratio=4, depth=4, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.embedding = nn.Linear((patch_size ** 2) * 3, embed_dim)

        h = img_size // patch_size
        w = h // 2 + 1


        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, h=h, w=w)
            for i in range(depth)
        ])

        self.head = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)
        x = self.softmax(self.head(x))
        return x

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
    out = gfnet(x)
    print(out.shape)

    

import torch
import torch.nn as nn


class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)
    model = SimAM(64)
    outputs = model(input)
    print(outputs.shape)
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# relative positional embedding

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # get block neighborhoods, and prepare a halo-ed version (blocks with padding) for deriving key values

        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)

        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        # derive queries, keys, values

        q = self.to_q(q_inp)
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        # scale

        q *= self.scale

        # attention

        sim = einsum('b i d, b j d -> b i j', q, k)

        # add relative positional bias

        sim += self.rel_pos_emb(q)

        # mask out padding (in the paper, they claim to not need masks, but what about padding?)

        mask = torch.ones(1, 1, h, w, device = device)
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        mask = mask.bool()

        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1)

        # aggregate

        out = einsum('b i j, b j d -> b i d', attn, v)

        # merge and combine heads

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)

        # merge blocks back to original feature map

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        return out

if __name__ == '__main__':
    input=torch.randn(1,512,8,8)
    halo = HaloAttention(dim=512,
        block_size=2,
        halo_size=1,)
    output=halo(input)
    print(output.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init



class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
    output=ssa(input,input,input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict



class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)

        




if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    eca = ECAAttention(kernel_size=3)
    output=eca(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=sa(input,input,input)
    print(output.shape)

    
import torch
from torch import nn
from operator import itemgetter
# from axial_attention.reversible import ReversibleSequence
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim = 1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim = 1)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim = 1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 1)
            dx = torch.cat([dx1, dx2], dim = 1)

        return x, dx

class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim = 1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim = 1)

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route = (True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {'f_args': f_args, 'g_args': g_args}
        x = torch.cat((x, x), dim = 1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim = 1)).mean(dim = 0)


# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)
      
    return permutations

# helper classes

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x

class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

# axial pos emb

class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index = 1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f'param_{i}', parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f'param_{i}')
        return x

# attention

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

# axial attention class

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

# axial image transformer

class AxialImageTransformer(nn.Module):
    def __init__(self, dim, depth, heads = 8, dim_heads = None, dim_index = 1, reversible = True, axial_pos_emb_shape = None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding = 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding = 1)
        )

        self.pos_emb = AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index) if exists(axial_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList([PermuteToFrom(permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))) for permutation in permutations])
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)            

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)

# input=torch.randn(3, 128, 7, 7)

# attn = AxialAttention(
#     dim = 3,               # embedding dimension
#     dim_index = 1,         # where is the embedding dimension
#     dim_heads = 32,        # dimension of each head. defaults to dim // heads if not supplied
#     heads = 1,             # number of heads for multi-head attention
#     num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
#     sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
# )
# print(attn(input).shape) # (1, 3, 256, 256)

if __name__ == '__main__':
    input=torch.randn(3, 128, 7, 7)
    model = AxialImageTransformer(
        dim = 128,
        depth = 12,
        reversible = True
    )
    outputs = model(input)
    print(outputs.shape)
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F



class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    cot = CoTAttention(dim=512,kernel_size=3)
    output=cot(input)
    print(output.shape)

    
import torch
import torch.nn as nn
import torch.nn.functional as F

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride

        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out
        
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

if __name__ == '__main__':
    input=torch.randn(50,256,7,7)
    acmix = ACmix(in_planes=256, out_planes=256)
    output=acmix(input)
    print(output.shape)

import numpy as np
import torch
from torch import nn
from torch.nn import init



class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention(channel=512,reduction=8)
    output=se(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x 


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    sge = SpatialGroupEnhance(groups=8)
    output=sge(input)
    print(output.shape)

    
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Wh-1 * 2Ww-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops

class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        group_size (tuple[int]): The height and width of the group.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, group_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.group_size = group_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias

        if position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            
            # generate mother-set
            position_bias_h = torch.arange(1 - self.group_size[0], self.group_size[0])
            position_bias_w = torch.arange(1 - self.group_size[1], self.group_size[1])
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Wh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).float()
            self.register_buffer("biases", biases)

            # get pair-wise relative position index for each token inside the group
            coords_h = torch.arange(self.group_size[0])
            coords_w = torch.arange(self.group_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.position_bias:
            pos = self.pos(self.biases) # 2Wh-1 * 2Ww-1, heads
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.group_size[0] * self.group_size[1], self.group_size[0] * self.group_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, group_size={self.group_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 group with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        if self.position_bias:
            flops += self.pos.flops(N)
        return flops


class CrossFormerBlock(nn.Module):
    r""" CrossFormer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        group_size (int): Group size.
        lsda_flag (int): use SDA or LDA, 0 for SDA and 1 for LDA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, group_size=7, lsda_flag=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_patch_size=1):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.group_size = group_size
        self.lsda_flag = lsda_flag
        self.mlp_ratio = mlp_ratio
        self.num_patch_size = num_patch_size
        if min(self.input_resolution) <= self.group_size:
            # if group size is larger than input resolution, we don't partition groups
            self.lsda_flag = 0
            self.group_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)

        self.attn = Attention(
            dim, group_size=to_2tuple(self.group_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            position_bias=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size %d, %d, %d" % (L, H, W)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # group embeddings
        G = self.group_size
        if self.lsda_flag == 0: # 0 for SDA
            x = x.reshape(B, H // G, G, W // G, G, C).permute(0, 1, 3, 2, 4, 5)
        else: # 1 for LDA
            x = x.reshape(B, G, H // G, G, W // G, C).permute(0, 2, 4, 1, 3, 5)
        x = x.reshape(B * H * W // G**2, G**2, C)

        # multi-head self-attention
        x = self.attn(x, mask=self.attn_mask)  # nW*B, G*G, C

        # ungroup embeddings
        x = x.reshape(B, H // G, W // G, G, G, C)
        if self.lsda_flag == 0:
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        else:
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, H, W, C)
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"group_size={self.group_size}, lsda_flag={self.lsda_flag}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # LSDA
        nW = H * W / self.group_size / self.group_size
        flops += nW * self.attn.flops(self.group_size * self.group_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, patch_size=[2], num_input_patch_size=1):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim)

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                out_dim = 2 * dim // 2 ** i
            else:
                out_dim = 2 * dim // 2 ** (i + 1)
            stride = 2
            padding = (ps - stride) // 2
            self.reductions.append(nn.Conv2d(dim, out_dim, kernel_size=ps, 
                                                stride=stride, padding=padding))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        xs = []
        for i in range(len(self.reductions)):
            tmp_x = self.reductions[i](x).flatten(2).transpose(1, 2)
            xs.append(tmp_x)
        x = torch.cat(xs, dim=2)
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                out_dim = 2 * self.dim // 2 ** i
            else:
                out_dim = 2 * self.dim // 2 ** (i + 1)
            flops += (H // 2) * (W // 2) * ps * ps * out_dim * self.dim
        return flops


class Stage(nn.Module):
    """ CrossFormer blocks for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        group_size (int): variable G in the paper, one group has GxG embeddings
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, group_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 patch_size_end=[4], num_patch_size=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            lsda_flag = 0 if (i % 2 == 0) else 1
            self.blocks.append(CrossFormerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, group_size=group_size,
                                 lsda_flag=lsda_flag,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 num_patch_size=num_patch_size))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, 
                                         patch_size=patch_size_end, num_input_patch_size=num_patch_size)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: [4].
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.projs = nn.ModuleList()
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = patch_size[0]
            padding = (ps - patch_size[0]) // 2
            self.projs.append(nn.Conv2d(in_chans, dim, kernel_size=ps, stride=stride, padding=padding))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        xs = []
        for i in range(len(self.projs)):
            tx = self.projs[i](x).flatten(2).transpose(1, 2)
            xs.append(tx)  # B Ph*Pw C
        x = torch.cat(xs, dim=2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        for i, ps in enumerate(self.patch_size):
            if i == len(self.patch_size) - 1:
                dim = self.embed_dim // 2 ** i
            else:
                dim = self.embed_dim // 2 ** (i + 1)
            flops += Ho * Wo * dim * self.in_chans * (self.patch_size[i] * self.patch_size[i])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class CrossFormer(nn.Module):
    r""" CrossFormer
        A PyTorch impl of : `CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention`  -

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each stage.
        num_heads (tuple(int)): Number of attention heads in different layers.
        group_size (int): Group size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=[4], in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 group_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, merge_size=[[2], [2], [2]], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()

        num_patch_sizes = [len(patch_size)] + [len(m) for m in merge_size]
        for i_layer in range(self.num_layers):
            patch_size_end = merge_size[i_layer] if i_layer < self.num_layers - 1 else None
            num_patch_size = num_patch_sizes[i_layer]
            layer = Stage(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               group_size=group_size[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               patch_size_end=patch_size_end,
                               num_patch_size=num_patch_size)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CrossFormer(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans= 3,
        num_classes=1000,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2,4], [2, 4]]
    )
    output=model(input)
    print(output.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model

        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ea = ExternalAttention(d_model=512,S=8)
    output=ea(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings
import hiddenlayer as hl


class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,H=7,W=7,ratio=3,apply_transform=True):

        super(EMSA, self).__init__()
        self.H=H
        self.W=W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.ratio=ratio
        if(self.ratio>1):
            self.sr=nn.Sequential()
            self.sr_conv=nn.Conv2d(d_model,d_model,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=d_model)
            self.sr_ln=nn.LayerNorm(d_model)

        self.apply_transform=apply_transform and h>1
        if(self.apply_transform):
            self.transform=nn.Sequential()
            self.transform.add_module('conv',nn.Conv2d(h,h,kernel_size=1,stride=1))
            self.transform.add_module('softmax',nn.Softmax(-1))
            self.transform.add_module('in',nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq ,c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if(self.ratio>1):
            x=queries.permute(0,2,1).view(b_s,c,self.H,self.W) #bs,c,H,W
            x=self.sr_conv(x) #bs,c,h,w
            x=x.contiguous().view(b_s,c,-1).permute(0,2,1) #bs,n',c
            x=self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if(self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att) # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1) # (b_s, h, nq, n')


        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



if __name__ == '__main__':
    input = torch.randn(50, 64, 512)
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
    output = emsa(input, input, input)

    # Visualize the graph using hiddenlayer
    hl_graph = hl.build_graph(emsa, input)
    # hl_graph.save('lightweight', format='png')
    # graph.theme = hl.graph.THEMES['blue']  # Optional: Set the theme for the graph
    hl_graph.save()  # Save the graph in SVG format

    print(output.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) #bs,k,n,c
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=512 ):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        return x

        


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    s2att = S2Attention(channels=512)
    output=s2att(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.shape[0],-1)

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        gate_channels=[channel]
        gate_channels+=[channel//reduction]*num_layers
        gate_channels+=[channel]


        self.ca=nn.Sequential()
        self.ca.add_module('flatten',Flatten())
        for i in range(len(gate_channels)-2):
            self.ca.add_module('fc%d'%i,nn.Linear(gate_channels[i],gate_channels[i+1]))
            self.ca.add_module('bn%d'%i,nn.BatchNorm1d(gate_channels[i+1]))
            self.ca.add_module('relu%d'%i,nn.ReLU())
        self.ca.add_module('last_fc',nn.Linear(gate_channels[-2],gate_channels[-1]))
        

    def forward(self, x) :
        res=self.avgpool(x)
        res=self.ca(res)
        res=res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res

class SpatialAttention(nn.Module):
    def __init__(self,channel,reduction=16,num_layers=3,dia_val=2):
        super().__init__()
        self.sa=nn.Sequential()
        self.sa.add_module('conv_reduce1',nn.Conv2d(kernel_size=1,in_channels=channel,out_channels=channel//reduction))
        self.sa.add_module('bn_reduce1',nn.BatchNorm2d(channel//reduction))
        self.sa.add_module('relu_reduce1',nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d'%i,nn.Conv2d(kernel_size=3,in_channels=channel//reduction,out_channels=channel//reduction,padding=1,dilation=dia_val))
            self.sa.add_module('bn_%d'%i,nn.BatchNorm2d(channel//reduction))
            self.sa.add_module('relu_%d'%i,nn.ReLU())
        self.sa.add_module('last_conv',nn.Conv2d(channel//reduction,1,kernel_size=1))

    def forward(self, x) :
        res=self.sa(x)
        res=res.expand_as(x)
        return res




class BAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,dia_val=2):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(channel=channel,reduction=reduction,dia_val=dia_val)
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    bam = BAMBlock(channel=512,reduction=16,dia_val=2)
    output=bam(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3=nn.Sequential(
            nn.Conv2d(channel,channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu=nn.SiLU()
        

    def forward(self, x):
        b, c, _, _ = x.size()
        x1=self.conv1x1(x)
        x2=self.conv3x3(x)
        x3=self.sse(x)*x
        y=self.silu(x1+x2+x3)
        return y


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import namedtuple
from graphviz import Digraph
import torch
from torch.autograd import Variable
import warnings

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))
excluded_nodes = [] #'Admm','ViewBackward', 'ExpandBackward', 'CloneBackward', 'UnsafeViewBackward'
collapsed_nodes = {}

def make_dot(var, params=None, show_attrs=False, max_attr_chars=50):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes represent the variables that require grad. Orange nodes represent
    saved tensors. Green nodes represent the output tensors. Dark green nodes
    represent the base tensors of output views.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.
    """

    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}
    else:
        param_map = {}

    node_attr = dict(
        style='filled',
        shape='box',
        align='left',
        fontsize='12',
        ranksep='0.1',
        height='0.5',
        fontname='Helvetica',
        fillcolor='#BBDEFB',
        fontcolor='#0D47A1',
    )

    edge_attr = dict(
        style='bold',
        color='#00C853',
        fontname='Helvetica',
    )
    dot = Digraph(node_attr=node_attr, edge_attr=edge_attr, graph_attr=dict(size="12,12"))
    
    seen = set()
    var_color = '#4FC3F7'  # Light Blue 300
    saved_tensor_color = '#FFD54F'  # Amber 300
    output_tensor_color = '#AED581'  # Light Green 300
    base_tensor_color = '#C8E6C9'  # Green 100
    dark_base_tensor_color = '#1B5E20'  # Green 900

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def get_var_name(var, name=None):
        if not name:
            name = param_map[id(var)] if id(var) in param_map else ''
        return '%s\n %s' % (name, size_to_str(var.size()))

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor=output_tensor_color)
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor=var_color)
            else:
                node_name = str(type(var).__name__)
                if 'Backward' in node_name:
                    if node_name not in collapsed_nodes:
                        collapsed_nodes[node_name] = str(id(var))
                        dot.node(str(id(var)), 'Backward', fillcolor='lightgray')
                    else:
                        dot.edge(str(id(var)), collapsed_nodes[node_name], style='dashed', color='gray')
                else:
                    dot.node(str(id(var)), node_name)
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
            if var not in seen:
                if torch.is_tensor(var):
                    dot.node(str(id(var)), size_to_str(var.size()), fillcolor=output_tensor_color)
                elif hasattr(var, 'variable'):
                    u = var.variable
                    name = param_map[id(u)] if params is not None else ''
                    node_name = '%s\n %s' % (name, size_to_str(u.size()))
                    dot.node(str(id(var)), node_name, fillcolor=var_color)
                else:
                    node_name = str(type(var).__name__)
                    if not any(excluded_node in node_name for excluded_node in excluded_nodes):
                        dot.node(str(id(var)), node_name)
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            dot.edge(str(id(u[0])), str(id(var)))
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        dot.edge(str(id(t)), str(id(var)))
                        add_nodes(t)
    def add_base_tensor(var, color=base_tensor_color):
        if var in seen:
            return
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor=color)
        if var.grad_fn:
            add_nodes(var.grad_fn)
            dot.edge(str(id(var.grad_fn)), str(id(var)))
        if var._is_view():
            add_base_tensor(var._base, color=dark_base_tensor_color)
            dot.edge(str(id(var._base)), str(id(var)), style="dotted", color=dark_base_tensor_color)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_base_tensor(v)
    else:
        add_base_tensor(var)

    resize_graph(dot)

    return dot

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.

    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h,dropout=.1,H=7,W=7,ratio=3,apply_transform=True):

        super(EMSA, self).__init__()
        self.H=H
        self.W=W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.ratio=ratio
        if(self.ratio>1):
            self.sr=nn.Sequential()
            self.sr_conv=nn.Conv2d(d_model,d_model,kernel_size=ratio+1,stride=ratio,padding=ratio//2,groups=d_model)
            self.sr_ln=nn.LayerNorm(d_model)

        self.apply_transform=apply_transform and h>1
        if(self.apply_transform):
            self.transform=nn.Sequential()
            self.transform.add_module('conv',nn.Conv2d(h,h,kernel_size=1,stride=1))
            self.transform.add_module('softmax',nn.Softmax(-1))
            self.transform.add_module('in',nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq ,c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if(self.ratio>1):
            x=queries.permute(0,2,1).view(b_s,c,self.H,self.W) #bs,c,H,W
            x=self.sr_conv(x) #bs,c,h,w
            x=x.contiguous().view(b_s,c,-1).permute(0,2,1) #bs,n',c
            x=self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if(self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att) # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1) # (b_s, h, nq, n')


        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out



if __name__ == '__main__':
    input = torch.randn(50, 64, 512)
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True)
    output = emsa(input, input, input)

    # Visualize the graph
    dot = make_dot(output, params=dict(emsa.named_parameters()))
    # Attempt to modify the canvas size
    dot.graph_attr['size'] = "120,120"  # Set the size as width,height in inches

   
    dot.render('emsa_graph', format='svg')  # Save the graph as PNG
    print(output.shape)
    
'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)
    model = CrissCrossAttention(64)
    outputs = model(input)
    print(outputs.shape)
import numpy as np
import torch
from torch import nn
from torch.nn import init



class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out






class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(channel_out) #bs,c//2,h,w
        spatial_wq=self.sp_wq(channel_out) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*channel_out
        return spatial_out




if __name__ == '__main__':
    input=torch.randn(1,512,7,7)
    psa = SequentialPolarizedSelfAttention(channel=512)
    output=psa(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict



class SKAttention(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V

        




if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SKAttention(channel=512,reduction=8)
    output=se(input)
    print(output.shape)

    
import numpy as np
import torch
from torch import nn
from torch.nn import init
import math
from torch.nn import functional as F

class OutlookAttention(nn.Module):

    def __init__(self,dim,num_heads=1,kernel_size=3,padding=1,stride=1,qkv_bias=False,
                    attn_drop=0.1):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.head_dim=dim//num_heads
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.scale=self.head_dim**(-0.5)

        self.v_pj=nn.Linear(dim,dim,bias=qkv_bias)
        self.attn=nn.Linear(dim,kernel_size**4*num_heads)

        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(attn_drop)

        self.unflod=nn.Unfold(kernel_size,padding,stride) #手动卷积
        self.pool=nn.AvgPool2d(kernel_size=stride,stride=stride,ceil_mode=True) 

    def forward(self, x) :
        B,H,W,C=x.shape

        #映射到新的特征v
        v=self.v_pj(x).permute(0,3,1,2) #B,C,H,W
        h,w=math.ceil(H/self.stride),math.ceil(W/self.stride)
        v=self.unflod(v).reshape(B,self.num_heads,self.head_dim,self.kernel_size*self.kernel_size,h*w).permute(0,1,4,3,2) #B,num_head,H*W,kxk,head_dim

        #生成Attention Map
        attn=self.pool(x.permute(0,3,1,2)).permute(0,2,3,1) #B,H,W,C
        attn=self.attn(attn).reshape(B,h*w,self.num_heads,self.kernel_size*self.kernel_size \
                    ,self.kernel_size*self.kernel_size).permute(0,2,1,3,4) #B，num_head，H*W,kxk,kxk
        attn=self.scale*attn
        attn=attn.softmax(-1)
        attn=self.attn_drop(attn)

        #获取weighted特征
        out=(attn @ v).permute(0,1,4,3,2).reshape(B,C*self.kernel_size*self.kernel_size,h*w) #B,dimxkxk,H*W
        out=F.fold(out,output_size=(H,W),kernel_size=self.kernel_size,
                    padding=self.padding,stride=self.stride) #B,C,H,W
        out=self.proj(out.permute(0,2,3,1)) #B,H,W,C
        out=self.proj_drop(out)

        return out

        

if __name__ == '__main__':
    input=torch.randn(50,28,28,512)
    outlook = OutlookAttention(dim=512)
    output=outlook(input)
    print(output.shape)





import numpy as np
import torch
from torch import nn
from torch.functional import norm
from torch.nn import init


def XNorm(x,gamma):
    norm_tensor=torch.norm(x,2,-1,True)
    return x*gamma/norm_tensor


class UFOAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(UFOAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)
        self.gamma=nn.Parameter(torch.randn((1,h,1,1)))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        kv=torch.matmul(k, v) #bs,h,c,c
        kv_norm=XNorm(kv,self.gamma) #bs,h,c,c
        q_norm=XNorm(q,self.gamma) #bs,h,n,c
        out=torch.matmul(q_norm,kv_norm).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        
        return out


if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ufo = UFOAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=ufo(input,input,input)
    print(output.shape)

    
