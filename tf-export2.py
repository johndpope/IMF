import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow import keras
import numpy as np

from vit import CrossAttentionModule
from rich.console import Console
from rich.traceback import install
from nobuco import converter, ChannelOrderingStrategy
# Create a Console instance
console = Console(width=3000)

# Install Rich traceback handling
# install(show_locals=True)
install()


class KerasCrossAttentionModule(keras.layers.Layer):
    def __init__(self, dim_spatial=4096, dim_qk=256, dim_v=256):
        super().__init__()
        self.dim_spatial = dim_spatial
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.dim_head = dim_qk
        self.scale = tf.math.sqrt(tf.cast(dim_qk, tf.float32))

        # Separate positional encodings for queries and keys
        self.q_pos_embedding = self.add_weight(
            "q_pos_embedding",
            shape=(1, dim_spatial, dim_qk),
            initializer="random_normal",
            trainable=True
        )
        self.k_pos_embedding = self.add_weight(
            "k_pos_embedding",
            shape=(1, dim_spatial, dim_qk),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        queries, keys, values = inputs

        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        q = tf.reshape(queries, (-1, self.dim_qk, self.dim_spatial))
        q = tf.transpose(q, [0, 2, 1])
        q = q + self.q_pos_embedding  # (b, dim_spatial, dim_qk)

        # (b, dim_qk, h, w) -> (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_qk)
        k = tf.reshape(keys, (-1, self.dim_qk, self.dim_spatial))
        k = tf.transpose(k, [0, 2, 1])
        k = k + self.k_pos_embedding  # (b, dim_spatial, dim_qk)

        # (b, dim_v, h, w) -> (b, dim_v, dim_spatial) -> (b, dim_spatial, dim_v)
        v = tf.reshape(values, (-1, self.dim_v, self.dim_spatial))
        v = tf.transpose(v, [0, 2, 1])

        # (b, dim_spatial, dim_qk) * (b, dim_qk, dim_spatial) -> (b, dim_spatial, dim_spatial)
        dots = tf.matmul(q, k, transpose_b=True) / self.scale

        attn = tf.nn.softmax(dots, axis=-1)  # (b, dim_spatial, dim_spatial)

        # (b, dim_spatial, dim_spatial) * (b, dim_spatial, dim_v) -> (b, dim_spatial, dim_v)
        out = tf.matmul(attn, v)

        # Reshape back to original dimensions
        out = tf.transpose(out, [0, 2, 1])  # (b, dim_v, dim_spatial)
        h = w = int(tf.math.sqrt(tf.cast(self.dim_spatial, tf.float32)))
        out = tf.reshape(out, (-1, self.dim_v, h, w))

        return out
    
class MultiheadAttentionModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        print(f"\n⚾ MultiheadAttentionModel input shape: {x.shape}")
        x = x.permute(1, 0, 2)
        print(f"After permute shape: {x.shape}")
        attn_output, _ = self.multihead_attn(x, x, x)
        print(f"Attention output shape: {attn_output.shape}")
        return attn_output.permute(1, 0, 2)

class KerasMultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, value_dim=d_model // num_heads, use_bias=True)
                
    def call(self, x):
        x = tf.transpose(x, [1, 0, 2])
        attn_output = self.mha(x, x)
        return tf.transpose(attn_output, [1, 0, 2])

@converter(MultiheadAttentionModel, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_MultiheadAttentionModel(self, x):
    d_model = self.multihead_attn.embed_dim
    num_heads = self.multihead_attn.num_heads
    head_dim = d_model // num_heads
    keras_mha = KerasMultiHeadAttention(d_model=d_model, num_heads=num_heads)
    
    # Build the layer
    dummy_input = tf.keras.Input(shape=(None, d_model))
    keras_mha(dummy_input)
    
    # Transfer weights
    in_proj_weight = self.multihead_attn.in_proj_weight.detach().numpy()
    in_proj_bias = self.multihead_attn.in_proj_bias.detach().numpy()
    out_proj_weight = self.multihead_attn.out_proj.weight.detach().numpy()
    out_proj_bias = self.multihead_attn.out_proj.bias.detach().numpy()
    
    # Split weights for query, key, and value
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3)
    
    # Reshape weights to match Keras MultiHeadAttention expectations
    q_weight = q_weight.T.reshape(d_model, num_heads, head_dim)
    k_weight = k_weight.T.reshape(d_model, num_heads, head_dim)
    v_weight = v_weight.T.reshape(d_model, num_heads, head_dim)
    out_proj_weight = out_proj_weight.T

    print("MultiheadAttentionModel q_weight shape:", q_weight.shape)
    print("k_weight shape:", k_weight.shape)
    print("v_weight shape:", v_weight.shape)
    print("out_proj_weight shape:", out_proj_weight.shape)
    print("q_bias shape:", q_bias.shape)
    print("k_bias shape:", k_bias.shape)
    print("v_bias shape:", v_bias.shape)
    print("out_proj_bias shape:", out_proj_bias.shape)
    
    keras_mha.mha.set_weights([
        q_weight, k_weight, v_weight,
        out_proj_weight,
        q_bias, k_bias, v_bias,
        out_proj_bias
    ])
    
    return keras_mha



class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = MultiheadAttentionModel(embed_dim=dim, num_heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        print(f"\n🎑 TransformerBlock input shape: {x.shape}")
        B, C, H, W = x.shape
        x_reshaped = x.view(B, C, H*W).permute(0, 2, 1)
        print(f"After reshape and permute: {x_reshaped.shape}")

        
        x_norm = self.norm1(x_reshaped)
        att_output = self.attention(x_norm)
        x_reshaped = x_reshaped + att_output

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        return x_reshaped.permute(0, 2, 1).view(B, C, H, W)

class KerasTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = KerasMultiHeadAttention(d_model=dim, num_heads=heads)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim),
            keras.layers.Activation('gelu'),
            keras.layers.Dense(dim)
        ])
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x):
        B, C, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_reshaped = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), (B, H * W, C))
        
        x_norm = self.norm1(x_reshaped)
        att_output = self.attention(x_norm)
        x_reshaped = x_reshaped + att_output

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        return tf.transpose(tf.reshape(x_reshaped, (B, H, W, C)), [0, 3, 1, 2])

@converter(TransformerBlock, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_TransformerBlock(self, x):
    keras_block = KerasTransformerBlock(
        dim=self.attention.multihead_attn.embed_dim,
        heads=self.attention.multihead_attn.num_heads,
        dim_head=self.attention.multihead_attn.head_dim,
        mlp_dim=self.mlp[0].out_features
    )
    
    # Build the Keras block by calling it with dummy input
    input_shape = x.shape
    print(f"Input shape to TransformerBlock: {input_shape}")
    
    dummy_input = tf.zeros(input_shape)
    keras_block(dummy_input)
    
    # Transfer weights for MultiheadAttention
    pytorch_mha = self.attention.multihead_attn
    keras_mha = keras_block.attention.mha
    
    d_model = pytorch_mha.embed_dim
    num_heads = pytorch_mha.num_heads
    head_dim = d_model // num_heads
    
    # Print model information
    print(f"MultiheadAttention d_model: {d_model}, num_heads: {num_heads}, head_dim: {head_dim}")
    
    in_proj_weight = pytorch_mha.in_proj_weight.detach().numpy()
    in_proj_bias = pytorch_mha.in_proj_bias.detach().numpy()
    out_proj_weight = pytorch_mha.out_proj.weight.detach().numpy()
    out_proj_bias = pytorch_mha.out_proj.bias.detach().numpy()
    
    # Split and reshape the weights
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3)
    
    # Reshape the weights
    q_weight = q_weight.T.reshape(d_model, num_heads, head_dim)
    k_weight = k_weight.T.reshape(d_model, num_heads, head_dim)
    v_weight = v_weight.T.reshape(d_model, num_heads, head_dim)
    out_proj_weight = out_proj_weight.T
    
    # Print shapes after reshaping
    print(f"q_weight shape after reshape: {q_weight.shape}")  # (256, 8, 32)
    print(f"k_weight shape after reshape: {k_weight.shape}")  # (256, 8, 32)
    print(f"v_weight shape after reshape: {v_weight.shape}")  # (256, 8, 32)
    
    # Set the weights in the Keras model
    keras_mha.set_weights([
        q_weight, k_weight, v_weight,
        out_proj_weight,
        q_bias, k_bias, v_bias,
        out_proj_bias
    ])
    

    # Transfer LayerNorm weights
    keras_block.norm1.set_weights([
        self.norm1.weight.detach().numpy(),
        self.norm1.bias.detach().numpy()
    ])
    keras_block.norm2.set_weights([
        self.norm2.weight.detach().numpy(),
        self.norm2.bias.detach().numpy()
    ])

    # Transfer MLP weights
    keras_block.mlp.layers[0].set_weights([
        self.mlp[0].weight.detach().numpy().T,
        self.mlp[0].bias.detach().numpy()
    ])
    keras_block.mlp.layers[2].set_weights([
        self.mlp[2].weight.detach().numpy().T,
        self.mlp[2].bias.detach().numpy()
    ])
    
    return keras_block





class ImplicitMotionAlignment(nn.Module):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[1], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)
        ])
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim

    def forward(self, ml_c, ml_r, fl_r):
        V_prime = self.cross_attention(ml_c, ml_r, fl_r)
        for block in self.transformer_blocks:
            V_prime = block(V_prime)
        return V_prime

class KerasImplicitMotionAlignment(keras.layers.Layer):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = KerasCrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[1], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = [KerasTransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)]

    def call(self, inputs):
        ml_c, ml_r, fl_r = inputs
        V_prime = self.cross_attention([ml_c, ml_r, fl_r])
        for block in self.transformer_blocks:
            V_prime = block(V_prime)
        return V_prime


@converter(ImplicitMotionAlignment, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_ImplicitMotionAlignment(self, ml_c, ml_r, fl_r):
    feature_dim = self.feature_dim
    motion_dim = self.motion_dim
    spatial_dim = self.spatial_dim
    depth = len(self.transformer_blocks)
    heads = self.transformer_blocks[0].attention.multihead_attn.num_heads
    dim_head = self.transformer_blocks[0].attention.multihead_attn.head_dim
    mlp_dim = self.transformer_blocks[0].mlp[0].out_features

    print(f"ImplicitMotionAlignment parameters:")
    print(f"feature_dim: {feature_dim}, motion_dim: {motion_dim}, spatial_dim: {spatial_dim}")
    print(f"depth: {depth}, heads: {heads}, dim_head: {dim_head}, mlp_dim: {mlp_dim}")

    keras_model = KerasImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim, depth, heads, dim_head, mlp_dim)
    
    # Build the Keras model by calling it with dummy inputs
    dummy_ml_c = tf.zeros_like(ml_c)
    dummy_ml_r = tf.zeros_like(ml_r)
    dummy_fl_r = tf.zeros_like(fl_r)
    keras_model([dummy_ml_c, dummy_ml_r, dummy_fl_r])
    
    # Transfer weights for CrossAttentionModule
    q_pos_embedding = self.cross_attention.q_pos_embedding.detach().numpy()
    k_pos_embedding = self.cross_attention.k_pos_embedding.detach().numpy()
    print(f"CrossAttentionModule:")
    print(f"q_pos_embedding shape: {q_pos_embedding.shape}")
    print(f"k_pos_embedding shape: {k_pos_embedding.shape}")
    keras_model.cross_attention.set_weights([q_pos_embedding, k_pos_embedding])
    
    # Transfer weights for TransformerBlocks
    for i, (pytorch_block, keras_block) in enumerate(zip(self.transformer_blocks, keras_model.transformer_blocks)):
        print(f"\nTransformerBlock {i}:")
        
        # Transfer LayerNorm weights
        norm1_weight = pytorch_block.norm1.weight.detach().numpy()
        norm1_bias = pytorch_block.norm1.bias.detach().numpy()
        norm2_weight = pytorch_block.norm2.weight.detach().numpy()
        norm2_bias = pytorch_block.norm2.bias.detach().numpy()
        print(f"norm1_weight shape: {norm1_weight.shape}, norm1_bias shape: {norm1_bias.shape}")
        print(f"norm2_weight shape: {norm2_weight.shape}, norm2_bias shape: {norm2_bias.shape}")
        keras_block.norm1.set_weights([norm1_weight, norm1_bias])
        keras_block.norm2.set_weights([norm2_weight, norm2_bias])

        # Transfer MLP weights
        mlp1_weight = pytorch_block.mlp[0].weight.detach().numpy().T
        mlp1_bias = pytorch_block.mlp[0].bias.detach().numpy()
        mlp2_weight = pytorch_block.mlp[2].weight.detach().numpy().T
        mlp2_bias = pytorch_block.mlp[2].bias.detach().numpy()
        print(f"mlp1_weight shape: {mlp1_weight.shape}, mlp1_bias shape: {mlp1_bias.shape}")
        print(f"mlp2_weight shape: {mlp2_weight.shape}, mlp2_bias shape: {mlp2_bias.shape}")
        keras_block.mlp.layers[0].set_weights([mlp1_weight, mlp1_bias])
        keras_block.mlp.layers[2].set_weights([mlp2_weight, mlp2_bias])

        # Transfer MultiheadAttention weights
        pytorch_mha = pytorch_block.attention.multihead_attn
        keras_mha = keras_block.attention.mha
        
        d_model = pytorch_mha.embed_dim
        num_heads = pytorch_mha.num_heads
        head_dim = d_model // num_heads
        
        in_proj_weight = pytorch_mha.in_proj_weight.detach().numpy()
        in_proj_bias = pytorch_mha.in_proj_bias.detach().numpy()
        out_proj_weight = pytorch_mha.out_proj.weight.detach().numpy()
        out_proj_bias = pytorch_mha.out_proj.bias.detach().numpy()
        
        q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = np.split(in_proj_bias, 3)
        
        # Reshape weights
        # Reshape weights
        q_weight = q_weight.T.reshape(d_model, num_heads, head_dim)
        k_weight = k_weight.T.reshape(d_model, num_heads, head_dim)
        v_weight = v_weight.T.reshape(d_model, num_heads, head_dim)
        out_proj_weight = out_proj_weight.T

        print(f"ImplicitMotionAlignment MultiheadAttention: i= {i}")
        print(f"q_weight shape: {q_weight.shape}")
        print(f"k_weight shape: {k_weight.shape}")
        print(f"v_weight shape: {v_weight.shape}")
        print(f"out_proj_weight shape: {out_proj_weight.shape}")
        print(f"q_bias shape: {q_bias.shape}")
        print(f"k_bias shape: {k_bias.shape}")
        print(f"v_bias shape: {v_bias.shape}")
        print(f"out_proj_bias shape: {out_proj_bias.shape}")

        keras_mha.set_weights([
            q_weight, k_weight, v_weight,
            out_proj_weight,
            q_bias, k_bias, v_bias,
            out_proj_bias
        ])
    return keras_model

# Add CrossAttentionModule and its Keras equivalent here
# (Implementation details omitted for brevity)

# Main conversion function
def convert_implicit_motion_alignment(pytorch_model):
    # Create dummy inputs
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    ml_c = torch.randn(B, C_m, H, W)
    ml_r = torch.randn(B, C_m, H, W)
    fl_r = torch.randn(B, C_f, H, W)

    keras_model = nobuco.pytorch_to_keras(
        pytorch_model,
        args=[ml_c, ml_r, fl_r],
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH
    )

    return keras_model

@nobuco.converter(CrossAttentionModule, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_CrossAttentionModule(self, queries, keys, values):
    keras_module = KerasCrossAttentionModule(
        dim_spatial=self.dim_spatial,
        dim_qk=self.dim_head,
        dim_v=values.shape[1]
    )
    
    # Build the layer
    dummy_queries = tf.zeros_like(queries)
    dummy_keys = tf.zeros_like(keys)
    dummy_values = tf.zeros_like(values)
    keras_module([dummy_queries, dummy_keys, dummy_values])
    
    # Transfer weights
    keras_module.set_weights([
        self.q_pos_embedding.detach().numpy(),
        self.k_pos_embedding.detach().numpy()
    ])
    
    return keras_module


# Usage example
if __name__ == "__main__":
    # Initialize your PyTorch model
    feature_dim, motion_dim, spatial_dim = 256, 256, (64, 64)
    pytorch_model = ImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim)

    # Convert to Keras
    keras_model = convert_implicit_motion_alignment(pytorch_model)

    # Save the Keras model
    keras_model.save("implicit_motion_alignment.keras")
    print("Keras model saved successfully.")