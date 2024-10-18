import tensorflow as tf
from tensorflow import keras
import numpy as np

class KerasCrossAttentionModule(keras.layers.Layer):
    def __init__(self, dim_spatial=4096, dim_qk=256, dim_v=256):
        super().__init__()
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.scale = dim_qk ** -0.5
        self.q_pos_embedding = self.add_weight(
            name="q_pos_embedding",
            shape=(1, dim_spatial, dim_qk),
            initializer='random_normal',
            trainable=True
        )
        self.k_pos_embedding = self.add_weight(
            name="k_pos_embedding",
            shape=(1, dim_spatial, dim_qk),
            initializer='random_normal',
            trainable=True
        )
        self.attend = keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        queries, keys, values = inputs
        print(f"KerasCrossAttentionModule input shapes: queries: {queries.shape}, keys: {keys.shape}, values: {values.shape}")
        
        q = tf.reshape(tf.transpose(queries, [0, 2, 3, 1]), (-1, tf.shape(queries)[2] * tf.shape(queries)[3], self.dim_qk))
        q = q + self.q_pos_embedding
        k = tf.reshape(tf.transpose(keys, [0, 2, 3, 1]), (-1, tf.shape(keys)[2] * tf.shape(keys)[3], self.dim_qk))
        k = k + self.k_pos_embedding
        v = tf.reshape(tf.transpose(values, [0, 2, 3, 1]), (-1, tf.shape(values)[2] * tf.shape(values)[3], self.dim_v))

        dots = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = self.attend(dots)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1]), tf.shape(values))
        print(f"KerasCrossAttentionModule output shape: {out.shape}")
        return out


    def __repr__(self):
        return (f"KerasCrossAttentionModule(dim_spatial={self.dim_spatial}, dim_qk={self.dim_qk}, "
                f"dim_v={self.dim_v}, scale={self.scale})\n"
                f"Query Positional Embedding Shape: {self.q_pos_embedding.shape}\n"
                f"Key Positional Embedding Shape: {self.k_pos_embedding.shape}\n")
    

class KerasMultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
                
    def call(self, x):
        return self.mha(x, x)

class KerasTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = KerasMultiHeadAttention(d_model=dim, num_heads=heads)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim),
            keras.layers.Activation(tf.nn.gelu),
            keras.layers.Dense(dim)
        ])
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.inputs = None  # Add this line
        self.dim = dim
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
    def call(self, x):
        self.inputs = x  # Add this line

        print(f"KerasTransformerBlock input shape: {x.shape}")
        B, C, H, W = tf.unstack(tf.shape(x))
        x_reshaped = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), (B, H*W, C))
        
        x_norm = self.norm1(x_reshaped)
        att_output = self.attention(x_norm)
        x_reshaped = x_reshaped + att_output
        print(f"KerasTransformerBlock: After attention, x_reshaped.shape = {x_reshaped.shape}")

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output
        print(f"KerasTransformerBlock: After feedforward, x_reshaped.shape = {x_reshaped.shape}")

        output = tf.reshape(tf.transpose(x_reshaped, [0, 2, 1]), (B, C, H, W))
        print(f"KerasTransformerBlock output shape: {output.shape}")
        return output


    def __repr__(self):
            # Define how the object is represented
            return (f"KerasTransformerBlock(dim={self.dim}, heads={self.heads}, "
                    f"dim_head={self.dim_head}, mlp_dim={self.mlp_dim})\n"
                    f"Norm Layers: norm1={self.norm1}, norm2={self.norm2}\n"
                    f"MLP: {self.mlp}\n"
                    f"Attention: {self.attention}\n"
                    f"Last Input Shape: {self.inputs.shape if self.inputs is not None else 'None'}")


class KerasImplicitMotionAlignment(keras.Model):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = KerasCrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[1], dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = [KerasTransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)]

    def call(self, inputs):
        ml_c, ml_r, fl_r = inputs
        print(f"KerasImplicitMotionAlignment input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        V_prime = self.cross_attention([ml_c, ml_r, fl_r])
        print(f"After cross-attention shape: {V_prime.shape}")
        for i, block in enumerate(self.transformer_blocks):
            V_prime = block(V_prime)
            print(f"After transformer block {i} shape: {V_prime.shape}")
        return V_prime

    def __repr__(self):
        return (f"KerasImplicitMotionAlignment(feature_dim={self.feature_dim}, motion_dim={self.motion_dim}, "
                f"spatial_dim={self.spatial_dim}, depth={self.depth}, heads={self.heads}, "
                f"dim_head={self.dim_head}, mlp_dim={self.mlp_dim})\n"
                f"Cross Attention: {self.cross_attention}\n"
                f"Transformer Blocks: {[block for block in self.transformer_blocks]}")



if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Example dimensions
    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024
    spatial_dim = (64, 64)

    # Create random input tensors (in PyTorch order)
    ml_c = tf.random.normal((B, C_m, H, W))
    ml_r = tf.random.normal((B, C_m, H, W))
    fl_r = tf.random.normal((B, C_f, H, W))

    # Initialize the KerasImplicitMotionAlignment model
    model = KerasImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim, depth, heads, dim_head, mlp_dim)

    # Forward pass
    output = model([ml_c, ml_r, fl_r])

    print(f"\nFinal output shape: {output.shape}")