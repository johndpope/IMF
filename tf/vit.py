import tensorflow as tf
from tensorflow.keras import layers
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        position = tf.range(max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-math.log(10000.0) / d_model))
        pe = tf.zeros((max_len, 1, d_model))
        pe = tf.Variable(pe, trainable=False)
        pe = tf.tensor_scatter_nd_update(pe, indices=[[i, 0, j] for i in range(max_len) for j in range(0, d_model, 2)],
                                         updates=tf.sin(position * div_term))
        pe = tf.tensor_scatter_nd_update(pe, indices=[[i, 0, j] for i in range(max_len) for j in range(1, d_model, 2)],
                                         updates=tf.cos(position * div_term))
        self.pe = pe

    def call(self, x):
        return self.pe[:tf.shape(x)[0], :]

# Transformer Block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=dim_head)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim),
            layers.Activation(tf.nn.gelu),
            layers.Dense(dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        B, H, W, C = x.shape
        x_reshaped = tf.reshape(x, (B, H*W, C))
        x_norm = self.norm1(x_reshaped)
        att_output = self.attention(x_norm, x_norm, x_norm)
        x_reshaped = x_reshaped + att_output
        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output
        output = tf.reshape(x_reshaped, (B, H, W, C))
        return output
class CrossAttentionModule(tf.keras.layers.Layer):
    def __init__(self, dim_spatial=4096, dim_qk=256, dim_v=256):
        super().__init__()
        self.dim_qk = dim_qk
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
        self.attend = tf.keras.layers.Softmax(axis=-1)

    def call(self, queries, keys, values):
        q = tf.reshape(tf.transpose(queries, [0, 2, 3, 1]), (-1, tf.shape(queries)[1], self.dim_qk))
        q = q + self.q_pos_embedding
        k = tf.reshape(tf.transpose(keys, [0, 2, 3, 1]), (-1, tf.shape(keys)[1], self.dim_qk))
        k = k + self.k_pos_embedding
        v = tf.reshape(tf.transpose(values, [0, 2, 3, 1]), (-1, tf.shape(values)[1], tf.shape(values)[-1]))

        dots = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = self.attend(dots)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1]), tf.shape(values))
        return out

class ImplicitMotionAlignment(tf.keras.layers.Layer):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, dim_head=64, mlp_dim=1024):
        super().__init__()
        self.cross_attention = CrossAttentionModule(dim_spatial=spatial_dim[0] * spatial_dim[1],
                                                    dim_qk=motion_dim, dim_v=feature_dim)
        self.transformer_blocks = [TransformerBlock(feature_dim, heads, dim_head, mlp_dim) for _ in range(depth)]
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim

    def call(self, ml_c, ml_r, fl_r):
        v_prime = self.cross_attention(ml_c, ml_r, fl_r)
        for block in self.transformer_blocks:
            v_prime = block(v_prime)
        return v_prime
    
    @staticmethod
    def visualize_embeddings(embeddings, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, (title, emb) in enumerate(embeddings):
            if i >= len(axes):
                break
            emb = tf.reshape(emb, [-1, emb.shape[-1]]).numpy()
            if emb.shape[0] > 10000:
                indices = np.random.choice(emb.shape[0], 10000, replace=False)
                emb = emb[indices]

            tsne = TSNE(n_components=2, random_state=42)
            emb_2d = tsne.fit_transform(emb)

            axes[i].scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5)
            axes[i].set_title(title)
            axes[i].set_xticks([])
            axes[i].set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# Example usage
if __name__ == "__main__":
    # Set device
    device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
    print(f"Using device: {device}")

    # Example dimensions
    B, H, W = 1, 64, 64
    CF = 256  # Feature dimension
    CM = 256  # Motion dimension
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024

    # Create random input tensors
    mlC = tf.random.normal((B, H, W, CM))
    mlR = tf.random.normal((B, H, W, CM))
    flR = tf.random.normal((B, H, W, CF))

    # Initialize the ImplicitMotionAlignment module
    model = ImplicitMotionAlignment(CF, CM, (H, W), depth, heads, dim_head, mlp_dim)

    # Forward pass
    with tf.device(device):
        output = model(mlC, mlR, flR)

    print(f"Input shapes: mlC: {mlC.shape}, mlR: {mlR.shape}, flR: {flR.shape}")
    print(f"Output shape: {output.shape}")

    # Visualize embeddings (if embeddings are collected)
    # embeddings = [("Layer Name", tensor.numpy())]  # Replace with actual embeddings
    # ImplicitMotionAlignment.visualize_embeddings(embeddings, "embeddings_visualization.png")
    # print("Embedding visualization saved as 'embeddings_visualization.png'")
