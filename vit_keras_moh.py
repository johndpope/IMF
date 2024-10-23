import tensorflow as tf
from tensorflow import keras
import numpy as np

class KerasCrossAttentionModule(tf.keras.layers.Layer):
    def __init__(self, dim_spatial=4096, dim_qk=256, dim_v=256):
        super().__init__()
        self.dim_spatial = dim_spatial
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
        self.attend = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        queries, keys, values = inputs
        
        q = tf.reshape(tf.transpose(queries, [0, 2, 3, 1]), (-1, tf.shape(queries)[2] * tf.shape(queries)[3], self.dim_qk))
        q = q + self.q_pos_embedding
        k = tf.reshape(tf.transpose(keys, [0, 2, 3, 1]), (-1, tf.shape(keys)[2] * tf.shape(keys)[3], self.dim_qk))
        k = k + self.k_pos_embedding
        v = tf.reshape(tf.transpose(values, [0, 2, 3, 1]), (-1, tf.shape(values)[2] * tf.shape(values)[3], self.dim_v))

        dots = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = self.attend(dots)
        out = tf.matmul(attn, v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1]), tf.shape(values))
        
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim_spatial": self.dim_spatial,
            "dim_qk": self.dim_qk,
            "dim_v": self.dim_v,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class KerasMoHAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, shared_heads=1, routed_heads=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shared_heads = shared_heads
        self.routed_heads = routed_heads
        self.head_dim = dim // num_heads
        
        print(f"\nInitializing KerasMoHAttention:")
        print(f"dim={dim}, num_heads={num_heads}, shared_heads={shared_heads}, routed_heads={routed_heads}")
        print(f"head_dim={self.head_dim}")

        # Linear projections
        self.q = tf.keras.layers.Dense(dim)
        self.k = tf.keras.layers.Dense(dim)
        self.v = tf.keras.layers.Dense(dim)
        self.proj = tf.keras.layers.Dense(dim)

        # Routing layers
        self.router = tf.keras.layers.Dense(num_heads - shared_heads)
        self.shared_router = tf.keras.layers.Dense(2)
        
        # Learnable parameters
        self.temperature = self.add_weight(
            "temperature",
            shape=(num_heads, 1, 1),
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True
        )
        self.query_embedding = self.add_weight(
            "query_embedding",
            shape=(num_heads, 1, self.head_dim),
            initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            trainable=True
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        print(f"\nInput shape: {x.shape}")

        # Linear projections with reshape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        print(f"After linear projections - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Reshape to multi-head format
        q = tf.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = tf.transpose(v, [0, 2, 1, 3])
        print(f"After reshape to multi-head - q: {q.shape}, k: {k.shape}, v: {v.shape}")

        # Normalize query and add embedding
        q_norm = tf.math.l2_normalize(q, axis=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * tf.nn.softplus(self.temperature)
        print(f"q_norm shape: {q_norm.shape}")
        print(f"q_norm_scaled shape: {q_norm_scaled.shape}")

        # Compute attention scores
        attn = tf.matmul(q_norm_scaled, k, transpose_b=True)
        attn = attn / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)
        print(f"Attention scores shape: {attn.shape}")

        # Apply attention to values
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [batch_size, seq_len, self.dim])
        print(f"After attention shape: {x.shape}")

        # Compute routing gates
        logits = self.router(x)  # [B, N, H-S]
        gates = tf.nn.softmax(logits, axis=-1)
        print(f"Router logits shape: {logits.shape}")
        print(f"Gates shape: {gates.shape}")

        # Get top-k routing heads and create mask
        _, indices = tf.math.top_k(gates, k=self.routed_heads)  # [B, N, R]
        print(f"Top-k indices shape: {indices.shape}")
        
        # Create one-hot encodings maintaining sequence dimension
        one_hot_indices = tf.one_hot(indices, self.num_heads - self.shared_heads)  # [B, N, R, H-S]
        print(f"One-hot indices shape: {one_hot_indices.shape}")
        
        # Sum along the routed_heads dimension (axis=2) to get mask
        mask = tf.reduce_sum(one_hot_indices, axis=2)  # [B, N, H-S]
        print(f"Mask shape: {mask.shape}")
        
        print(f"Gates min/max: {tf.reduce_min(gates):.4f}/{tf.reduce_max(gates):.4f}")
        print(f"Mask min/max: {tf.reduce_min(mask):.4f}/{tf.reduce_max(mask):.4f}")
        
        # Apply routing gates
        print(f"Gates broadcasting shape check - gates: {gates.shape}, mask: {mask.shape}")
        routed_head_gates = gates * mask  # Both are [B, N, H-S]
        routed_head_gates = routed_head_gates * tf.cast(self.routed_heads, tf.float32)
        print(f"Routed head gates shape: {routed_head_gates.shape}")

        # Compute shared head weights
        shared_head_weight = self.shared_router(x)  # [B, N, 2]
        shared_head_gates = tf.nn.softmax(shared_head_weight, axis=-1) * tf.cast(self.shared_heads, tf.float32)
        
        # Split routing weights
        weight_0 = tf.nn.softmax(shared_head_weight, axis=-1) * 2.0
        
        # Apply weights to gates
        shared_head_gates = tf.expand_dims(weight_0[..., 0], -1) * shared_head_gates
        routed_head_gates = tf.expand_dims(weight_0[..., 1], -1) * routed_head_gates
        
        # Combine gates
        masked_gates = tf.concat([shared_head_gates, routed_head_gates], axis=-1)
        print(f"Final masked gates shape: {masked_gates.shape}")
        
        # Apply final transformation
        x = tf.einsum('bnh,bnd->bnd', masked_gates, x)
        
        return self.proj(x)
    
class KerasTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, shared_heads=1, routed_heads=3, mlp_dim=1024):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.shared_heads = shared_heads
        self.routed_heads = routed_heads
        self.mlp_dim = mlp_dim

        self.attention = KerasMoHAttention(
            dim=dim,
            num_heads=num_heads,
            shared_heads=shared_heads,
            routed_heads=routed_heads
        )
        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(dim)
        ])

    def call(self, x):
        # Get input shape
        batch_size = tf.shape(x)[0]
        
        # Handle spatial dimensions
        if len(x.shape) == 4:
            # Input is [B, C, H, W]
            height = tf.shape(x)[2]
            width = tf.shape(x)[3]
            channels = x.shape[1]
            
            # Reshape to [B, H*W, C]
            x = tf.transpose(x, [0, 2, 3, 1])  # [B, H, W, C]
            x = tf.reshape(x, [batch_size, height * width, channels])
        
        # Layer Norm 1 and Attention
        attended = self.attention(self.norm1(x))
        x = x + attended
        
        # Layer Norm 2 and MLP
        output = self.mlp(self.norm2(x))
        x = x + output
        
        # Reshape back if needed
        if len(tf.shape(attended)) == 3:
            x = tf.reshape(x, [batch_size, height, width, channels])
            x = tf.transpose(x, [0, 3, 1, 2])  # [B, C, H, W]
            
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "shared_heads": self.shared_heads,
            "routed_heads": self.routed_heads,
            "mlp_dim": self.mlp_dim,
        })
        return config

    @classmethod 
    def from_config(cls, config):
        return cls(**config)
    
    
class KerasImplicitMotionAlignment(tf.keras.Model):
    def __init__(self, feature_dim, motion_dim, spatial_dim, depth=4, heads=8, shared_heads=2, routed_heads=6, mlp_dim=1024):
        super().__init__()
        self.feature_dim = feature_dim
        self.motion_dim = motion_dim
        self.spatial_dim = spatial_dim
        self.depth = depth
        self.heads = heads
        self.shared_heads = shared_heads
        self.routed_heads = routed_heads
        self.mlp_dim = mlp_dim

        self.cross_attention = KerasCrossAttentionModule(
            dim_spatial=spatial_dim[0] * spatial_dim[1], 
            dim_qk=motion_dim, 
            dim_v=feature_dim
        )
        
        self.transformer_blocks = [
            KerasTransformerBlock(
                dim=feature_dim,
                num_heads=heads,
                shared_heads=shared_heads,
                routed_heads=routed_heads,
                mlp_dim=mlp_dim
            ) for _ in range(depth)
        ]

    def call(self, inputs):
        # Unpack inputs from list
        if isinstance(inputs, (list, tuple)):
            ml_c, ml_r, fl_r = inputs
        else:
            ml_c, ml_r, fl_r = inputs[0], inputs[1], inputs[2]

        print(f"KerasImplicitMotionAlignment input shapes: ml_c: {ml_c.shape}, ml_r: {ml_r.shape}, fl_r: {fl_r.shape}")
        
        V_prime = self.cross_attention([ml_c, ml_r, fl_r])
        print(f"After cross-attention shape: {V_prime.shape}")
        
        for i, block in enumerate(self.transformer_blocks):
            V_prime = block(V_prime)
            print(f"After transformer block {i} shape: {V_prime.shape}")
            
        return V_prime

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "motion_dim": self.motion_dim,
            "spatial_dim": self.spatial_dim,
            "depth": self.depth,
            "heads": self.heads,
            "shared_heads": self.shared_heads,
            "routed_heads": self.routed_heads,
            "mlp_dim": self.mlp_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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