import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import torch
from model import IMFModel
from nobuco import converter, ChannelOrderingStrategy
import tensorflow as tf
import torch.nn as nn

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional
from torch import Tensor
import torch.nn as nn
from nobuco.commons import ChannelOrderingStrategy
from nobuco.converters.node_converter import converter
from vit import TransformerBlock

class KerasTransformerBlock(keras.layers.Layer):
    def __init__(self, dim, heads, dim_head, mlp_dim):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim

        self.attention = keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim_head)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([
            keras.layers.Dense(mlp_dim),
            keras.layers.Activation('gelu'),
            keras.layers.Dense(dim)
        ])

    def call(self, x):
        # Adjusted to match Keras's expectations
        B, C, H, W = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x_reshaped = tf.reshape(x, (B, H * W, C))  # Shape: (B, seq_len, C)

        x_norm = self.norm1(x_reshaped)
        att_output = self.attention(x_norm, x_norm)
        x_reshaped = x_reshaped + att_output

        ff_output = self.mlp(self.norm2(x_reshaped))
        x_reshaped = x_reshaped + ff_output

        output = tf.reshape(x_reshaped, (B, H, W, C))
        output = tf.transpose(output, perm=[0, 3, 1, 2])  # Back to (B, C, H, W)
        return output

@converter(TransformerBlock, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_TransformerBlock(self, x):
    keras_block = KerasTransformerBlock(
        dim=self.attention.embed_dim,
        heads=self.attention.num_heads,
        dim_head=self.attention.head_dim,
        mlp_dim=self.mlp[0].out_features
    )
    
    # Build the Keras block by calling it with dummy input
    input_shape = x.shape
    dummy_input = tf.zeros(input_shape)
    keras_block(dummy_input)
    
    # Transfer weights
    transfer_transformer_block_weights(self, keras_block)
    
    def func(x):
        return keras_block(x)
    return func


def transfer_transformer_block_weights(pytorch_block, keras_block):
    # Transfer LayerNorm weights
    keras_block.norm1.set_weights([
        pytorch_block.norm1.weight.detach().numpy(),
        pytorch_block.norm1.bias.detach().numpy()
    ])
    keras_block.norm2.set_weights([
        pytorch_block.norm2.weight.detach().numpy(),
        pytorch_block.norm2.bias.detach().numpy()
    ])

    # Transfer MLP weights
    transfer_linear_weights(pytorch_block.mlp[0], keras_block.mlp.layers[0])
    transfer_linear_weights(pytorch_block.mlp[2], keras_block.mlp.layers[2])

    # Transfer MultiHeadAttention weights
    transfer_multihead_attention_weights(pytorch_block.attention, keras_block.attention)

def transfer_linear_weights(pytorch_linear, keras_dense):
    weight = pytorch_linear.weight.detach().numpy()
    bias = pytorch_linear.bias.detach().numpy()
    # Transpose weight matrix
    keras_dense.set_weights([weight.T, bias])

def transfer_multihead_attention_weights(pytorch_mha, keras_mha):
    # Extract PyTorch weights
    in_proj_weight = pytorch_mha.in_proj_weight.detach().numpy()  # Shape: (3*embed_dim, embed_dim)
    in_proj_bias = pytorch_mha.in_proj_bias.detach().numpy()  # Shape: (3*embed_dim,)
    out_proj_weight = pytorch_mha.out_proj.weight.detach().numpy()  # Shape: (embed_dim, embed_dim)
    out_proj_bias = pytorch_mha.out_proj.bias.detach().numpy()  # Shape: (embed_dim,)

    embed_dim = pytorch_mha.embed_dim  # Total embedding dimension

    # Split weights
    q_weight = in_proj_weight[:embed_dim, :]  # Shape: (embed_dim, embed_dim)
    k_weight = in_proj_weight[embed_dim:2*embed_dim, :]
    v_weight = in_proj_weight[2*embed_dim:, :]
    q_bias = in_proj_bias[:embed_dim]
    k_bias = in_proj_bias[embed_dim:2*embed_dim]
    v_bias = in_proj_bias[2*embed_dim:]

    # Transpose weights to match Keras's expected shape
    q_weight = q_weight.T  # Shape: (embed_dim, embed_dim)
    k_weight = k_weight.T
    v_weight = v_weight.T
    out_proj_weight = out_proj_weight.T  # Shape: (embed_dim, embed_dim)

    # Set weights in the correct order
    keras_mha.set_weights([
        q_weight,          # query_dense.kernel
        q_bias,            # query_dense.bias
        k_weight,          # key_dense.kernel
        k_bias,            # key_dense.bias
        v_weight,          # value_dense.kernel
        v_bias,            # value_dense.bias
        out_proj_weight,   # output_dense.kernel
        out_proj_bias      # output_dense.bias
    ])



class MultiheadAttentionWrapper(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    
    def build(self, input_shape):
        # Build the MultiHeadAttention layer
        self.mha.build(input_shape)
        super().build(input_shape)
    
    def call(self, query, value, key=None, attention_mask=None, return_attention_scores=False):
        # Transpose inputs
        query = tf.transpose(query, [1, 0, 2])
        value = tf.transpose(value, [1, 0, 2])
        if key is not None:
            key = tf.transpose(key, [1, 0, 2])
        
        # Call MultiHeadAttention
        output = self.mha(query, value, key, attention_mask=attention_mask, return_attention_scores=return_attention_scores)
        
        # Transpose output
        if return_attention_scores:
            output, attention_scores = output
            output = tf.transpose(output, [1, 0, 2])
            
            # Average attention scores across heads
            attention_scores = tf.reduce_mean(attention_scores, axis=1)
            
            return output, attention_scores
        else:
            output = tf.transpose(output, [1, 0, 2])
            return output
        
@nobuco.converter(nn.MultiheadAttention, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_MultiheadAttention(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
    wrapper = MultiheadAttentionWrapper(num_heads=self.num_heads, key_dim=self.head_dim, embed_dim=self.embed_dim)
    
    # Build the layer
    dummy_input = tf.keras.Input(shape=(None, self.embed_dim))
    wrapper.build(dummy_input.shape)
    
    # Prepare weights
    in_proj_weight = self.in_proj_weight.detach().numpy().T
    in_proj_bias = self.in_proj_bias.detach().numpy() if self.in_proj_bias is not None else None
    out_proj_weight = self.out_proj.weight.detach().numpy().T
    out_proj_bias = self.out_proj.bias.detach().numpy() if self.out_proj.bias is not None else None
    
    # Split the in_proj_weight and in_proj_bias into Q, K, V parts
    q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=1)
    q_bias, k_bias, v_bias = np.split(in_proj_bias, 3) if in_proj_bias is not None else (None, None, None)
    
    # Set the weights for the MultiHeadAttention layer
    wrapper.mha.set_weights([
        q_weight, k_weight, v_weight,
        out_proj_weight,
        q_bias, k_bias, v_bias,
        out_proj_bias
    ])
    
    def func(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return wrapper(query, value, key, attention_mask=attn_mask, return_attention_scores=need_weights)
    
    return func


@converter(torch.flip, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_flip(input: torch.Tensor, dims):
    def func(input, dims):
        # Ensure dims is a tensor
        dims = tf.convert_to_tensor(dims, dtype=tf.int32)
        
        # Convert PyTorch-style negative indices to TensorFlow-style
        rank = tf.rank(input)
        dims = tf.where(dims < 0, dims + rank, dims)
        
        # TensorFlow's reverse operation expects a list of axes to reverse
        # We don't need to create a boolean mask anymore
        return tf.reverse(input, dims)
    return func
if __name__ == "__main__":
    # Load your PyTorch model
    pytorch_model = IMFModel()
    pytorch_model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])

    x_current = torch.randn(1, 3, 256, 256)
    x_reference = torch.randn(1, 3, 256, 256)
    keras_model = nobuco.pytorch_to_keras(
        pytorch_model,
        args=[x_current,x_reference], kwargs=None,
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH
    )

    # tfjs.converters.save_keras_model(keras_model, 'keras')
    keras_model.save("imf",save_format="tf")

    # Test the models
    with torch.no_grad():
        pytorch_output = pytorch_model(x_current, x_reference)

    keras_output = keras_model.predict([x_current.numpy(), x_reference.numpy()])

    # Compare outputs
    print("PyTorch output shape:", pytorch_output.shape)
    print("Keras output shape:", keras_output.shape)
    print("Output difference:", np.abs(pytorch_output.numpy() - keras_output).max())
    print("Relative difference:", np.abs((pytorch_output.numpy() - keras_output) / pytorch_output.numpy()).max())