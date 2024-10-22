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
from vit import CrossAttentionModule, ImplicitMotionAlignment, TransformerBlock
from vit_keras import KerasCrossAttentionModule, KerasImplicitMotionAlignment, KerasTransformerBlock
from rich.console import Console
from rich.traceback import install
import os




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


@converter(CrossAttentionModule, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
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
@converter(TransformerBlock, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_TransformerBlock(self, x):
    keras_block = KerasTransformerBlock(
        dim=self.attention.embed_dim,
        heads=self.attention.num_heads,
        dim_head=self.attention.head_dim,
        mlp_dim=self.mlp[0].out_features
    )
    
    # Build the Keras block by calling it with dummy input
    dummy_input = tf.zeros_like(x)
    keras_block(dummy_input)
    
    # Transfer weights for MultiheadAttention
    pytorch_mha = self.attention
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
    
    q_weight = q_weight.T.reshape(d_model, num_heads, head_dim)
    k_weight = k_weight.T.reshape(d_model, num_heads, head_dim)
    v_weight = v_weight.T.reshape(d_model, num_heads, head_dim)
    out_proj_weight = out_proj_weight.T
    
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

@converter(ImplicitMotionAlignment, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_ImplicitMotionAlignment(self, ml_c, ml_r, fl_r):
    feature_dim = self.feature_dim
    motion_dim = self.motion_dim
    spatial_dim = self.spatial_dim
    depth = len(self.transformer_blocks)
    heads = self.transformer_blocks[0].attention.num_heads
    dim_head = self.transformer_blocks[0].attention.head_dim
    mlp_dim = self.transformer_blocks[0].mlp[0].out_features

    print(f"ImplicitMotionAlignment parameters:")
    print(f"feature_dim: {feature_dim}, motion_dim: {motion_dim}, spatial_dim: {spatial_dim}")
    print(f"depth: {depth}, heads: {heads}, dim_head: {dim_head}, mlp_dim: {mlp_dim}")

    keras_model = KerasImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim, depth, heads, dim_head, mlp_dim)
    
    # Build the Keras model by calling it with dummy inputs
    dummy_ml_c = tf.zeros_like(ml_c)
    dummy_ml_r = tf.zeros_like(ml_r)
    dummy_fl_r = tf.zeros_like(fl_r)
    keras_model(dummy_ml_c, dummy_ml_r, dummy_fl_r)
    
    # Transfer weights for CrossAttentionModule
    keras_model.cross_attention.set_weights([
        self.cross_attention.q_pos_embedding.detach().numpy(),
        self.cross_attention.k_pos_embedding.detach().numpy()
    ])
    
    # Transfer weights for TransformerBlocks
    for i, (pytorch_block, keras_block) in enumerate(zip(self.transformer_blocks, keras_model.transformer_blocks)):
        print(f"Transferring weights for TransformerBlock {i}")

        # Transfer weights for MultiheadAttention
        pytorch_mha = pytorch_block.attention
        keras_mha = keras_block.attention.mha

        d_model = pytorch_mha.embed_dim
        num_heads = pytorch_mha.num_heads
        head_dim = d_model // num_heads

        in_proj_weight = pytorch_mha.in_proj_weight.detach().numpy()
        in_proj_bias = pytorch_mha.in_proj_bias.detach().numpy()
        out_proj_weight = pytorch_mha.out_proj.weight.detach().numpy()
        out_proj_bias = pytorch_mha.out_proj.bias.detach().numpy()

        # Split and reshape weights
        q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = np.split(in_proj_bias, 3)
        
        q_weight = q_weight.T.reshape(d_model, num_heads, head_dim)
        k_weight = k_weight.T.reshape(d_model, num_heads, head_dim)
        v_weight = v_weight.T.reshape(d_model, num_heads, head_dim)
        out_proj_weight = out_proj_weight.T

        keras_mha.set_weights([
            q_weight, k_weight, v_weight,
            out_proj_weight,
            q_bias, k_bias, v_bias,
            out_proj_bias
        ])

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
        keras_block.mlp.layers[0].set_weights([
            pytorch_block.mlp[0].weight.detach().numpy().T,
            pytorch_block.mlp[0].bias.detach().numpy()
        ])
        keras_block.mlp.layers[2].set_weights([
            pytorch_block.mlp[2].weight.detach().numpy().T,
            pytorch_block.mlp[2].bias.detach().numpy()
        ])

    return keras_model



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
    output_dir = "imf"
    keras_model.save(output_dir,save_format="tf")

    # Test the models
    with torch.no_grad():
        pytorch_output = pytorch_model(x_current, x_reference)

    keras_output = keras_model.predict([x_current.numpy(), x_reference.numpy()])

    # Compare outputs
    print("PyTorch output shape:", pytorch_output.shape)
    print("Keras output shape:", keras_output.shape)
    print("Output difference:", np.abs(pytorch_output.numpy() - keras_output).max())
    print("Relative difference:", np.abs((pytorch_output.numpy() - keras_output) / pytorch_output.numpy()).max())

    #  tensorflowjs_converter --input_format=tf_saved_model --quantize_float16="*"   --output_format=tfjs_graph_model --signature_name=serving_default --saved_model_tags=serve     imf graphmodel
    os.system(f'''
    tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    --quantize_float16="*" \
    {output_dir} \
    'graph_model'
    ''')