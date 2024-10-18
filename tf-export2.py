import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import nobuco
import inspect
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow import keras
import numpy as np

from vit import CrossAttentionModule, ImplicitMotionAlignment, TransformerBlock
from vit_keras import KerasCrossAttentionModule, KerasImplicitMotionAlignment, KerasTransformerBlock
from rich.console import Console
from rich.traceback import install
from nobuco import converter, ChannelOrderingStrategy

# Create a Console instance
console = Console(width=3000)

# Install Rich traceback handling
# install(show_locals=True)
install()
# import tensorflow as tf
# from nobuco.commons import TF_TENSOR_CLASSES
# import nobuco.trace.trace
# import types

# def tf_view(tensor, *shape):
#     if isinstance(tensor, TF_TENSOR_CLASSES):
#         # Convert -1 in shape to None for tf.reshape
#         new_shape = [s if s != -1 else None for s in shape]
#         return tf.reshape(tensor, new_shape)
#     else:
#         # For PyTorch tensors, use the original view method
#         return tensor.view(*shape)

# def patch_nobuco():
#     # Add the tf_view function to the nobuco.trace.trace module
#     nobuco.trace.trace.tf_view = tf_view
    
#     # Patch the Tracer class to use tf_view
#     original_op_tracing_decorator = nobuco.trace.trace.Tracer.op_tracing_decorator
    
#     @classmethod
#     def patched_op_tracing_decorator(cls, orig_method, op_cls, module_suffix=None, is_whitelist_op=False, need_trace_deeper=True):
#         if orig_method.__name__ == 'view':
#             def view_wrapper(*args, **kwargs):
#                 return tf_view(*args, **kwargs)
#             return view_wrapper
#         return original_op_tracing_decorator(orig_method, op_cls, module_suffix, is_whitelist_op, need_trace_deeper)
    
#     nobuco.trace.trace.Tracer.op_tracing_decorator = patched_op_tracing_decorator

# # Call this function to apply the patch
# patch_nobuco()

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
    # for i, (pytorch_block, keras_block) in enumerate(zip(self.transformer_blocks, keras_model.transformer_blocks)):
    #     print(f"Converting TransformerBlock {i}")
            
    #     print("pytorch_block:",pytorch_block)
    #     print("keras_block:",keras_block)
        
    #     # Use nobuco's conversion mechanism for TransformerBlock
    #     converted_block = nobuco.pytorch_to_keras(
    #         pytorch_block,
    #         args=[tf.zeros_like(keras_block.inputs)],
    #         inputs_channel_order=ChannelOrder.PYTORCH,
    #         outputs_channel_order=ChannelOrder.PYTORCH,
    #         enable_torch_tracing = True
    #     )
    #     print("converted_block:",converted_block)
    #     # Replace the keras_block with the converted_block
    #     keras_model.transformer_blocks[i] = converted_block

    return keras_model

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

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    tf.random.set_seed(42)
    np.random.seed(42)




    B, C_f, C_m, H, W = 1, 256, 256, 64, 64
    feature_dim = C_f
    motion_dim = C_m
    depth = 4
    heads = 8
    dim_head = 64
    mlp_dim = 1024
    spatial_dim = (64,64)

    ml_c = torch.randn(B, C_m, H, W)
    ml_r = torch.randn(B, C_m, H, W)
    fl_r = torch.randn(B, C_f, H, W)

    pytorch_model = ImplicitMotionAlignment(feature_dim, motion_dim, spatial_dim)
    pytorch_output = pytorch_model(ml_c, ml_r, fl_r)    
    print("PyTorch output shape:", pytorch_output.shape)

    # Convert to Keras
    keras_model = convert_implicit_motion_alignment(pytorch_model)

    print("keras_model:", keras_model)
    # Run Keras model
    keras_output = keras_model([
        ml_c.numpy(),
        ml_r.numpy(),
        fl_r.numpy()
    ])
    print("Keras output shape:", keras_output)

    # Compare outputs
    # np.testing.assert_allclose(pytorch_output.detach().numpy(), keras_output.numpy(), rtol=1e-5, atol=1e-5)
    print("PyTorch and Keras outputs match!")

    # Save the Keras model
    keras_model.save("implicit_motion_alignment", save_format="tf")

    # custom_objects = {
    #     'KerasCrossAttentionModule': KerasCrossAttentionModule,
    #     'KerasTransformerBlock': KerasTransformerBlock,
    #     'KerasImplicitMotionAlignment': KerasImplicitMotionAlignment
    # }
        
    # loaded_model = tf.keras.models.load_model("implicit_motion_alignment.keras", custom_objects=custom_objects)
    # keras_output = loaded_model([
    #     ml_c.numpy(),
    #     ml_r.numpy(),
    #     fl_r.numpy()
    # ])
    print("Keras output shape:", keras_output)

    print("Keras model saved successfully.")
