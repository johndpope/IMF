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

from vit_moh import CrossAttentionModule, ImplicitMotionAlignment, TransformerBlock
from vit_keras_moh import KerasCrossAttentionModule, KerasImplicitMotionAlignment, KerasTransformerBlock
from rich.console import Console
from rich.traceback import install
import os
from typing import Dict
from typing import List, Tuple
from model import LatentTokenDecoder,FrameDecoder


console = Console(width=3000)

# Install Rich traceback handling
# install(show_locals=True)
install()

console.print("using keras MoH !!")


@converter(ImplicitMotionAlignment, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_ImplicitMotionAlignment(self, ml_c, ml_r, fl_r):
    feature_dim = self.feature_dim
    motion_dim = self.motion_dim
    spatial_dim = self.spatial_dim
    depth = len(self.transformer_blocks)
    
    # Get head configuration from first transformer block
    num_heads = self.transformer_blocks[0].attention.num_heads
    shared_heads = self.transformer_blocks[0].attention.shared_heads
    routed_heads = self.transformer_blocks[0].attention.routed_heads
    mlp_dim = self.transformer_blocks[0].mlp[0].out_features

    print(f"ImplicitMotionAlignment parameters:")
    print(f"feature_dim: {feature_dim}, motion_dim: {motion_dim}, spatial_dim: {spatial_dim}")
    print(f"depth: {depth}, heads: {num_heads}, shared_heads: {shared_heads}, routed_heads: {routed_heads}, mlp_dim: {mlp_dim}")

    keras_model = KerasImplicitMotionAlignment(
        feature_dim=feature_dim, 
        motion_dim=motion_dim, 
        spatial_dim=spatial_dim,
        depth=depth,
        heads=num_heads,
        shared_heads=shared_heads,
        routed_heads=routed_heads,
        mlp_dim=mlp_dim
    )
    
    # Build the Keras model by calling it with dummy inputs
    dummy_ml_c = tf.zeros_like(ml_c)
    dummy_ml_r = tf.zeros_like(ml_r)
    dummy_fl_r = tf.zeros_like(fl_r)
    keras_model([dummy_ml_c, dummy_ml_r, dummy_fl_r])
    
    # Transfer weights for CrossAttentionModule
    keras_model.cross_attention.set_weights([
        self.cross_attention.q_pos_embedding.detach().numpy(),
        self.cross_attention.k_pos_embedding.detach().numpy()
    ])
    
    # Transfer weights for each transformer block
    for i, (pytorch_block, keras_block) in enumerate(zip(self.transformer_blocks, keras_model.transformer_blocks)):
        # Transfer MoHAttention weights
        pytorch_mha = pytorch_block.attention
        keras_mha = keras_block.attention

        # Transfer linear layer weights
        keras_mha.q.set_weights([
            pytorch_mha.q.weight.detach().numpy().T,
            pytorch_mha.q.bias.detach().numpy()
        ])
        keras_mha.k.set_weights([
            pytorch_mha.k.weight.detach().numpy().T,
            pytorch_mha.k.bias.detach().numpy()
        ])
        keras_mha.v.set_weights([
            pytorch_mha.v.weight.detach().numpy().T,
            pytorch_mha.v.bias.detach().numpy()
        ])
        keras_mha.proj.set_weights([
            pytorch_mha.proj.weight.detach().numpy().T,
            pytorch_mha.proj.bias.detach().numpy()
        ])

        # Transfer router weights
        keras_mha.router.set_weights([
            pytorch_mha.router.weight.detach().numpy().T,
            pytorch_mha.router.bias.detach().numpy()
        ])
        keras_mha.shared_router.set_weights([
            pytorch_mha.shared_router.weight.detach().numpy().T,
            pytorch_mha.shared_router.bias.detach().numpy()
        ])

        # Transfer temperature and query embedding
        keras_mha.temperature.assign(pytorch_mha.temperature.detach().numpy())
        keras_mha.query_embedding.assign(pytorch_mha.query_embedding.detach().numpy())

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
        keras_block.mlp.layers[1].set_weights([
            pytorch_block.mlp[2].weight.detach().numpy().T,
            pytorch_block.mlp[2].bias.detach().numpy()
        ])

    return keras_model


@converter(TransformerBlock, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_PYTORCH_ORDER)
def converter_TransformerBlock(self, x):
    # Create Keras block with MoHAttention parameters
    keras_block = KerasTransformerBlock(
        dim=self.attention.dim,
        num_heads=self.attention.num_heads,
        shared_heads=self.attention.shared_heads,
        routed_heads=self.attention.routed_heads,
        mlp_dim=self.mlp[0].out_features
    )
    
    # Build the Keras block by calling it with dummy input
    dummy_input = tf.zeros_like(x)
    keras_block(dummy_input)
    
    # Transfer MoHAttention weights
    pytorch_mha = self.attention
    keras_mha = keras_block.attention

    # Transfer linear layer weights
    keras_mha.q.set_weights([
        pytorch_mha.q.weight.detach().numpy().T,
        pytorch_mha.q.bias.detach().numpy()
    ])
    keras_mha.k.set_weights([
        pytorch_mha.k.weight.detach().numpy().T,
        pytorch_mha.k.bias.detach().numpy()
    ])
    keras_mha.v.set_weights([
        pytorch_mha.v.weight.detach().numpy().T,
        pytorch_mha.v.bias.detach().numpy()
    ])
    keras_mha.proj.set_weights([
        pytorch_mha.proj.weight.detach().numpy().T,
        pytorch_mha.proj.bias.detach().numpy()
    ])
    
    # Transfer router weights
    keras_mha.router.set_weights([
        pytorch_mha.router.weight.detach().numpy().T,
        pytorch_mha.router.bias.detach().numpy()
    ])
    keras_mha.shared_router.set_weights([
        pytorch_mha.shared_router.weight.detach().numpy().T,
        pytorch_mha.shared_router.bias.detach().numpy()
    ])
    
    # Transfer learnable parameters
    keras_mha.temperature.assign(pytorch_mha.temperature.detach().numpy())
    keras_mha.query_embedding.assign(pytorch_mha.query_embedding.detach().numpy())

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




class IMFClientModel(nn.Module):
    def __init__(self, 
                 latent_dim: int = 32,
                 feature_dims: List[int] = [128, 256, 512, 512],
                 motion_dims: List[int] = [256, 512, 512, 512],
                 spatial_dims: List[Tuple[int, int]] = [(64, 64), (32, 32), (16, 16), (8, 8)]):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.spatial_dims = spatial_dims
        self.motion_dims = motion_dims
        
        # Initialize LatentTokenDecoder
        self.latent_token_decoder = LatentTokenDecoder(latent_dim=latent_dim)
        
        # Initialize ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(len(feature_dims)):
            feature_dim = feature_dims[i]
            motion_dim = motion_dims[i]
            spatial_dim = spatial_dims[i]
            alignment_module = ImplicitMotionAlignment(
                feature_dim=feature_dim, 
                motion_dim=motion_dim,
                spatial_dim=spatial_dim,
                depth=4,  # You can adjust this
                heads=8,  # You can adjust this
                mlp_dim=feature_dim * 4,  # Typically 4x the feature_dim
                shared_heads=2,  # You can adjust this
                routed_heads=6   # You can adjust this
            )
            self.implicit_motion_alignment.append(alignment_module)
        
        # Initialize FrameDecoder
        self.frame_decoder = FrameDecoder()

    def forward(self, t_c, t_r, f_r):
        """
        Args:
            t_c: Current frame latent token (B, latent_dim)
            t_r: Reference frame latent token (B, latent_dim)
            f_r: List of reference frame features [(B, C, H, W), ...]
        """
        # Generate motion features from tokens
        m_c = self.latent_token_decoder(t_c)
        m_r = self.latent_token_decoder(t_r)
        
        # Align features
        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            align_layer = self.implicit_motion_alignment[i]
            m_c_i = m_c[i]
            m_r_i = m_r[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)
        
        # Generate final image
        x_reconstructed = self.frame_decoder(aligned_features)
        return x_reconstructed

def filter_state_dict_for_client(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Filter the state dict to only include client-relevant keys"""
    client_keys = {
        'latent_token_decoder',
        'implicit_motion_alignment',
        'frame_decoder'
    }
    
    # Create new state dict with renamed keys
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Check if the key belongs to any of our client components
        if any(client_key in key for client_key in client_keys):
            new_key = key
            # If the key starts with 'model.', remove it
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
            new_state_dict[new_key] = value
            
    return new_state_dict

def load_client_model(checkpoint_path: str) -> IMFClientModel:
    """Load and initialize IMFClientModel from full checkpoint"""
    # Initialize client model
    client_model = IMFClientModel(
        latent_dim=32,
        feature_dims=[128, 256, 512, 512],
        motion_dims=[256, 512, 512, 512],
        spatial_dims=[(64, 64), (32, 32), (16, 16), (8, 8)]
    )
    
    # Load full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get the state dict (handle both cases where it might be wrapped)
    if 'model_state_dict' in checkpoint:
        full_state_dict = checkpoint['model_state_dict']
    else:
        full_state_dict = checkpoint
    
    # Filter state dict for client model
    client_state_dict = filter_state_dict_for_client(full_state_dict)
    
    # Load filtered state dict
    client_model.load_state_dict(client_state_dict, strict=False)
    client_model.eval()
    
    return client_model

if __name__ == "__main__":
    # Load client model instead of full model
    pytorch_model = load_client_model("./checkpoints/checkpoint.pth")
    pytorch_model.eval()

    # Create dummy inputs for the client model
    t_c = torch.randn(1, 32)  # Latent token for current frame
    t_r = torch.randn(1, 32)  # Latent token for reference frame
    f_r = [  # Reference frame features at different scales
        torch.randn(1, 128, 64, 64),
        torch.randn(1, 256, 32, 32),
        torch.randn(1, 512, 16, 16),
        torch.randn(1, 512, 8, 8)
    ]

    # Convert to Keras model
    keras_model = nobuco.pytorch_to_keras(
        pytorch_model,
        args=[t_c, t_r, f_r],
        kwargs=None,
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH
    )

    # Save Keras model
    output_dir = "imf_client"
    keras_model.save(output_dir, save_format="tf")

    # Test the models
    with torch.no_grad():
        pytorch_output = pytorch_model(t_c, t_r, f_r)

    # Convert inputs to numpy for Keras
    keras_inputs = [
        t_c.numpy(),
        t_r.numpy(),
        *[f.numpy() for f in f_r]
    ]
    keras_output = keras_model.predict(keras_inputs)

    # Compare outputs
    print("PyTorch output shape:", pytorch_output.shape)
    print("Keras output shape:", keras_output.shape)
    print("Output difference:", np.abs(pytorch_output.numpy() - keras_output).max())
    print("Relative difference:", np.abs((pytorch_output.numpy() - keras_output) / pytorch_output.numpy()).max())

    # Convert to TensorFlow.js model
    os.system(f'''
    tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_format=tfjs_graph_model \
    --signature_name=serving_default \
    --saved_model_tags=serve \
    --quantize_float16="*" \
    {output_dir} \
    'graph_model_client'
    ''')