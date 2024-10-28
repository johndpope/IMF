import coremltools as ct
import torch
import numpy as np
# from model import IMFClientModel
import torch.nn as nn
from rich.console import Console
from rich.traceback import install
import os
from typing import Dict
from typing import List, Tuple
from model import LatentTokenDecoder,FrameDecoder,ImplicitMotionAlignment


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


def convert_to_coreml(pytorch_model, checkpoint_path, output_path="IMFClient.mlpackage"):
    """
    Convert PyTorch IMFClient model to Core ML format
    
    Args:
        pytorch_model: The PyTorch model instance
        checkpoint_path: Path to the PyTorch checkpoint
        output_path: Path where the Core ML model will be saved
    """
    # Load and prepare model
    pytorch_model = load_client_model(checkpoint_path)
    pytorch_model.eval()
    
    # Define input shapes
    t_c_shape = (1, 32)  # Latent token for current frame
    t_r_shape = (1, 32)  # Latent token for reference frame
    f_r_shapes = [  # Reference frame features at different scales
        (1, 128, 64, 64),
        (1, 256, 32, 32),
        (1, 512, 16, 16),
        (1, 512, 8, 8)
    ]
    
    # Create example inputs
    example_inputs = (
        torch.randn(*t_c_shape),
        torch.randn(*t_r_shape),
        [torch.randn(*shape) for shape in f_r_shapes]
    )
    
    # Trace the model
    traced_model = torch.jit.trace(pytorch_model, example_inputs)
    
    # Define input descriptions
    input_descriptions = {
        "t_c": "Current frame latent token",
        "t_r": "Reference frame latent token",
        "f_r_0": "Reference frame features at 64x64",
        "f_r_1": "Reference frame features at 32x32",
        "f_r_2": "Reference frame features at 16x16",
        "f_r_3": "Reference frame features at 8x8"
    }
    
    # Define output descriptions
    output_descriptions = {
        "output": "Reconstructed frame"
    }
    
    # Convert to Core ML
    model = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="t_c", shape=t_c_shape),
            ct.TensorType(name="t_r", shape=t_r_shape),
            ct.TensorType(name="f_r_0", shape=f_r_shapes[0]),
            ct.TensorType(name="f_r_1", shape=f_r_shapes[1]),
            ct.TensorType(name="f_r_2", shape=f_r_shapes[2]),
            ct.TensorType(name="f_r_3", shape=f_r_shapes[3])
        ],
        outputs=[
            ct.TensorType(name="output")
        ],
        minimum_deployment_target=ct.target.iOS16,  # Adjust based on your needs
        convert_to="mlprogram",  # Use ML Program for better performance
        compute_precision=ct.precision.FLOAT16,  # Use FP16 for smaller size
    )
    
    # Add metadata
    model.author = "Model Author"
    model.license = "Model License"
    model.short_description = "IMF Client Model for frame interpolation"
    model.version = "1.0"
    
    # Add input/output descriptions
    for k, v in input_descriptions.items():
        model.input_description[k] = v
    for k, v in output_descriptions.items():
        model.output_description[k] = v
    
    # Save the model
    model.save(output_path)
    
    return model

if __name__ == "__main__":
    # Initialize model
    pytorch_model = IMFClientModel(
        latent_dim=32,
        feature_dims=[128, 256, 512, 512],
        motion_dims=[256, 512, 512, 512],
        spatial_dims=[(64, 64), (32, 32), (16, 16), (8, 8)]
    )
    
    # Convert and save
    coreml_model = convert_to_coreml(
        pytorch_model,
        checkpoint_path="./checkpoints/checkpoint.pth",
        output_path="IMFClient.mlpackage"
    )
    
    # Test the converted model
    # Create dummy inputs
    t_c = np.random.randn(1, 32).astype(np.float32)
    t_r = np.random.randn(1, 32).astype(np.float32)
    f_r = [
        np.random.randn(1, 128, 64, 64).astype(np.float32),
        np.random.randn(1, 256, 32, 32).astype(np.float32),
        np.random.randn(1, 512, 16, 16).astype(np.float32),
        np.random.randn(1, 512, 8, 8).astype(np.float32)
    ]
    
    # PyTorch prediction
    with torch.no_grad():
        pytorch_output = pytorch_model(
            torch.from_numpy(t_c),
            torch.from_numpy(t_r),
            [torch.from_numpy(f) for f in f_r]
        ).numpy()
    
    # Core ML prediction
    coreml_inputs = {
        "t_c": t_c,
        "t_r": t_r,
        "f_r_0": f_r[0],
        "f_r_1": f_r[1],
        "f_r_2": f_r[2],
        "f_r_3": f_r[3]
    }
    coreml_output = coreml_model.predict(coreml_inputs)["output"]
    
    # Compare outputs
    print("PyTorch output shape:", pytorch_output.shape)
    print("Core ML output shape:", coreml_output.shape)
    print("Max absolute difference:", np.abs(pytorch_output - coreml_output).max())
    print("Max relative difference:", np.abs((pytorch_output - coreml_output) / (pytorch_output + 1e-7)).max())