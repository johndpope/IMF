import coremltools as ct
import torch
import numpy as np
from model import IMFClientModel


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