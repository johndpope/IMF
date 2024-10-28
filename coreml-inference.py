import numpy as np
import coremltools as ct
from rich.console import Console
from pathlib import Path

console = Console()

def run_coreml_inference(model_path: str, 
                        t_c: np.ndarray,
                        t_r: np.ndarray,
                        f_r_0: np.ndarray,
                        f_r_1: np.ndarray,
                        f_r_2: np.ndarray,
                        f_r_3: np.ndarray) -> np.ndarray:
    """
    Run inference using Core ML model
    
    Args:
        model_path: Path to Core ML model (.mlpackage)
        t_c: Current frame latent token (1, 32)
        t_r: Reference frame latent token (1, 32)
        f_r_0: Reference frame features at 64x64 (1, 128, 64, 64)
        f_r_1: Reference frame features at 32x32 (1, 256, 32, 32)
        f_r_2: Reference frame features at 16x16 (1, 512, 16, 16)
        f_r_3: Reference frame features at 8x8 (1, 512, 8, 8)
    Returns:
        Predicted output as numpy array
    """
    # Load model
    model = ct.models.MLModel(model_path)
    
    # Prepare inputs
    inputs = {
        't_c': t_c.astype(np.float32),
        't_r': t_r.astype(np.float32),
        'f_r_0': f_r_0.astype(np.float32),
        'f_r_1': f_r_1.astype(np.float32),
        'f_r_2': f_r_2.astype(np.float32),
        'f_r_3': f_r_3.astype(np.float32)
    }
    
    # Run inference
    output = model.predict(inputs)
    return output['output']

def create_sample_input(batch_size=1):
    """Create sample input data"""
    return {
        't_c': np.random.randn(batch_size, 32).astype(np.float32),
        't_r': np.random.randn(batch_size, 32).astype(np.float32),
        'f_r_0': np.random.randn(batch_size, 128, 64, 64).astype(np.float32),
        'f_r_1': np.random.randn(batch_size, 256, 32, 32).astype(np.float32),
        'f_r_2': np.random.randn(batch_size, 512, 16, 16).astype(np.float32),
        'f_r_3': np.random.randn(batch_size, 512, 8, 8).astype(np.float32)
    }

if __name__ == "__main__":
    # Model path
    model_path = "IMFClient.mlpackage"
    
    if not Path(model_path).exists():
        console.print(f"[red]Error: Model not found at {model_path}[/red]")
        exit(1)
    
    # Create sample input
    inputs = create_sample_input()
    
    # Run inference
    console.print("Running Core ML inference...")
    try:
        output = run_coreml_inference(
            model_path=model_path,
            t_c=inputs['t_c'],
            t_r=inputs['t_r'],
            f_r_0=inputs['f_r_0'],
            f_r_1=inputs['f_r_1'],
            f_r_2=inputs['f_r_2'],
            f_r_3=inputs['f_r_3']
        )
        console.print(f"Success! Output shape: {output.shape}")
        
    except Exception as e:
        console.print(f"[red]Inference failed: {str(e)}[/red]")
        if "macOS version 10.13" in str(e):
            console.print("[yellow]Note: Core ML inference only works on macOS 10.13 or later[/yellow]")
