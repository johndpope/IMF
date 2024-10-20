import torch
import ai_edge_torch
import tensorflow as tf
from model import IMFModel
import os 
os.environ['PJRT_DEVICE'] = 'CPU'

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.traceback import Traceback

import inspect
console = Console()

# Load and prepare the PyTorch model
pytorch_model = IMFModel()
pytorch_model.eval()
checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
pytorch_model.load_state_dict(checkpoint['model_state_dict'])
pytorch_model.eval()

# Prepare sample inputs
x_current = torch.randn(1, 3, 256, 256)
x_reference = torch.randn(1, 3, 256, 256)
sample_inputs = (x_current, x_reference)

# Convert PyTorch model to TensorFlow model with additional flags
tfl_converter_flags = {
    'experimental_enable_resource_variables': True,
    # Add any other necessary flags here
}

try:
    # ai_edge_torch.experimental_enable_resource_variables = True
    tf_model = ai_edge_torch.convert(pytorch_model, sample_inputs, _ai_edge_converter_flags=tfl_converter_flags)
    tf_model.export("imf.tflite")
    console.print("TFLite model saved as 'imf.tflite'")
except Exception as e:
    console.print(f"Error during conversion: {e}")
    # If AI Edge Torch conversion fails, you might need to use a different approach
    # such as ONNX or reimplementing the model in TensorFlow