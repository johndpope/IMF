import torch
import torch.onnx
from model import IMFModel
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from onnxconverter_common import float16
import onnx
import numpy as np

class IMFDecoder(nn.Module):
    def __init__(self, model):
        super(IMFDecoder, self).__init__()
        self.model = model

    def decode_latent_tokens(self,f_r,t_r,t_c):
        return  self.model.decode_latent_tokens(f_r,t_r,t_c)



# Define the IMFEncoder class
class IMFEncoder(nn.Module):
    def __init__(self, model):
        super(IMFEncoder, self).__init__()
        self.model = model

    def forward(self, x_current, x_reference):
        f_r = self.model.dense_feature_encoder(x_reference)
        t_r = self.model.latent_token_encoder(x_reference)
        t_c = self.model.latent_token_encoder(x_current)
        return f_r, t_r, t_c  # Fixed indentation here

# Define the trace handler and utility functions
def trace_handler(module, input, output):
    print(f"\nModule: {module.__class__.__name__}")
    for idx, inp in enumerate(input):
        print_tensor_info(inp, f"  Input[{idx}]")
    if isinstance(output, torch.Tensor):
        print_tensor_info(output, "  Output")
    else:
        for idx, out in enumerate(output):
            print_tensor_info(out, f"  Output[{idx}]")

def print_tensor_info(tensor, name, indent=0):
    indent_str = ' ' * indent
    print(f"{indent_str}{name}:")
    if isinstance(tensor, torch.Tensor):
        print(f"{indent_str}  Shape: {tensor.shape}")
        print(f"{indent_str}  Dtype: {tensor.dtype}")
        print(f"{indent_str}  Device: {tensor.device}")
        print(f"{indent_str}  Requires grad: {tensor.requires_grad}")
    elif isinstance(tensor, (list, tuple)):
        print(f"{indent_str}  Type: {type(tensor).__name__}, Length: {len(tensor)}")
        for idx, item in enumerate(tensor):
            print_tensor_info(item, f"{name}[{idx}]", indent=indent + 2)
    else:
        print(f"{indent_str}  Type: {type(tensor).__name__}")

def print_model_structure(model):
    print("Model Structure:")
    for name, module in model.named_modules():
        print(f"{name}: {module.__class__.__name__}")
        if hasattr(module, 'weight'):
            print(f"  Weight shape: {module.weight.shape}")
        if hasattr(module, 'bias') and module.bias is not None:
            print(f"  Bias shape: {module.bias.shape}")


import onnx
from onnx import numpy_helper
import numpy as np

def convert_float64_to_float32(tensor):
    return onnx.helper.make_tensor(
        name=tensor.name,
        data_type=onnx.TensorProto.FLOAT,
        dims=tensor.dims,
        vals=numpy_helper.to_array(tensor).astype(np.float32).tobytes(),
        raw=True,
    )

def convert_int64_to_int32(tensor):
    return onnx.helper.make_tensor(
        name=tensor.name,
        data_type=onnx.TensorProto.INT32,
        dims=tensor.dims,
        vals=numpy_helper.to_array(tensor).astype(np.int32).tobytes(),
        raw=True,
    )

def convert_model_to_32bit(model, output_path):    
    # Convert initializers
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.DOUBLE:
            new_initializer = convert_float64_to_float32(initializer)
            model.graph.initializer.remove(initializer)
            model.graph.initializer.extend([new_initializer])
        elif initializer.data_type == onnx.TensorProto.INT64:
            new_initializer = convert_int64_to_int32(initializer)
            model.graph.initializer.remove(initializer)
            model.graph.initializer.extend([new_initializer])
    
    # Convert inputs
    for input in model.graph.input:
        if input.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            input.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        elif input.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            input.type.tensor_type.elem_type = onnx.TensorProto.INT32
    
    # Convert outputs
    for output in model.graph.output:
        if output.type.tensor_type.elem_type == onnx.TensorProto.DOUBLE:
            output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        elif output.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            output.type.tensor_type.elem_type = onnx.TensorProto.INT32
    
    # Save the converted model
    onnx.save(model, output_path)
    print(f"Converted model saved to {output_path}")


def export_to_onnx(model, x_current, x_reference, file_name):
    try:
        print("Model structure before tracing:")
        print_model_structure(model)

        print("\nInput tensor information:")
        print_tensor_info(x_current, "x_current")
        print_tensor_info(x_reference, "x_reference")

        hooks = []
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(trace_handler))

        # Use torch.jit.trace to create a traced version of the model
        print("\nTracing model...")
        traced_model = torch.jit.trace(model, (x_current, x_reference))
        print("Model traced successfully")

        for hook in hooks:
            hook.remove()

        print("\nModel structure after tracing:")
        print_model_structure(traced_model)

        print("\nExporting to ONNX...")
        torch.onnx.export(
            model,
            (x_current, x_reference),
            file_name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['x_current', 'x_reference'],
            output_names=['f_r', 't_r', 't_c'],
            dynamic_axes={
                'x_current': {0: 'batch_size'},
                'x_reference': {0: 'batch_size'},
                'f_r': {0: 'batch_size'},
                't_r': {0: 'batch_size'},
                't_c': {0: 'batch_size'}
            },
            verbose=True
        )
        print(f"Model exported successfully to {file_name}")

        # Load the ONNX model
        onnx_model = onnx.load(file_name)

    # Convert int64 to int32
        print("Converting int64 to int32...")
        web_compatible_file = file_name.replace('.onnx', '_web.onnx')
        onnx_model = convert_model_to_32bit(onnx_model,web_compatible_file)



    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        import traceback
        traceback.print_exc()

# Load your model
model = IMFModel()
model.eval()

# Load the checkpoint
checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# Create the IMFEncoder instance
encoder_model = IMFEncoder(model)
encoder_model.eval()

# Load images and preprocess
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

x_current = load_image("x_current.png")
x_reference = load_image("x_reference.png")

# Export the model
export_to_onnx(encoder_model, x_current, x_reference, "imf_encoder.onnx")