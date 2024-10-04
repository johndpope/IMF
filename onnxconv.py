import torch
import torch.onnx
from IMF.model import IMFModel



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
            "imf_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['x_current', 'x_reference'],
            output_names=['output'],
            dynamic_axes={
                'x_current': {0: 'batch_size'},
                'x_reference': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            verbose=True
        )

                          
        print(f"Model exported successfully to {file_name}")
    except Exception as e:
        print(f"Error during ONNX export: {str(e)}")
        import traceback
        traceback.print_exc()

# Load your model
model = IMFModel()
model.eval()

# Load the checkpoint
checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location=lambda storage, loc: storage)
state_dict = checkpoint['model_state_dict']

# # Adjust the weights in the state_dict
# for key in state_dict.keys():
#     if 'csonv.weight' in key and state_dict[key].dim() == 5:
#         state_dict[key] = state_dict[key].squeeze(0)
# Create dummy input tensors
x_current = torch.randn(1, 3, 256, 256)
x_reference = torch.randn(1, 3, 256, 256)

# Export the model
export_to_onnx(model, x_current, x_reference, "imf_model.onnx")