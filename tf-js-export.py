import torch
from model import IMFModel
import tensorflowjs as tfjs
import os
import onnx
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

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

def export_to_tfjs(model, output_path):
    model.eval()
    
    # Create example inputs
    x_current = load_image("x_current.png")
    x_reference = load_image("x_reference.png")

    # Get PyTorch model output
    with torch.no_grad():
        pytorch_output = model(x_current, x_reference)

    # Export the model to ONNX format
    onnx_path = os.path.join(output_path, "model.onnx")
    torch.onnx.dynamo_export(model,x_current, x_reference).save(onnx_path)
    
    print(f"✅ ONNX model saved to {onnx_path}")

    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model checked successfully")

    # Run inference with ONNX Runtime
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {
        'l_x_current_': x_current.numpy(),
        'l_x_reference_': x_reference.numpy()
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    # Compare PyTorch and ONNX Runtime outputs
    for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_output, ort_outputs)):
        np.testing.assert_allclose(pytorch_out.numpy(), onnx_out, rtol=1e-03, atol=1e-05)
        print(f"Output {i} matched between PyTorch and ONNX Runtime")

    print("✅ ONNX model validation successful")

    # Convert ONNX model to TensorFlow.js format
    tfjs.converters.convert_onnx_to_tfjs(onnx_path, output_path)
    
    print(f"✅ Model exported successfully to {output_path}")

if __name__ == "__main__":
    # Load your PyTorch model
    model = IMFModel()
    model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export the model to TensorFlow.js format
    export_to_tfjs(model, "tfjs_imf_encoder")