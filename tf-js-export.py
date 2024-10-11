import torch
from model import IMFModel
from torchvision import transforms
from PIL import Image
import tensorflowjs as tfjs
import os

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
    
    # Create a traced model
    x_current = torch.randn(1, 3, 256, 256)
    x_reference = torch.randn(1, 3, 256, 256)
    traced_model = torch.jit.trace(model, (x_current, x_reference))
    
    # Export the model to ONNX format
    onnx_path = os.path.join(output_path, "model.onnx")
    torch.onnx.export(traced_model, (x_current, x_reference), onnx_path,
                      input_names=['x_current', 'x_reference'],
                      output_names=['output'])
    
    # Convert ONNX model to TensorFlow.js format
    tfjs.converters.convert_onnx_to_tfjs(onnx_path, output_path)
    
    print(f"Model exported successfully to {output_path}")

if __name__ == "__main__":
    # Load your PyTorch model
    model = IMFModel()
    model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Export the model to TensorFlow.js format
    export_to_tfjs(model, "tfjs_imf_encoder")