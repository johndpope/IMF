import torch
from model import IMFModel
import tensorflowjs as tfjs
import os
import onnx
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms
import tensorflow as tf
from tensorflow import keras
import torch.nn as nn
from scc4onnx import order_conversion
from onnx_tf.backend import prepare


class TFIMFModel(tf.Module):
        def __init__(self, pt_model):
            super(TFIMFModel, self).__init__()
            self.pt_model = pt_model

        @tf.function(input_signature=[
            tf.TensorSpec(shape=[1, 3, 256, 256], dtype=tf.float32),
            tf.TensorSpec(shape=[1, 3, 256, 256], dtype=tf.float32)
        ])
        def __call__(self, x_current, x_reference):
            x_current_pt = torch.from_numpy(x_current.numpy())
            x_reference_pt = torch.from_numpy(x_reference.numpy())
            outputs_pt = self.pt_model(x_current_pt, x_reference_pt)
            return tf.convert_to_tensor(outputs_pt.detach().numpy())
        
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



# def export_to_tf(model, output_path):
#     model.eval()
    
#     # Create example inputs
#     x_current = load_image("x_current.png")
#     x_reference = load_image("x_reference.png")

#     # Convert PyTorch model to TensorFlow
#     class TFIMFModel(tf.Module):
#         def __init__(self, pt_model):
#             super(TFIMFModel, self).__init__()
#             self.pt_model = pt_model

#         @tf.function(input_signature=[
#             tf.TensorSpec(shape=[1, 3, 256, 256], dtype=tf.float32),
#             tf.TensorSpec(shape=[1, 3, 256, 256], dtype=tf.float32)
#         ])
#         def __call__(self, x_current, x_reference):
#             x_current_pt = torch.from_numpy(x_current.numpy())
#             x_reference_pt = torch.from_numpy(x_reference.numpy())
#             outputs_pt = self.pt_model(x_current_pt, x_reference_pt)
#             return tf.convert_to_tensor(outputs_pt.detach().numpy())

#     # Create and save TensorFlow model
#     tf_model = TFIMFModel(model)
#     tf.saved_model.save(tf_model, output_path)
    
#     print(f"✅ Model exported successfully to {output_path}")

# def export_to_tfjs(model, output_path):
#     model.eval()
    
#     # Create example inputs
#     x_current = load_image("x_current.png")
#     x_reference = load_image("x_reference.png")

#     # Get PyTorch model output
#     with torch.no_grad():
#         pytorch_output = model(x_current, x_reference)

#     # Export the model to ONNX format
#     onnx_path = os.path.join(output_path, "model.onnx")
#     torch.onnx.dynamo_export(model,x_current, x_reference).save(onnx_path)
    
#     print(f"✅ ONNX model saved to {onnx_path}")

#     # Validate ONNX model
#     onnx_model = onnx.load(onnx_path)
#     onnx.checker.check_model(onnx_model)
#     print("✅ ONNX model checked successfully")

#     # Run inference with ONNX Runtime
#     # ort_session = onnxruntime.InferenceSession(onnx_path)
#     # ort_inputs = {
#     #     'l_x_current_': x_current.numpy(),
#     #     'l_x_reference_': x_reference.numpy()
#     # }
#     # ort_outputs = ort_session.run(None, ort_inputs)

#     # # Compare PyTorch and ONNX Runtime outputs
#     # for i, (pytorch_out, onnx_out) in enumerate(zip(pytorch_output, ort_outputs)):
#     #     np.testing.assert_allclose(pytorch_out.numpy(), onnx_out, rtol=1e-03, atol=1e-05)
#     #     print(f"Output {i} matched between PyTorch and ONNX Runtime")

#     print("✅ ONNX model validation successful")

#     # Convert ONNX model to TensorFlow.js format
#     tfjs.converters.convert_onnx_to_tfjs(onnx_path, output_path)
    
#     print(f"✅ Model exported successfully to {output_path}")




def export_pytorch_to_tfjs(pytorch_model, output_path):
    # Set the model to evaluation mode
    pytorch_model.eval()

    # Create dummy input tensors
    x_current = torch.randn(1, 3, 256, 256)
    x_reference = torch.randn(1, 3, 256, 256)

    # Export the model to ONNX format
    onnx_path = "model.onnx"
    torch.onnx.dynamo_export(pytorch_model, 
                      (x_current, x_reference),
                      onnx_path,
                      input_names=['x_current', 'x_reference'],
                      output_names=['output'])

    # Optimize the ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.optimizer.optimize(onnx_model)

    # Convert ONNX model to TensorFlow.js format

    tf_rep = prepare(onnx_model)
    tf_rep.export_graph("tf_model_dir")
# python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com 12



    # tfjs.converters.convert_onnx_to_tfjs(onnx_path, output_path)

    print(f"Model successfully exported to TensorFlow.js format at {output_path}")


if __name__ == "__main__":
    # Load your PyTorch model
    pytorch_model = IMFModel()
    pytorch_model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])


    export_pytorch_to_tfjs(pytorch_model,"tfjs_imf_encoder")
