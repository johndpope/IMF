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


def export_to_tf(model, output_path):
    model.eval()
    
    # Create example inputs
    x_current = load_image("x_current.png")
    x_reference = load_image("x_reference.png")

    # Convert PyTorch model to TensorFlow
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

    # Create and save TensorFlow model
    tf_model = TFIMFModel(model)
    tf.saved_model.save(tf_model, output_path)
    
    print(f"✅ Model exported successfully to {output_path}")

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


def create_keras_model(pytorch_model):
    # This function needs to be implemented based on your IMFModel architecture
    # Here's a placeholder implementation that you need to adjust
    inputs_current = keras.Input(shape=(3, 256, 256))
    inputs_reference = keras.Input(shape=(3, 256, 256))
    
    x = keras.layers.Concatenate(axis=1)([inputs_current, inputs_reference])
    
    # Add layers that match your PyTorch model architecture
    # For example:
    x = keras.layers.Conv2D(64, 3, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    # ... add more layers as needed ...
    
    outputs = keras.layers.Conv2D(3, 1, padding='same')(x)
    
    return keras.Model(inputs=[inputs_current, inputs_reference], outputs=outputs)

def transfer_weights(pytorch_model, keras_model):
    pytorch_layers = list(pytorch_model.modules())
    keras_layers = keras_model.layers

    for pytorch_layer, keras_layer in zip(pytorch_layers, keras_layers):
        if isinstance(keras_layer, keras.layers.Conv2D):
            if hasattr(pytorch_layer, 'weight'):
                weights = [
                    pytorch_layer.weight.detach().numpy().transpose(2, 3, 1, 0),
                    pytorch_layer.bias.detach().numpy() if pytorch_layer.bias is not None else np.zeros(keras_layer.filters)
                ]
                keras_layer.set_weights(weights)
        elif isinstance(keras_layer, keras.layers.BatchNormalization):
            if hasattr(pytorch_layer, 'weight'):
                weights = [
                    pytorch_layer.weight.detach().numpy(),
                    pytorch_layer.bias.detach().numpy(),
                    pytorch_layer.running_mean.detach().numpy(),
                    pytorch_layer.running_var.detach().numpy(),
                ]
                keras_layer.set_weights(weights)
        # Add more conditions for other layer types as needed

    print("Weights transferred from PyTorch to Keras model")


def export_to_keras(pytorch_model, output_path):
    # pytorch_model.eval()
    
    # Create Keras model with matching architecture
    keras_model = create_keras_model(pytorch_model)
    
    # Transfer weights from PyTorch to Keras model
    transfer_weights(pytorch_model, keras_model)
    
    # Test the Keras model
    x_current = np.random.rand(1, 3, 256, 256).astype(np.float32)
    x_reference = np.random.rand(1, 3, 256, 256).astype(np.float32)
    keras_output = keras_model([x_current, x_reference])
    print(f"Keras model output shape: {keras_output.shape}")
    print(f"output_path: {output_path}")
    # Save the Keras model
    keras_model.save(output_path)
    print(f"✅ Model exported successfully to Keras format at {output_path}")
    
    return keras_model

if __name__ == "__main__":
    # Load your PyTorch model
    pytorch_model = IMFModel()
    pytorch_model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model_state_dict'])

    # Export the model to Keras format
    keras_model = export_to_keras(pytorch_model, "keras_imf_encoder.h5")


    # Export the model to TensorFlow.js format
    # export_to_tfjs(pytorch_model, "tfjs_imf_encoder") - BROKEN