import torch
import ai_edge_torch
import tensorflow as tf
from model import IMFModel
import os 
os.environ['PJRT_DEVICE'] = 'CPU'

# Assuming you have your PyTorch model defined as 'pytorch_model'
pytorch_model = IMFModel()
pytorch_model.eval()
# Load the checkpoint
checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
pytorch_model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
pytorch_model.eval()

x_current = torch.randn(1, 3, 256, 256)
x_reference = torch.randn(1, 3, 256, 256)

sample_inputs = (x_current,x_reference)
# Convert PyTorch model to TensorFlow model
tf_model = ai_edge_torch.convert(pytorch_model,sample_inputs)
tf_model.export("imf.tflite")
# Convert TensorFlow model to TFLite
# converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
# tflite_model = converter.convert()

# Save the TFLite model
# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

print("TFLite model saved as 'model.tflite'")