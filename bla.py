import torch
import torchvision
import ai_edge_torch

# Use resnet18 with pre-trained weights.
resnet18 = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
sample_inputs = (torch.randn(1, 3, 224, 224),)

# Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
# are setting the model to evaluation mode prior to conversion.
edge_model = ai_edge_torch.convert(resnet18.eval(), sample_inputs)
edge_model.export("resnet18.tflite")