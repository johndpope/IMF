import torch
import torch.nn as nn
from model import IMFModel
from torchvision import transforms
from PIL import Image
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import json
import os
import io


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

def export_to_tfjs(model, x_current, x_reference, output_path):
    model.eval()
    
    # Create model structure
    model_json = {
        "format": "graph-model",
        "generatedBy": "pytorch-to-tfjs",
        "convertedBy": "pytorch-to-tfjs-converter",
        "modelTopology": {
            "node": [],
            "input": [],
            "output": []
        },
        "weightsManifest": []
    }

    # Add input nodes
    model_json["modelTopology"]["input"].append({
        "name": "x_current",
        "dtype": "DT_FLOAT",
        "tensorShape": {"dim": [{"size": str(d)} for d in x_current.shape]}
    })
    model_json["modelTopology"]["input"].append({
        "name": "x_reference",
        "dtype": "DT_FLOAT",
        "tensorShape": {"dim": [{"size": str(d)} for d in x_reference.shape]}
    })

    # Helper function to add a node to the model topology
    def add_node(name, op, input, output, attr=None):
        node = {
            "name": name,
            "op": op,
            "input": input,
            "output": output
        }
        if attr:
            node["attr"] = attr
        model_json["modelTopology"]["node"].append(node)

    # Export DenseFeatureEncoder
    add_node("dense_feature_encoder", "Custom", ["x_reference"], ["f_r"])

    # Export LatentTokenEncoder
    add_node("latent_token_encoder_current", "Custom", ["x_current"], ["t_c"])
    add_node("latent_token_encoder_reference", "Custom", ["x_reference"], ["t_r"])

    # Export LatentTokenDecoder
    add_node("latent_token_decoder_current", "Custom", ["t_c"], ["m_c"])
    add_node("latent_token_decoder_reference", "Custom", ["t_r"], ["m_r"])

    # Export ImplicitMotionAlignment modules
    for i in range(len(model.implicit_motion_alignment)):
        add_node(f"implicit_motion_alignment_{i}", "Custom", [f"m_c_{i}", f"m_r_{i}", f"f_r_{i}"], [f"aligned_feature_{i}"])

    # Export FrameDecoder
    aligned_features = [f"aligned_feature_{i}" for i in range(len(model.implicit_motion_alignment))]
    add_node("frame_decoder", "Custom", aligned_features, ["x_reconstructed"])

    # Add output node
    model_json["modelTopology"]["output"].append({
        "name": "x_reconstructed",
        "dtype": "DT_FLOAT",
        "tensorShape": {"dim": [{"size": str(d)} for d in x_current.shape]}
    })

    # Save model.json
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'model.json'), 'w') as f:
        json.dump(model_json, f)

    # Export weights
    weight_data = io.BytesIO()
    torch.save(model.state_dict(), weight_data)
    weight_data.seek(0)
    
    weights_manifest = [{
        "paths": ["group1-shard1of1.bin"],
        "weights": [{
            "name": name,
            "shape": list(param.shape),
            "dtype": "float32"
        } for name, param in model.named_parameters()]
    }]
    
    with open(os.path.join(output_path, 'weights_manifest.json'), 'w') as f:
        json.dump(weights_manifest, f)
    
    with open(os.path.join(output_path, 'group1-shard1of1.bin'), 'wb') as f:
        f.write(weight_data.getvalue())

    print(f"Model exported successfully to {output_path}")

if __name__ == "__main__":
    # Load your PyTorch model
    model = IMFModel()
    model.eval()

    # Load the checkpoint
    checkpoint = torch.load("./checkpoints/checkpoint.pth", map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load example images
    x_current = load_image("x_current.png")
    x_reference = load_image("x_reference.png")

    # Export the model to TensorFlow.js format
    export_to_tfjs(model, x_current, x_reference, "tfjs_imf_encoder")