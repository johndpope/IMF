import torch
import torch.nn as nn
from sam2.modeling.sam2_base import SAM2Base
from sam2.build_sam2 import build_sam2

class SAM2FeatureExtractor(nn.Module):
    def __init__(self, config_file, ckpt_path=None, device="cuda"):
        super().__init__()
        self.sam2 = build_sam2(config_file, ckpt_path, device)
        self.image_encoder = self.sam2.image_encoder

    def forward(self, x):
        debug_print(f"⚾ SAM2FeatureExtractor input shape: {x.shape}")
        
        # Get the original output
        original_output = self.image_encoder(x)
        debug_print(f"Original SAM2 output shape: {original_output['vision_features'].shape}")

        # Extract intermediate features
        features = []
        for i, feat in enumerate(original_output['backbone_fpn']):
            features.append(feat)
            debug_print(f"Feature {i+1} shape: {feat.shape}")

        debug_print(f"SAM2FeatureExtractor output shapes: {[f.shape for f in features]}")

        return features

# Usage example
config_file = "path/to/sam2_config.yaml"
ckpt_path = "path/to/sam2_checkpoint.pt"
feature_extractor = SAM2FeatureExtractor(config_file, ckpt_path)

# Example input
x = torch.randn(1, 3, 256, 256)
features = feature_extractor(x)