import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from timm import create_model
from transformers import Swinv2Model, Swinv2Config
import torchvision.models as models
from torchvision.models import swin_v2_b


DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        

from torchvision.models import swin_v2_s, Swin_V2_S_Weights

class SwinV2LatentTokenEncoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Load pre-trained Swin V2 Small model
        self.swin_v2 = swin_v2_s(weights=Swin_V2_S_Weights.DEFAULT)
        
        # Remove the classification head
        self.swin_v2.head = nn.Identity()
        
        # Add a custom head to match the desired latent dimension
        self.latent_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        # SwinV2 forward pass
        features = self.swin_v2(x)
        
        # Custom head to produce latent vector
        latent = self.latent_head(features)
        
        return latent
class SwinV2FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_v2_b(pretrained=True)
        
        # Remove the avgpool and fc layers
        self.swin.avgpool = nn.Identity()
        self.swin.head = nn.Identity()

    def forward(self, x):
        debug_print(f"👟 SwinV2FeatureExtractor input shape: {x.shape}")
        
        features = []
        
        # Pass through the patch embedding
        x = self.swin.features[0](x)
        x = self.swin.features[1](x)
        
        # Stage 1
        x = self.swin.features[2](x)
        features.append(x.permute(0, 3, 1, 2))
        debug_print(f"Stage 1 shape: {features[-1].shape}")
        
        # Stage 2
        x = self.swin.features[3](x)
        x = self.swin.features[4](x)
        features.append(x.permute(0, 3, 1, 2))
        debug_print(f"Stage 2 shape: {features[-1].shape}")
        
        # Stage 3
        x = self.swin.features[5](x)
        x = self.swin.features[6](x)
        features.append(x.permute(0, 3, 1, 2))
        debug_print(f"Stage 3 shape: {features[-1].shape}")
        
        # Stage 4
        x = self.swin.features[7](x)
        features.append(x.permute(0, 3, 1, 2))
        debug_print(f"Stage 4 shape: {features[-1].shape}")
        
        return features
class SwinFeatureExtractor(nn.Module):
    def __init__(self, output_channels=[128, 256, 512, 512]):
        super().__init__()
        debug_print(f"Initializing SwinFeatureExtractor with output_channels: {output_channels}")
        
        # Load a pre-trained Swin Transformer model
        self.swin = create_model('swin_base_patch4_window7_224', pretrained=True, features_only=True)
        
        # Freeze the Swin parameters
        for param in self.swin.parameters():
            param.requires_grad = False
        
        # Get the number of channels for each stage
        swin_channels = self.swin.feature_info.channels()
        
        # Add additional convolutional layers to adjust channel dimensions
        self.adjust1 = nn.Conv2d(swin_channels[1], output_channels[0], kernel_size=1)
        self.adjust2 = nn.Conv2d(swin_channels[2], output_channels[1], kernel_size=1)
        self.adjust3 = nn.Conv2d(swin_channels[3], output_channels[2], kernel_size=1)
        self.adjust4 = nn.Conv2d(swin_channels[4], output_channels[3], kernel_size=1)

    def forward(self, x):
        debug_print(f"👟 SwinFeatureExtractor input shape: {x.shape}")
        
        # Get the features from Swin Transformer
        features = self.swin(x)
        
        # Skip the first feature map and adjust the rest
        adjusted_features = []
        for i, feat in enumerate(features[1:], start=1):
            adjusted = getattr(self, f'adjust{i}')(feat)
            debug_print(f"After adjust{i}: {adjusted.shape}")
            adjusted_features.append(adjusted)
        
        debug_print(f"SwinFeatureExtractor output: {len(adjusted_features)} features")
        for i, feat in enumerate(adjusted_features):
            debug_print(f"  Feature {i+1} shape: {feat.shape}")
        
        return adjusted_features


class ViTFeatureExtractor(nn.Module):
    def __init__(self, output_channels=[128, 256, 512, 512]):
        super().__init__()
        debug_print(f"Initializing ViTFeatureExtractor with output_channels: {output_channels}")
        
        # Load a pre-trained ViT-B/16 model
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Freeze the ViT parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Define the number of features at each layer
        vit_features = 768  # ViT-B/16 has 768 features at each layer
        
        # Add additional convolutional layers to adjust channel dimensions
        self.adjust1 = nn.Conv2d(vit_features, output_channels[0], kernel_size=1)
        self.adjust2 = nn.Conv2d(vit_features, output_channels[1], kernel_size=1)
        self.adjust3 = nn.Conv2d(vit_features, output_channels[2], kernel_size=1)
        self.adjust4 = nn.Conv2d(vit_features, output_channels[3], kernel_size=1)

    def forward(self, x):
        debug_print(f"👟 ViTFeatureExtractor input shape: {x.shape}")
        
        # Get the hidden states from ViT
        vit_output = self.vit(x, output_hidden_states=True)
        hidden_states = vit_output.hidden_states
        
        # We'll use the output of the 4th, 8th, and 12th layers (skipping the first)
        selected_layers = [4, 8, 12]
        features = []
        
        for i, layer_num in enumerate(selected_layers):
            # Get the hidden state and reshape it to 2D
            hidden_state = hidden_states[layer_num]
            B, N, C = hidden_state.shape
            H = W = int((N - 1) ** 0.5)  # Subtract 1 for the class token
            hidden_state = hidden_state[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            # Apply the adjustment layer
            adjusted_feature = getattr(self, f'adjust{i+1}')(hidden_state)
            debug_print(f"After adjust{i+1}: {adjusted_feature.shape}")
            features.append(adjusted_feature)
        
        # For the last feature, use the output of the last layer
        last_hidden_state = vit_output.last_hidden_state
        B, N, C = last_hidden_state.shape
        H = W = int((N - 1) ** 0.5)
        last_hidden_state = last_hidden_state[:, 1:, :].reshape(B, H, W, C).permute(0, 3, 1, 2)
        last_feature = self.adjust4(last_hidden_state)
        debug_print(f"After adjust4: {last_feature.shape}")
        features.append(last_feature)
        
        debug_print(f"ViTFeatureExtractor output: {len(features)} features")
        for i, feat in enumerate(features):
            debug_print(f"  Feature {i+1} shape: {feat.shape}")
        
        return features

# keep everything in 1 class to allow copying / pasting into claude / chatgpt
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, output_channels=[128, 256, 512, 512]):
        super().__init__()
        debug_print(f"Initializing ResNetFeatureExtractor with output_channels: {output_channels}")
        # Load a pre-trained ResNet model
        resnet = models.resnet50(pretrained=pretrained)
        
        # We'll use the first 4 layers of ResNet
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Add additional convolutional layers to adjust channel dimensions
        self.adjust1 = nn.Conv2d(256, output_channels[0], kernel_size=1)
        self.adjust2 = nn.Conv2d(512, output_channels[1], kernel_size=1)
        self.adjust3 = nn.Conv2d(1024, output_channels[2], kernel_size=1)
        self.adjust4 = nn.Conv2d(2048, output_channels[3], kernel_size=1)

    def forward(self, x):
        debug_print(f"👟 ResNetFeatureExtractor input shape: {x.shape}")
        features = []
        
        x = self.layer0(x)
        debug_print(f"After layer0: {x.shape}")
        
        x = self.layer1(x)
        debug_print(f"After layer1: {x.shape}")
        feature1 = self.adjust1(x)
        debug_print(f"After adjust1: {feature1.shape}")
        features.append(feature1)
        
        x = self.layer2(x)
        debug_print(f"After layer2: {x.shape}")
        feature2 = self.adjust2(x)
        debug_print(f"After adjust2: {feature2.shape}")
        features.append(feature2)
        
        x = self.layer3(x)
        debug_print(f"After layer3: {x.shape}")
        feature3 = self.adjust3(x)
        debug_print(f"After adjust3: {feature3.shape}")
        features.append(feature3)
        
        x = self.layer4(x)
        debug_print(f"After layer4: {x.shape}")
        feature4 = self.adjust4(x)
        debug_print(f"After adjust4: {feature4.shape}")
        features.append(feature4)
        
        debug_print(f"ResNetFeatureExtractor output: {len(features)} features")
        for i, feat in enumerate(features):
            debug_print(f"  Feature {i+1} shape: {feat.shape}")
        
        return features
        
        return features