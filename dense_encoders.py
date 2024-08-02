import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from timm import create_model
from transformers import Swinv2Model, Swinv2Config
import torchvision.models as models


DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        
class SwinV2FeatureExtractor(nn.Module):
    def __init__(self, output_channels=[128, 256, 512, 512]):
        super().__init__()
        debug_print(f"Initializing SwinV2FeatureExtractor with output_channels: {output_channels}")
        
        # Load a pre-trained Swin Transformer V2 model
        self.swin = Swinv2Model.from_pretrained("microsoft/swinv2-base-patch4-window8-256")
        
        # Freeze the Swin parameters
        for param in self.swin.parameters():
            param.requires_grad = False
        
        # Get the correct number of channels for each stage
        self.swin_channels = [128, 256, 512, 1024]  # Adjusted based on the error message
        debug_print(f"Swin channels: {self.swin_channels}")
        
        # Add additional convolutional layers to adjust channel dimensions
        self.adjust1 = nn.Conv2d(self.swin_channels[0], output_channels[0], kernel_size=1)
        self.adjust2 = nn.Conv2d(self.swin_channels[1], output_channels[1], kernel_size=1)
        self.adjust3 = nn.Conv2d(self.swin_channels[2], output_channels[2], kernel_size=1)
        self.adjust4 = nn.Conv2d(self.swin_channels[3], output_channels[3], kernel_size=1)

    def forward(self, x):
        debug_print(f"👟 SwinV2FeatureExtractor input shape: {x.shape}")
        
        # Get the features from Swin Transformer V2
        outputs = self.swin(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # We'll use the output of the 1st, 2nd, 3rd, and 4th (final) stages
        selected_stages = [1, 3, 5, 7]  # Adjusted to get the correct feature maps
        features = []
        
        for i, stage in enumerate(selected_stages):
            # Get the hidden state and reshape it to 2D
            hidden_state = hidden_states[stage]
            B, N, C = hidden_state.shape
            H = W = int(N ** 0.5)
            hidden_state = hidden_state.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            debug_print(f"Stage {stage} shape: {hidden_state.shape}")
            
            # Apply the adjustment layer
            adjusted_feature = getattr(self, f'adjust{i+1}')(hidden_state)
            debug_print(f"After adjust{i+1}: {adjusted_feature.shape}")
            features.append(adjusted_feature)
        
        debug_print(f"SwinV2FeatureExtractor output: {len(features)} features")
        for i, feat in enumerate(features):
            debug_print(f"  Feature {i+1} shape: {feat.shape}")
        
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