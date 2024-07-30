from facenet_pytorch import InceptionResnetV1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lpips import LPIPS

class PerceptualLoss(nn.Module):
    def __init__(self, device, weights={'vgg19': 20.0, 'vggface': 5.0, 'gaze': 4.0, 'lpips': 10.0}):
        super(PerceptualLoss, self).__init__()
        self.device = device
        self.weights = weights

        # VGG19 network
        vgg19 = models.vgg19(pretrained=True).features
        self.vgg19 = nn.Sequential(*[vgg19[i] for i in range(30)]).to(device).eval()
        self.vgg19_layers = [1, 6, 11, 20, 29]

        # VGGFace network
        self.vggface = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        self.vggface_layers = [4, 5, 6, 7]


        # LPips   
        self.lpips = LPIPS(net='vgg').to(device).eval()

    def forward(self, predicted, target, use_fm_loss=False):
        # Normalize input images
        predicted = self.normalize_input(predicted)
        target = self.normalize_input(target)

        # Compute VGG19 perceptual loss
        vgg19_loss = self.compute_vgg19_loss(predicted, target)

        # Compute VGGFace perceptual loss
        vggface_loss = self.compute_vggface_loss(predicted, target)

        # Compute gaze loss
        # gaze_loss = self.gaze_loss(predicted, target)
        
        # Compute LPIPS loss
        lpips_loss = self.lpips(predicted, target).mean()

        # Compute total perceptual loss
        total_loss = (
            self.weights['vgg19'] * vgg19_loss +
            self.weights['vggface'] * vggface_loss +
            self.weights['lpips'] * lpips_loss +
            self.weights['gaze'] * 1 #gaze_loss
        )

        if use_fm_loss:
            # Compute feature matching loss
            fm_loss = self.compute_feature_matching_loss(predicted, target)
            total_loss += fm_loss

        return total_loss

    def compute_vgg19_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target)

    def compute_vggface_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vggface, self.vggface_layers, predicted, target)

    def compute_feature_matching_loss(self, predicted, target):
        return self.compute_perceptual_loss(self.vgg19, self.vgg19_layers, predicted, target, detach=True)

    def compute_perceptual_loss(self, model, layers, predicted, target, detach=False):
        loss = 0.0
        predicted_features = predicted
        target_features = target
        #print(f"predicted_features:{predicted_features.shape}")
        #print(f"target_features:{target_features.shape}")

        for i, layer in enumerate(model.children()):
            # print(f"i{i}")
            if isinstance(layer, nn.Conv2d):
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            elif isinstance(layer, nn.Linear):
                predicted_features = predicted_features.view(predicted_features.size(0), -1)
                target_features = target_features.view(target_features.size(0), -1)
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)
            else:
                predicted_features = layer(predicted_features)
                target_features = layer(target_features)

            if i in layers:
                if detach:
                    loss += torch.mean(torch.abs(predicted_features - target_features.detach()))
                else:
                    loss += torch.mean(torch.abs(predicted_features - target_features))

        return loss

    def normalize_input(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std