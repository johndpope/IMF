import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model import debug_print
import lpips

class LPIPSPerceptualLoss(nn.Module):
    def __init__(self, net='alex', debug=True):
        super(LPIPSPerceptualLoss, self).__init__()
        self.debug = debug
        self.lpips_model = lpips.LPIPS(net=net)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        debug_print(f"\nLPIPSPerceptualLoss Forward Pass:")
        debug_print(f"Input shapes: x: {x.shape}, y: {y.shape}")
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError(f"Expected 4D input tensors, got x: {x.dim()}D and y: {y.dim()}D")
        
        total_loss = 0
        for i in range(4):  # Downsample 4 times (i ∈ [0, 3])
            debug_print(f"\n  Scale {i+1}:")
            if i > 0:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=False)

            debug_print(f"    Current shapes: x: {x.shape}, y: {y.shape}")
            
            # Compute LPIPS loss
            scale_loss = self.lpips_model(x, y).mean()
            total_loss += scale_loss

            debug_print(f"    Scale {i+1} LPIPS loss: {scale_loss.item():.6f}")

        debug_print(f"\nTotal LPIPS loss: {total_loss.item():.6f}")
        return total_loss
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self, debug=True):
        super(VGGPerceptualLoss, self).__init__()
        self.debug = debug
        vgg = models.vgg19(pretrained=True).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:2]),
            nn.Sequential(*list(vgg.children())[2:7]),
            nn.Sequential(*list(vgg.children())[7:12]),
            nn.Sequential(*list(vgg.children())[12:21]),
            nn.Sequential(*list(vgg.children())[21:30])
        ])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        debug_print(f"\nVGGPerceptualLoss Forward Pass:")
        debug_print(f"Input shapes: x: {x.shape}, y: {y.shape}")
        if x.dim() != 4 or y.dim() != 4:
            raise ValueError(f"Expected 4D input tensors, got x: {x.dim()}D and y: {y.dim()}D")
        
        total_loss = 0
        for i in range(4):  # Downsample 4 times (i ∈ [0, 3])
            debug_print(f"\n  Scale {i+1}:")
            if i > 0:
                x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
                y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=False)

            debug_print(f"    Current shapes: x: {x.shape}, y: {y.shape}")
            
            # Compute VGG features
            x_features, y_features = [], []
            x_current, y_current = x, y
            for j, slice in enumerate(self.slices):
                x_current = slice(x_current)
                y_current = slice(y_current)
                debug_print(f"      After VGG slice {j+1}: x: {x_current.shape}, y: {y_current.shape}")
                x_features.append(x_current)
                y_features.append(y_current)
            
            # Compute loss for each VGG layer
            scale_loss = 0
            for j in range(len(x_features)):
                layer_loss = F.l1_loss(x_features[j], y_features[j])
                scale_loss += layer_loss
 
                debug_print(f"      Layer {j+1} loss: {layer_loss.item():.6f}")
            
            total_loss += scale_loss

            debug_print(f"    Scale {i+1} total loss: {scale_loss.item():.6f}")
        

        debug_print(f"\nTotal perceptual loss: {total_loss.item():.6f}")
        return total_loss


def wasserstein_loss(real_outputs, fake_outputs):
    """
    Wasserstein loss for GANs.
    """
    real_loss = sum(-torch.mean(out) for out in real_outputs)
    fake_loss = sum(torch.mean(out) for out in fake_outputs)
    return real_loss + fake_loss

def hinge_loss(real_outputs, fake_outputs):
    """
    Hinge loss for GANs.
    """
    real_loss = sum(torch.mean(F.relu(1 - out)) for out in real_outputs)
    fake_loss = sum(torch.mean(F.relu(1 + out)) for out in fake_outputs)
    return real_loss + fake_loss

def vanilla_gan_loss(real_outputs, fake_outputs):
    """
    Vanilla GAN loss.
    """
    real_loss = sum(F.binary_cross_entropy_with_logits(out, torch.ones_like(out)) for out in real_outputs)
    fake_loss = sum(F.binary_cross_entropy_with_logits(out, torch.zeros_like(out)) for out in fake_outputs)
    return real_loss + fake_loss

def gan_loss_fn(real_outputs, fake_outputs, loss_type):
    """
    Unified GAN loss function that can switch between different loss types.
    """
    if loss_type == "wasserstein":
        return wasserstein_loss(real_outputs, fake_outputs)
    elif loss_type == "hinge":
        return hinge_loss(real_outputs, fake_outputs)
    elif loss_type == "vanilla":
        return vanilla_gan_loss(real_outputs, fake_outputs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
