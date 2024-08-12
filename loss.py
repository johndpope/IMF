import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model import debug_print
import lpips



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
