import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model import debug_#print
import lpips

# Wasserstein Loss:
# Pros:

# Provides a meaningful distance metric between distributions.
# Often leads to more stable training and better convergence.
# Can help prevent mode collapse.

# Cons:

# Requires careful weight clipping or gradient penalty implementation for Lipschitz constraint.
# May converge slower than other losses in some cases.


# Hinge Loss:
# Pros:

# Often results in sharper and more realistic images.
# Good stability in training, especially for complex architectures.
# Works well with spectral normalization.

# Cons:

# May be sensitive to outliers.
# Can sometimes lead to more constrained generator outputs.


# Vanilla GAN Loss:
# Pros:

# Simple and straightforward implementation.
# Works well for many standard GAN applications.

# Cons:

# Can suffer from vanishing gradients and mode collapse.
# Often less stable than Wasserstein or Hinge loss, especially for complex models.

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

def feature_matching_loss(real_features, fake_features):
    loss = 0
    for r_feat, f_feat in zip(real_features, fake_features):
        loss += F.mse_loss(r_feat.mean(0), f_feat.mean(0))
    return loss

def gan_loss_fn(real_outputs, fake_outputs, loss_type):
    """
    Unified GAN loss function that can switch between different loss types.
    """
    if loss_type == "feature_matching":
        return feature_matching_loss(real_outputs, fake_outputs)
    if loss_type == "wasserstein":
        return wasserstein_loss(real_outputs, fake_outputs)
    elif loss_type == "hinge":
        return hinge_loss(real_outputs, fake_outputs)
    elif loss_type == "vanilla":
        return vanilla_gan_loss(real_outputs, fake_outputs)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")



def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    #print(f"Real samples shape: {real_samples.shape}")
    #print(f"Fake samples shape: {fake_samples.shape}")
    
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(real_samples.device)
    #print(f"Alpha shape: {alpha.shape}")
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #print(f"Interpolates shape: {interpolates.shape}")
    
    d_interpolates = discriminator(interpolates)
    #print(f"Discriminator output type: {type(d_interpolates)}")
    #print(f"Number of scales: {len(d_interpolates)}")
    
    gradient_penalty = 0
    for scale_output in d_interpolates:
        #print(f"Scale output shape: {scale_output.shape}")
        
        if scale_output.dim() > 2:
            scale_output = scale_output.mean(dim=[2, 3])  # Average over spatial dimensions
        #print(f"Processed scale output shape: {scale_output.shape}")
        
        fake = torch.ones(scale_output.shape).to(real_samples.device)
        gradients = torch.autograd.grad(
            outputs=scale_output, inputs=interpolates,
            grad_outputs=fake, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        #print(f"Gradients shape: {gradients.shape}")
        
        gradients = gradients.view(gradients.size(0), -1)
        #print(f"Reshaped gradients shape: {gradients.shape}")
        
        scale_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        #print(f"Scale gradient penalty: {scale_penalty.item()}")
        
        gradient_penalty += scale_penalty
    
    # Average the gradient penalty across all scales
    gradient_penalty /= len(d_interpolates)
    #print(f"Final gradient penalty: {gradient_penalty.item()}")
    
    return gradient_penalty

# Usage in training loop