import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from model import debug_print
import lpips
import torch.nn.functional as F
from torchvision import models
import mediapipe as mp
import numpy as np
import cv2
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

class MediaPipeEyeEnhancementLoss(nn.Module):
    def __init__(self, eye_weight=10.0):
        super(MediaPipeEyeEnhancementLoss, self).__init__()
        self.eye_weight = eye_weight
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    @torch.no_grad()
    def get_eye_boxes(self, image):
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        results = self.face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye_indices = list(range(362, 374))
        right_eye_indices = list(range(263, 273))

        def get_eye_box(indices):
            x_min = min(landmarks[i].x for i in indices)
            x_max = max(landmarks[i].x for i in indices)
            y_min = min(landmarks[i].y for i in indices)
            y_max = max(landmarks[i].y for i in indices)
            return [int(x_min * image_np.shape[1]), int(y_min * image_np.shape[0]),
                    int(x_max * image_np.shape[1]), int(y_max * image_np.shape[0])]

        return get_eye_box(left_eye_indices), get_eye_box(right_eye_indices)

    def forward(self, reconstructed, target):
        # Base loss (e.g., L1 or MSE)
        base_loss = F.l1_loss(reconstructed, target)

        eye_loss = 0.0
        for i in range(reconstructed.size(0)):  # Iterate over batch
            recon_eyes = self.get_eye_boxes(reconstructed[i])
            target_eyes = self.get_eye_boxes(target[i])

            if recon_eyes is None or target_eyes is None:
                continue  # Skip this sample if face not detected

            for recon_eye, target_eye in zip(recon_eyes, target_eyes):
                recon_eye_region = reconstructed[i, :, recon_eye[1]:recon_eye[3], recon_eye[0]:recon_eye[2]]
                target_eye_region = target[i, :, target_eye[1]:target_eye[3], target_eye[0]:target_eye[2]]
                
                # Ensure both regions have the same size
                min_h = min(recon_eye_region.size(1), target_eye_region.size(1))
                min_w = min(recon_eye_region.size(2), target_eye_region.size(2))
                recon_eye_region = recon_eye_region[:, :min_h, :min_w]
                target_eye_region = target_eye_region[:, :min_h, :min_w]
                
                eye_loss += F.l1_loss(recon_eye_region, target_eye_region)

        # Normalize eye loss by batch size
        eye_loss /= reconstructed.size(0)

        # Combine losses
        total_loss = base_loss + self.eye_weight * eye_loss
        return total_loss
