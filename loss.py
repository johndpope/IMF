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
    def __init__(self, eye_weight=1):
        super(MediaPipeEyeEnhancementLoss, self).__init__()
        self.eye_weight = eye_weight
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    def forward(self, reconstructed, target):
        # Base loss (e.g., L1 or MSE)
        base_loss = F.l1_loss(reconstructed, target)

        # Convert tensors to numpy arrays for MediaPipe
        recon_np = (reconstructed[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        target_np = (target[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Detect facial landmarks
        recon_results = self.face_mesh.process(cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
        target_results = self.face_mesh.process(cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR))

        eye_loss = 0.0
        if recon_results.multi_face_landmarks and target_results.multi_face_landmarks:
            # Get eye landmarks (assuming first face)
            recon_landmarks = recon_results.multi_face_landmarks[0].landmark
            target_landmarks = target_results.multi_face_landmarks[0].landmark

            # Define eye indices (you may need to adjust these)
            left_eye_indices = list(range(362, 374))
            right_eye_indices = list(range(263, 273))

            # Calculate bounding boxes for eyes
            def get_eye_box(landmarks, indices):
                x_min = min(landmarks[i].x for i in indices)
                x_max = max(landmarks[i].x for i in indices)
                y_min = min(landmarks[i].y for i in indices)
                y_max = max(landmarks[i].y for i in indices)
                return [int(x_min * recon_np.shape[1]), int(y_min * recon_np.shape[0]),
                        int(x_max * recon_np.shape[1]), int(y_max * recon_np.shape[0])]

            recon_left_eye = get_eye_box(recon_landmarks, left_eye_indices)
            recon_right_eye = get_eye_box(recon_landmarks, right_eye_indices)
            target_left_eye = get_eye_box(target_landmarks, left_eye_indices)
            target_right_eye = get_eye_box(target_landmarks, right_eye_indices)

            # Compute eye-specific loss
            for recon_eye, target_eye in zip([recon_left_eye, recon_right_eye], [target_left_eye, target_right_eye]):
                recon_eye_region = reconstructed[:, :, recon_eye[1]:recon_eye[3], recon_eye[0]:recon_eye[2]]
                target_eye_region = target[:, :, target_eye[1]:target_eye[3], target_eye[0]:target_eye[2]]
                
                # Ensure both regions have the same size
                min_h = min(recon_eye_region.size(2), target_eye_region.size(2))
                min_w = min(recon_eye_region.size(3), target_eye_region.size(3))
                recon_eye_region = recon_eye_region[:, :, :min_h, :min_w]
                target_eye_region = target_eye_region[:, :, :min_h, :min_w]
                
                eye_loss += F.l1_loss(recon_eye_region, target_eye_region)

        # Combine losses
        total_loss = base_loss + self.eye_weight * eye_loss
        return total_loss
