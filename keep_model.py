import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
import torch.nn.utils.spectral_norm as spectral_norm
from vit import ImplicitMotionAlignment
from resblocks import  FeatResBlock,UpConvResBlock,DownConvResBlock
from lia_resblocks import StyledConv,EqualConv2d,EqualLinear,ResBlock # these are correct https://github.com/hologerry/IMF/issues/4  "You can refer to this repo https://github.com/wyhsirius/LIA/ for StyleGAN2 related code, such as Encoder, Decoder."
from helper import normalize
from model import LatentTokenDecoder,LatentTokenEncoder,DenseFeatureEncoder,FrameDecoder,MappingNetwork,NoiseInjection
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import math
# from common import DownConvResBlock,UpConvResBlock
import colored_traceback.auto # makes terminal show color coded output when crash
import random

DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class IMFKeepModel(nn.Module):
    def __init__(self, latent_dim=32, base_channels=64, num_layers=4, noise_level=0.1, style_mix_prob=0.5):
        super().__init__()
        
        self.latent_token_encoder = LatentTokenEncoder(
            initial_channels=64,
            output_channels=[256, 256, 512, 512, 512, 512],
            dm=latent_dim
        ) 
        
        self.latent_token_decoder = LatentTokenDecoder()
        self.feature_dims = [128, 256, 512, 512]
        self.spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8)]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels=self.feature_dims)

        self.motion_dims = [256, 512, 512, 512]
        
        # KEEP components
        self.kalman_filter_network = KalmanFilterNetwork(latent_dim)
        self.cross_frame_attention = CrossFrameAttention(latent_dim)
        
        # Initialize ImplicitMotionAlignment modules
        self.implicit_motion_alignment = nn.ModuleList()
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            spatial_dim = self.spatial_dims[i]
            alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim, spatial_dim=spatial_dim)
            self.implicit_motion_alignment.append(alignment_module)

        self.frame_decoder = FrameDecoder()
        self.noise_level = noise_level
        self.style_mix_prob = style_mix_prob

        self.mapping_network = MappingNetwork(latent_dim, latent_dim, depth=8)
        self.noise_injection = NoiseInjection()

    def forward(self, x_current, x_reference, x_previous=None, t_previous=None):
        x_current = x_current.requires_grad_()
        x_reference = x_reference.requires_grad_()

        f_r = self.dense_feature_encoder(x_reference)
        t_r = self.latent_token_encoder(x_reference)
        t_c = self.latent_token_encoder(x_current)

        # Apply Kalman filtering
        if x_previous is not None and t_previous is not None:
            t_c = self.kalman_filter_network(t_c, t_previous, x_current, x_previous)

        m_c = self.latent_token_decoder(t_c)
        m_r = self.latent_token_decoder(t_r)

        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            f_r_i = f_r[i]
            align_layer = self.implicit_motion_alignment[i]
            m_c_i = m_c[i] 
            m_r_i = m_r[i]
            aligned_feature = align_layer(m_c_i, m_r_i, f_r_i)
            aligned_features.append(aligned_feature)

        # Apply Cross-Frame Attention
        aligned_features = self.cross_frame_attention(aligned_features)

        x_reconstructed = self.frame_decoder(aligned_features)
        x_reconstructed = normalize(x_reconstructed)

        return x_reconstructed, t_c

class KalmanFilterNetwork(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.uncertainty_network = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.gain_network = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, t_current, t_previous, x_current, x_previous):
        uncertainty = self.uncertainty_network(torch.cat([t_current, t_previous], dim=-1))
        gain = self.gain_network(torch.cat([t_current, t_previous, uncertainty], dim=-1))
        innovation = t_current - t_previous
        t_updated = t_previous + gain * innovation
        return t_updated

class CrossFrameAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(feature_dim, feature_dim // 8, 1)
        self.key_conv = nn.Conv2d(feature_dim, feature_dim // 8, 1)
        self.value_conv = nn.Conv2d(feature_dim, feature_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        B, C, H, W = features[0].shape
        queries = [self.query_conv(f).view(B, -1, H*W).permute(0, 2, 1) for f in features]
        keys = [self.key_conv(f).view(B, -1, H*W) for f in features]
        values = [self.value_conv(f).view(B, -1, H*W) for f in features]

        attention = [torch.bmm(q, k) for q, k in zip(queries, keys)]
        attention = [F.softmax(a, dim=-1) for a in attention]

        out = [torch.bmm(v, a.permute(0, 2, 1)).view(B, C, H, W) for v, a in zip(values, attention)]
        out = [self.gamma * o + f for o, f in zip(out, features)]

        return out