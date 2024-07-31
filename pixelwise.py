import torch.nn.utils.spectral_norm as spectral_norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PixelwiseSeparateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SNPixelwiseSeparateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                                 stride=stride, padding=padding, groups=in_channels))
        self.pointwise = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x