import torch
import torch.nn as nn

class GradientRescale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return GradientRescaleFunction.apply(x, self.scale).clone()  # Add .clone() here


    def extra_repr(self):
        return f'scale={self.scale}'

class GradientRescaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def add_gradient_rescale(module, scale=1.0):
    """
    Recursively adds GradientRescale modules to all submodules.
    """
    for name, child in module.named_children():
        if list(child.children()):
            # If the child has children, recurse
            add_gradient_rescale(child, scale)
        else:
            # If it's a leaf module, add GradientRescale
            setattr(module, name, nn.Sequential(child, GradientRescale(scale)))