import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

class ReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4*g*g / ((k+1)*(k+1))
        self.input_size, self.output_size = input_size, output_size
        phase_low = np.arange(-k, g) / g
        phase_height = phase_low + (k+1) / g
        self.phase_low = nn.Parameter(torch.Tensor(np.array([phase_low for i in range(input_size)])),
                                      requires_grad=train_ab)
        self.phase_height = nn.Parameter(torch.Tensor(np.array([phase_height for i in range(input_size)])),
                                         requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g+k, input_size))
    def forward(self, x):
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_height - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size, 1))
        return x


class ReLUKAN(nn.Module):
    def __init__(self, width, grid, k):
        super().__init__()
        self.width = width
        self.grid = grid
        self.k = k
        self.rk_layers = []
        for i in range(len(width) - 1):
            self.rk_layers.append(ReLUKANLayer(width[i], grid, k, width[i+1]))
            # if len(width) - i > 2:
            #     self.rk_layers.append()
        self.rk_layers = nn.ModuleList(self.rk_layers)

    def forward(self, x):
        for rk_layer in self.rk_layers:
            x = rk_layer(x)
        # x = x.reshape((len(x), self.width[-1]))
        return x


def time_model(model, input_tensor, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    end_time = time.time()
    return (end_time - start_time) / num_runs




class ImprovedReLUKANLayer(nn.Module):
    def __init__(self, input_size: int, g: int, k: int, output_size: int, train_ab: bool = True):
        super().__init__()
        self.g, self.k = g, k
        self.r = 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        
        phase_low = torch.arange(-k, g, dtype=torch.float32) / g
        phase_height = phase_low + (k + 1) / g
        
        self.phase_low = nn.Parameter(phase_low.repeat(input_size, 1), requires_grad=train_ab)
        self.phase_height = nn.Parameter(phase_height.repeat(input_size, 1), requires_grad=train_ab)
        
        self.conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add extra dimension for broadcasting
        x1 = F.relu(x - self.phase_low)
        x2 = F.relu(self.phase_height - x)
        x = (x1 * x2).pow(2) * self.r
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        return x.squeeze(-1)  # Remove last dimension

class ImprovedReLUKAN(nn.Module):
    def __init__(self, widths, grid, k):
        super().__init__()
        self.layers = nn.ModuleList([
            ImprovedReLUKANLayer(in_size, grid, k, out_size)
            for in_size, out_size in zip(widths[:-1], widths[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
# Create instances of both the original and improved models
original_model = ReLUKAN([1, 1], 5, 3)
improved_model = ImprovedReLUKAN(...)

# Create a sample input tensor
input_tensor = torch.Tensor([np.arange(0, 1024) / 1024]).T

# Time both models
original_time = time_model(original_model, input_tensor)
improved_time = time_model(improved_model, input_tensor)

print(f"Original model average time: {original_time:.6f} seconds")
print(f"Improved model average time: {improved_time:.6f} seconds")
print(f"Speedup: {original_time / improved_time:.2f}x")