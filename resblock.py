import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block = FeatResBlock(out_channels)
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        
        # Add a 1x1 convolution for the residual connection if channel sizes differ
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.upsample(x)
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        out = self.feat_res_block(out)
        return out


class DownConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(out_channels)
        self.feat_res_block2 = FeatResBlock(out_channels)

    def forward(self, x):
        debug_print(f"DownConvResBlock input shape: {x.shape}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        debug_print(f"After conv1, bn1, relu: {out.shape}")
        out = self.avgpool(out)
        debug_print(f"After avgpool: {out.shape}")
        out = self.conv2(out)
        debug_print(f"After conv2: {out.shape}")
        out = self.feat_res_block1(out)
        debug_print(f"After feat_res_block1: {out.shape}")
        out = self.feat_res_block2(out)
        debug_print(f"DownConvResBlock output shape: {out.shape}")
        return out



class FeatResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu2(out)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=2 if downsample else 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.downsample = downsample
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out
    

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, demodulate=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate

    def forward(self, x, style):
        batch, in_channel, height, width = x.shape
        style = style.view(batch, 1, in_channel, 1, 1)
        
        # Weight modulation
        weight = self.weight.unsqueeze(0) * style
        
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.weight.size(0), 1, 1, 1)

        weight = weight.view(
            batch * self.weight.size(0), in_channel, self.weight.size(2), self.weight.size(3)
        )

        x = x.view(1, batch * in_channel, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.weight.size(0), height, width)

        return out

class StyledConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, upsample=False, demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, kernel_size, demodulate=demodulate)
        self.style = nn.Linear(style_dim, in_channels)
        self.upsample = upsample
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, latent):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        style = self.style(latent)
        x = self.conv(x, style)
        x = self.activation(x)
        return x





def test_upconvresblock(block, input_shape):
    print("\nTesting UpConvResBlock")
    x = torch.randn(input_shape)
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape[2:] == tuple(2*x for x in x.shape[2:]), "UpConvResBlock should double spatial dimensions"
    assert output.shape[1] == block.conv2.out_channels, "Output channels should match block's out_channels"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_downconvresblock(block, input_shape):
    print("\nTesting DownConvResBlock")
    x = torch.randn(input_shape)
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape[2:] == tuple(x//2 for x in x.shape[2:]), "DownConvResBlock should halve spatial dimensions"
    assert output.shape[1] == block.conv2.out_channels, "Output channels should match block's out_channels"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_featresblock(block, input_shape):
    print("\nTesting FeatResBlock")
    x = torch.randn(input_shape)
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape == x.shape, "FeatResBlock should maintain input shape"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_modulatedconv2d(conv, input_shape, style_dim):
    print("\nTesting ModulatedConv2d")
    x = torch.randn(input_shape)
    style = torch.randn(input_shape[0], style_dim)
    output = conv(x, style)
    print(f"Input shape: {x.shape}, Style shape: {style.shape}, Output shape: {output.shape}")
    assert output.shape[1] == conv.weight.shape[0], "Output channels should match conv's out_channels"

    # Test gradient flow
    output.sum().backward()
    assert conv.weight.grad is not None, "No gradient for weight"
    print(f"Weight gradient shape: {conv.weight.grad.shape}")

def test_styledconv(conv, input_shape, latent_dim):
    print("\nTesting StyledConv")
    x = torch.randn(input_shape)
    latent = torch.randn(input_shape[0], latent_dim)
    output = conv(x, latent)
    print(f"Input shape: {x.shape}, Latent shape: {latent.shape}, Output shape: {output.shape}")
    expected_shape = list(input_shape)
    expected_shape[1] = conv.conv.weight.shape[0]
    if conv.upsample:
        expected_shape[2] *= 2
        expected_shape[3] *= 2
    assert list(output.shape) == expected_shape, f"Output shape {output.shape} doesn't match expected {expected_shape}"

    # Test gradient flow
    output.sum().backward()
    for name, param in conv.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_resblock(resblock, input_shape):
    print("\nTesting ResBlock")
    x = torch.randn(input_shape)
    output = resblock(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    expected_output_shape = list(input_shape)
    expected_output_shape[1] = resblock.out_channels
    if resblock.downsample:
        expected_output_shape[2] //= 2
        expected_output_shape[3] //= 2
    assert tuple(output.shape) == tuple(expected_output_shape), f"Expected shape {expected_output_shape}, got {output.shape}"

    # Test gradient flow
    output.sum().backward()
    for name, param in resblock.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

    # Test residual connection
    resblock.eval()
    with torch.no_grad():
        residual_output = resblock(x)
        identity = resblock.shortcut(x)
        main_path = resblock.bn2(resblock.conv2(resblock.relu(resblock.bn1(resblock.conv1(x)))))
        direct_output = resblock.relu(main_path + identity)
        assert torch.allclose(residual_output, direct_output, atol=1e-6), "Residual connection not working correctly"
    
    print("ResBlock test passed successfully!")



def test_block_with_dropout(block, input_shape, block_name):
    print(f"\nTesting {block_name}")
    x = torch.randn(input_shape)
    
    # Test in training mode
    block.train()
    output_train = block(x)
    print(f"Training mode - Input shape: {x.shape}, Output shape: {output_train.shape}")

    # Test in eval mode
    block.eval()
    with torch.no_grad():
        output_eval = block(x)
    print(f"Eval mode - Input shape: {x.shape}, Output shape: {output_eval.shape}")

    # Check if outputs are different in train and eval modes
    assert not torch.allclose(output_train, output_eval), f"{block_name} outputs should differ between train and eval modes due to dropout"

    # Test gradient flow
    block.train()  # Set back to train mode for gradient check
    output_train.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

    print(f"{block_name} test passed successfully!")
def visualize_feature_maps(block, input_shape, num_channels=4, latent_dim=None):
    x = torch.randn(input_shape)
    if isinstance(block, StyledConv):
        latent = torch.randn(input_shape[0], latent_dim)
        output = block(x, latent)
    else:
        output = block(x)
    
    fig, axs = plt.subplots(1, num_channels, figsize=(20, 5))
    for i in range(num_channels):
        axs[i].imshow(output[0, i].detach().cpu().numpy(), cmap='viridis')
        axs[i].axis('off')
        axs[i].set_title(f'Channel {i}')
    plt.tight_layout()
    plt.show()


# Run all tests
upconv = UpConvResBlock(64, 128)
test_upconvresblock(upconv, (1, 64, 56, 56))
visualize_feature_maps(upconv, (1, 64, 56, 56))

downconv = DownConvResBlock(128, 256)
test_downconvresblock(downconv, (1, 128, 56, 56))
visualize_feature_maps(downconv, (1, 128, 56, 56))

featres = FeatResBlock(256)
test_featresblock(featres, (1, 256, 28, 28))
visualize_feature_maps(featres, (1, 256, 28, 28))

modconv = ModulatedConv2d(64, 128, 3)
test_modulatedconv2d(modconv, (1, 64, 56, 56), 64)

styledconv = StyledConv(64, 128, 3, 32, upsample=True)
test_styledconv(styledconv, (1, 64, 56, 56), 32)
visualize_feature_maps(styledconv, (1, 64, 56, 56), num_channels=4, latent_dim=32)


resblock = ResBlock(64, 128, downsample=True)
test_resblock(resblock, (1, 64, 56, 56))
visualize_feature_maps(resblock, (1, 64, 56, 56))

# Usage
resblock = ResBlock(64, 64)
test_resblock(resblock, (1, 64, 56, 56))