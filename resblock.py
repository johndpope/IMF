import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


DEBUG = False
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)



# class DownConvResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, dropout_rate=0.1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         # self.bn2 = nn.BatchNorm2d(out_channels) # 🤷 works with not without
#         # self.dropout = nn.Dropout2d(dropout_rate) # 🤷
#         self.feat_res_block1 = FeatResBlock(out_channels)
#         self.feat_res_block2 = FeatResBlock(out_channels)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.avgpool(out)
#         out = self.conv2(out)
#         # out = self.bn2(out) # 🤷
#         # out = self.relu(out) # 🤷
#         # out = self.dropout(out) # 🤷
#         out = self.feat_res_block1(out)
#         out = self.feat_res_block2(out)
#         return out


# class FeatResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.bn1(x)
#         out = self.relu1(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out = self.relu2(out)
#         out = self.conv2(out)
#         out += residual
#         out = self.relu3(out)
#         return out


# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=False):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, downsample=False):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU(inplace=True)
        
#         self.skip_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if downsample else 1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
        
#         self.downsample = downsample
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#     def forward(self, x):
#         residual = self.skip_conv(x)
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         out += residual
#         out = self.relu2(out)
        
#         return out
    

class UpConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.feat_res_block1 = FeatResBlock(out_channels)
        self.feat_res_block2 = FeatResBlock(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.feat_res_block1(x)
        x = self.feat_res_block2(x)
        return x



class DownConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels) # 🤷 works with not without
        # self.dropout = nn.Dropout2d(dropout_rate) # 🤷
        self.feat_res_block1 = FeatResBlock(out_channels)
        self.feat_res_block2 = FeatResBlock(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        # out = self.bn2(out) # 🤷
        out = self.relu(out) # 🤷
        # out = self.dropout(out) # 🤷
        out = self.feat_res_block1(out)
        out = self.feat_res_block2(out)
        return out


class FeatResBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual
        out = self.relu2(out)
        return out
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, dropout_rate=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels  # Add this line
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=2 if downsample else 1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                          stride=2 if downsample else 1, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        out += residual
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




def test_upconvresblock(block, input_tensor):
    print("\nTesting UpConvResBlock")
    x = input_tensor
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape[2:] == tuple(2*x for x in x.shape[2:]), "UpConvResBlock should double spatial dimensions"
    assert output.shape[1] == block.conv2.out_channels, "Output channels should match block's out_channels"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_downconvresblock(block, input_tensor):
    print("\nTesting DownConvResBlock")
    x = input_tensor
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape[2:] == tuple(x//2 for x in x.shape[2:]), "DownConvResBlock should halve spatial dimensions"
    assert output.shape[1] == block.conv2.out_channels, "Output channels should match block's out_channels"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_featresblock(block, input_tensor):
    print("\nTesting FeatResBlock")
    x = input_tensor
    output = block(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    assert output.shape == x.shape, "FeatResBlock should maintain input shape"

    # Test gradient flow
    output.sum().backward()
    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        print(f"{name} gradient shape: {param.grad.shape}")

def test_modulatedconv2d(conv, input_tensor, style):
    print("\nTesting ModulatedConv2d")
    x = input_tensor
    output = conv(x, style)
    print(f"Input shape: {x.shape}, Style shape: {style.shape}, Output shape: {output.shape}")
    assert output.shape[1] == conv.weight.shape[0], "Output channels should match conv's out_channels"

    # Test gradient flow
    output.sum().backward()
    assert conv.weight.grad is not None, "No gradient for weight"
    print(f"Weight gradient shape: {conv.weight.grad.shape}")

def test_styledconv(conv, input_tensor, latent):
    print("\nTesting StyledConv")
    x = input_tensor
    output = conv(x, latent)
    print(f"Input shape: {x.shape}, Latent shape: {latent.shape}, Output shape: {output.shape}")
    expected_shape = list(x.shape)
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

def test_block_with_dropout(block, input_tensor, block_name):
    print(f"\nTesting {block_name}")
    x = input_tensor
    
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


def visualize_feature_maps(block, input_data, num_channels=4, latent_dim=None):
    if isinstance(input_data, torch.Tensor):
        x = input_data
    else:  # Assume it's a shape tuple
        x = torch.randn(input_data)
    
    if isinstance(block, StyledConv):
        latent = torch.randn(x.shape[0], latent_dim)
        output = block(x, latent)
    else:
        output = block(x)
    
    # Determine the number of intermediate outputs
    if isinstance(block, UpConvResBlock):
        intermediate_outputs = [
            block.conv1(block.upsample(x)),
            block.conv2(block.relu(block.bn1(block.conv1(block.upsample(x))))),
            block.feat_res_block1(block.conv2(block.relu(block.bn1(block.conv1(block.upsample(x)))))),
            output
        ]
        titles = ['After Conv1', 'After Conv2', 'After FeatResBlock1', 'Final Output']
    elif isinstance(block, DownConvResBlock):
        intermediate_outputs = [
            block.conv2(block.avgpool(block.relu(block.bn1(block.conv1(x))))),
            block.feat_res_block1(block.conv2(block.avgpool(block.relu(block.bn1(block.conv1(x)))))),
            block.feat_res_block2(block.feat_res_block1(block.conv2(block.avgpool(block.relu(block.bn1(block.conv1(x))))))),
            output
        ]
        titles = ['After Conv2', 'After FeatResBlock1', 'After FeatResBlock2', 'Final Output']
    elif isinstance(block, FeatResBlock):
        intermediate_outputs = [
            block.conv1(block.relu1(block.bn1(x))),
            block.conv2(block.relu2(block.bn2(block.conv1(block.relu1(block.bn1(x)))))),
            output
        ]
        titles = ['After Conv1', 'After Conv2', 'Final Output']
    elif isinstance(block, StyledConv):
        intermediate_outputs = [output]
        titles = ['Output']
    else:
        intermediate_outputs = [output]
        titles = ['Output']

    num_outputs = len(intermediate_outputs)
    fig, axs = plt.subplots(num_outputs, min(num_channels, output.shape[1]), figsize=(20, 5 * num_outputs))
    if num_outputs == 1 and min(num_channels, output.shape[1]) == 1:
        axs = np.array([[axs]])
    elif num_outputs == 1 or min(num_channels, output.shape[1]) == 1:
        axs = np.array([axs])

    for i, out in enumerate(intermediate_outputs):
        for j in range(min(num_channels, out.shape[1])):
            ax = axs[i, j] if num_outputs > 1 and min(num_channels, output.shape[1]) > 1 else axs[i]
            feature_map = out[0, j].detach().cpu().numpy()
            
            # Normalize the feature map
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{titles[i]}\nChannel {j}')
            else:
                ax.set_title(f'Channel {j}')

    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path).convert('RGB')
    
    # Define the preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess the image
    img_tensor = preprocess(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

def test_resblock_with_image(resblock, image_tensor):
    print("\nTesting ResBlock with Image")
    input_shape = image_tensor.shape
    print(f"Input shape: {input_shape}")
    
    # Pass the image through the ResBlock
    output = resblock(image_tensor)
    print(f"Output shape: {output.shape}")
    
    # Check output shape
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

    # Visualize input and output
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display input image
    input_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
    ax1.imshow(input_img)
    ax1.set_title("Input Image")
    ax1.axis('off')
    
    # Display output feature map (first channel)
    output_img = output.squeeze(0)[0].cpu().detach().numpy()
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    ax2.imshow(output_img, cmap='viridis')
    ax2.set_title("Output Feature Map (First Channel)")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("ResBlock test with image passed successfully!")


def visualize_resblock_rgb(block, input_tensor):
    x = input_tensor
    
    # Get intermediate outputs
    residual = block.shortcut(x)
    conv1_out = block.conv1(x)
    bn1_out = block.bn1(conv1_out)
    relu1_out = block.relu(bn1_out)
    conv2_out = block.conv2(relu1_out)
    bn2_out = block.bn2(conv2_out)
    dropout_out = block.dropout(bn2_out)
    
    skip_connection = dropout_out + residual
    final_output = block.relu(skip_connection)
    
    outputs = [x, conv1_out, bn1_out, relu1_out, conv2_out, bn2_out, dropout_out, residual, skip_connection, final_output]
    titles = ['Input', 'Conv1', 'BN1', 'ReLU1', 'Conv2', 'BN2', 'Dropout', 'Residual', 'Skip Connection', 'Final Output']
    
    fig, axs = plt.subplots(len(outputs), 4, figsize=(20, 4 * len(outputs)))
    
    for i, out in enumerate(outputs):
        out_np = out[0].detach().cpu().numpy()
        
        # Display first three channels (or less if there are fewer channels)
        for j in range(min(3, out_np.shape[0])):
            channel = out_np[j]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            # Create RGB image with the channel value in the corresponding color channel
            rgb_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
            rgb_channel[:, :, j] = channel
            
            axs[i, j].imshow(rgb_channel)
            axs[i, j].axis('off')
            axs[i, j].set_title(f'{titles[i]}\n{"RGB"[j]} Channel')
        
        # If there are fewer than 3 channels, hide the unused subplots
        for j in range(out_np.shape[0], 3):
            axs[i, j].axis('off')
        
        # Display combined representation
        if out_np.shape[0] >= 3:
            combined = np.stack([
                out_np[0],
                out_np[1] if out_np.shape[0] > 1 else np.zeros_like(out_np[0]),
                out_np[2] if out_np.shape[0] > 2 else np.zeros_like(out_np[0])
            ], axis=-1)
        else:
            combined = np.stack([out_np[0]] * 3, axis=-1)
        
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        axs[i, 3].imshow(combined)
        axs[i, 3].axis('off')
        axs[i, 3].set_title(f'{titles[i]}\nCombined')

    plt.tight_layout()
    plt.show()

def visualize_block_output(block, input_tensor, block_name, num_channels=4):
    print(f"\nVisualizing {block_name}")
    print(f"input_tensor: {input_tensor.shape}")
    
    x = input_tensor
    
    # Forward pass
    output = block(x)
    
    # If the output is a tuple or list (e.g., for blocks that return multiple tensors)
    if isinstance(output, (tuple, list)):
        output = output[0]  # Visualize only the first tensor
    
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

    # Visualize input and output
    fig, axs = plt.subplots(2, min(num_channels, output.shape[1]), figsize=(15, 8))
    
    # Display input channels
    for j in range(min(num_channels, x.shape[1])):
        ax = axs[0, j]
        channel = x[0, j].detach().cpu().numpy()
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        ax.imshow(channel, cmap='viridis')
        ax.set_title(f"Input Channel {j}")
        ax.axis('off')
    
    # Display output channels
    for j in range(min(num_channels, output.shape[1])):
        ax = axs[1, j]
        channel = output[0, j].detach().cpu().numpy()
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        ax.imshow(channel, cmap='viridis')
        ax.set_title(f"Output Channel {j}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

    print(f"{block_name} visualization complete!")


def visualize_block_output_rgb(block, input_tensor, block_name, num_channels=4):
    print(f"\nVisualizing {block_name}")
    x = input_tensor
    
    # Forward pass
    with torch.no_grad():
        output = block(x)
    
    # If the output is a tuple or list (e.g., for blocks that return multiple tensors)
    if isinstance(output, (tuple, list)):
        output = output[0]  # Visualize only the first tensor
    
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")

    # Visualize input and output
    fig, axs = plt.subplots(2, min(num_channels, max(x.shape[1], output.shape[1])), figsize=(20, 10))
    
    def visualize_channels(tensor, row):
        for j in range(min(num_channels, tensor.shape[1])):
            ax = axs[row, j]
            channel = tensor[0, j].detach().cpu().numpy()
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            # Create RGB image with the channel value in the corresponding color channel
            rgb_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
            if j < 3:
                rgb_channel[:, :, j] = channel
            else:
                # For channels beyond the first 3, use grayscale
                rgb_channel[:, :, :] = channel[:, :, np.newaxis]
            
            ax.imshow(rgb_channel)
            ax.set_title(f"{'Input' if row == 0 else 'Output'} Channel {j}")
            ax.axis('off')
    
    # Display input channels
    visualize_channels(x, 0)
    
    # Display output channels
    visualize_channels(output, 1)
    
    plt.tight_layout()
    plt.show()

    print(f"{block_name} visualization complete!")

def visualize_latent_token(token, save_path):
    """
    Visualize a 1D latent token as a colorful bar.
    
    Args:
    token (torch.Tensor): A 1D tensor representing the latent token.
    save_path (str): Path to save the visualization.
    """
    # Ensure the token is on CPU and convert to numpy
    token_np = token.cpu().detach().numpy()
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 0.5))
    
    # Normalize the token values to [0, 1] for colormap
    token_normalized = (token_np - token_np.min()) / (token_np.max() - token_np.min())
    
    # Create a colorful representation
    cmap = plt.get_cmap('viridis')
    colors = cmap(token_normalized)
    
    # Plot the token as a colorful bar
    ax.imshow(colors.reshape(1, -1, 4), aspect='auto')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add a title
    plt.title(f"Latent Token (dim={len(token_np)})")
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

if __name__ == "__main__":


       # Load the image
    image_path = "/media/oem/12TB/Downloads/CelebV-HQ/celebvhq/35666/0H5fm71cs4A_11/000000.png"
    image_tensor = load_and_preprocess_image(image_path)

    # Create a ResBlock
    resblock = ResBlock(3, 64, downsample=True)

    # Visualize the ResBlock
    visualize_resblock_rgb(resblock, image_tensor)


    upconv = UpConvResBlock(64, 128)
    visualize_block_output(upconv, image_tensor, "UpConvResBlock")

    # Run all tests with the image tensor
    # upconv = UpConvResBlock(3, 64)
    # test_upconvresblock(upconv, image_tensor)
    # visualize_feature_maps(upconv, image_tensor)

    downconv = DownConvResBlock(3, 64)
    test_downconvresblock(downconv, image_tensor)
    visualize_feature_maps(downconv, image_tensor)

    featres = FeatResBlock(3)
    test_featresblock(featres, image_tensor)
    visualize_feature_maps(featres, image_tensor)

    modconv = ModulatedConv2d(3, 64, 3)
    style = torch.randn(image_tensor.shape[0], 3)
    test_modulatedconv2d(modconv, image_tensor, style)


    # Test with dropout
    upconv = UpConvResBlock(3, 64)
    test_block_with_dropout(upconv, image_tensor, "UpConvResBlock")

    downconv = DownConvResBlock(3, 64)
    test_block_with_dropout(downconv, image_tensor, "DownConvResBlock")

    featres = FeatResBlock(3)
    test_block_with_dropout(featres, image_tensor, "FeatResBlock")

    resblock = ResBlock(3, 64)
    test_block_with_dropout(resblock, image_tensor, "ResBlock")

    # Test ResBlock with and without downsampling
    resblock = ResBlock(3, 64, downsample=False)
    test_resblock_with_image(resblock, image_tensor)

    resblock_down = ResBlock(3, 64, downsample=True)
    test_resblock_with_image(resblock_down, image_tensor)



# BROKEN
    # styledconv = StyledConv(3, 64, 3, 32, upsample=True)
    # latent = torch.randn(image_tensor.shape[0], 32)
    # test_styledconv(styledconv, image_tensor, latent)
    # visualize_feature_maps(styledconv, image_tensor, num_channels=4, latent_dim=32)
    

#     upconv = UpConvResBlock(64, 128)
# test_input = torch.randn(1, 64, 32, 32)
# visualize_block_output(upconv, test_input, "UpConvResBlock")

# downconv = DownConvResBlock(128, 64)
# test_input = torch.randn(1, 128, 32, 32)
# visualize_block_output(downconv, test_input, "DownConvResBlock")

# featres = FeatResBlock(64)
# test_input = torch.randn(1, 64, 32, 32)
# visualize_block_output(featres, test_input, "FeatResBlock")

# styledconv = StyledConv(64, 128, 3, 32)
# test_input = torch.randn(1, 64, 32, 32)
# latent = torch.randn(1, 32)
# visualize_block_output(lambda x: styledconv(x, latent), test_input, "StyledConv")