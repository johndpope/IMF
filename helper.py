import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import numpy as np
import wandb
import os
from torchvision.utils import save_image
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os
import io


def log_grad_flow(named_parameters, step=1):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    
    # Create the matplotlib figure
    plt.figure(figsize=(12, 6))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads), linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Create a wandb.Image from the buffer
    img = wandb.Image(buf)
    
    # Close the matplotlib figure to free up memory
    plt.close()

    # Create a wandb.Table with the data
    data = [[layer, grad] for layer, grad in zip(layers, ave_grads)]
    table = wandb.Table(data=data, columns=["layer", "average_gradient"])
    
    # Log the image and table
    wandb.log({
        "gradient_flow_plot": img,
        "gradient_flow_data": table,
        "max_gradient": np.max(ave_grads),
        "min_gradient": np.min(ave_grads),
        "mean_gradient": np.mean(ave_grads),
        "median_gradient": np.median(ave_grads)
    }, step=step)

def count_model_params(model, trainable_only=False, verbose=False):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
    model (nn.Module): The PyTorch model to analyze.
    trainable_only (bool): If True, count only trainable parameters. Default is False.
    verbose (bool): If True, print detailed breakdown of parameters. Default is False.
    
    Returns:
    float: Total number of (trainable) parameters in millions.
    dict: Breakdown of parameters by layer type.
    """
    total_params = 0
    trainable_params = 0
    param_counts = defaultdict(int)
    
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
            
            # Count parameters for each layer type
            layer_type = module.__class__.__name__
            param_counts[layer_type] += param.numel()
    
    if verbose:
        print(f"{'Layer Type':<20} {'Parameter Count':<15} {'% of Total':<10}")
        print("-" * 45)
        for layer_type, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_params * 100
            print(f"{layer_type:<20} {count:<15,d} {percentage:.2f}%")
        print("-" * 45)
        print(f"{'Total':<20} {total_params:<15,d} 100.00%")
        print(f"{'Trainable':<20} {trainable_params:<15,d} {trainable_params/total_params*100:.2f}%")
    
    if trainable_only:
        return trainable_params / 1e6, dict(param_counts)
    else:
        return total_params / 1e6, dict(param_counts)

def normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def sample_recon(model, data, accelerator, output_path, num_samples=1):
    model.eval()
    with torch.no_grad():
        try:
            x_reconstructed,x_current, x_reference = data
            batch_size = x_reconstructed.size(0)
            num_samples = min(num_samples, batch_size)
            
            # Select a subset of images if batch_size > num_samples
            x_reconstructed = x_reconstructed[:num_samples]
            x_reference = x_reference[:num_samples]
            x_current = x_current[:num_samples]
            
            # Prepare frames for saving (2 rows: clamped reconstructed and original reference)
            frames = torch.cat((x_reconstructed,x_current, x_reference), dim=0)
            
            # Ensure we have a valid output directory
            if output_path:
                output_dir = os.path.dirname(output_path)
                if not output_dir:
                    output_dir = '.'
                os.makedirs(output_dir, exist_ok=True)
                
                # Save frames as a grid (2 rows, num_samples columns)
                save_image(accelerator.gather(frames), output_path, nrow=num_samples, padding=2, normalize=False)
                print(f"Saved sample reconstructions to {output_path}")
            else:
                accelerator.print("Warning: No output path provided. Skipping image save.")
            
            # Log images to wandb
            wandb_images = []
            for i in range(num_samples):
                wandb_images.extend([
                    wandb.Image(x_reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reconstructed {i}"),
                    wandb.Image(x_current[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_current {i}"),
                    wandb.Image(x_reference[i].cpu().detach().numpy().transpose(1, 2, 0), caption=f"x_reference {i}")
                ])
            
            wandb.log({"Sample Reconstructions": wandb_images})
            
            return frames
        except RuntimeError as e:
            print(f"🔥 e:{e}")
        return None
                       

def monitor_gradients(model, epoch, batch_idx, log_interval=10):
    """
    Monitor gradients of the model parameters.
    
    :param model: The neural network model
    :param epoch: Current epoch number
    :param batch_idx: Current batch index
    :param log_interval: How often to log gradient statistics
    """
    if batch_idx % log_interval == 0:
        grad_stats = defaultdict(list)
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats['norm'].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")
                
                if torch.isinf(param.grad).any():
                    print(f"Inf gradient detected in {name}")
                
                grad_stats['names'].append(name)
        
        if grad_stats['norm']:
            avg_norm = np.mean(grad_stats['norm'])
            max_norm = np.max(grad_stats['norm'])
            min_norm = np.min(grad_stats['norm'])
            
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(f"Gradient norms - Avg: {avg_norm:.4f}, Max: {max_norm:.4f}, Min: {min_norm:.4f}")
            
            # Identify layers with unusually high or low gradients
            threshold_high = avg_norm * 10  # Adjust this multiplier as needed
            threshold_low = avg_norm * 0.1  # Adjust this multiplier as needed
            
            for name, norm in zip(grad_stats['names'], grad_stats['norm']):
                if norm > threshold_high:
                    print(f"High gradient in {name}: {norm:.4f}")
                elif norm < threshold_low:
                    print(f"Low gradient in {name}: {norm:.4f}")
        else:
            print("No gradients to monitor")



# helper for gradient vanishing / explosion
def hook_fn(name):
    def hook(grad):
        if torch.isnan(grad).any():
            # print(f"🔥 NaN gradient detected in {name}")
            return torch.zeros_like(grad)  # Replace NaN with zero
        elif torch.isinf(grad).any():
            # print(f"🔥 Inf gradient detected in {name}")
            return torch.clamp(grad, -1e6, 1e6)  # Clamp infinite values
        #else:
            # You can add more conditions or logging here
         #  grad_norm = grad.norm().item()
         #   print(f"Gradient norm for {name}: {grad_norm}")
        return grad
    return hook

def add_gradient_hooks(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(hook_fn(name))



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