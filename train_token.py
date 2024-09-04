import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf
import wandb
from tqdm.auto import tqdm
from model import TokenManipulationNetwork


class StyleFeatureEditorWrapper(nn.Module):
    def __init__(self, sfe_model, token_manipulation_network):
        super().__init__()
        self.sfe_model = sfe_model
        self.token_manipulation_network = token_manipulation_network

    def forward(self, x_current, x_reference, editing_condition):
        # SFE inversion
        w, F_k = self.sfe_model.inverter(x_current)
        
        # Token manipulation
        edited_w = self.token_manipulation_network(w, editing_condition)
        
        # SFE editing
        delta = self.sfe_model.get_delta(w, edited_w)
        F_k_edited = self.sfe_model.feature_editor(F_k, delta)
        
        # SFE reconstruction
        x_edited = self.sfe_model.frame_decoder(F_k_edited, edited_w[self.sfe_model.k+1:])
        
        return x_edited, w, edited_w, F_k, F_k_edited


def train_token_manipulation_network(config):
    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        cpu=config.accelerator.cpu
    )

    # Load the main IMF model
    imf_model = IMFModel(
        latent_dim=config.model.latent_dim,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        use_resnet_feature=config.model.use_resnet_feature
    )
    imf_checkpoint = torch.load(config.imf_model_path, map_location='cpu')
    imf_model.load_state_dict(imf_checkpoint['model_state_dict'])
    imf_model.eval()

    # Create the TokenManipulationNetwork
    token_manipulation_network = TokenManipulationNetwork(
        token_dim=config.token_manipulation.token_dim,
        condition_dim=config.token_manipulation.condition_dim,
        hidden_dim=config.token_manipulation.hidden_dim
    )

    sfe_model = StyleFeatureEditor(config.sfe)
    sfe_model.load_state_dict(torch.load(config.sfe_checkpoint_path))
    sfe_model.eval()  # Set SFE to evaluation mode


    combined_model = StyleFeatureEditorWrapper(sfe_model, token_manipulation_network)

    # Set up optimizer and dataset
    optimizer = optim.Adam(token_manipulation_network.parameters(), lr=config.token_manipulation.learning_rate)
    dataset = YourDataset(config.dataset.root_dir)  # Replace with your actual dataset
    dataloader = DataLoader(dataset, batch_size=config.token_manipulation.batch_size, shuffle=True)

    # Prepare for distributed training
    imf_model, token_manipulation_network, optimizer, dataloader = accelerator.prepare(
        imf_model, token_manipulation_network, optimizer, dataloader
    )

    # Load checkpoint if exists
    start_epoch = 0
    if config.token_manipulation.load_checkpoint:
        checkpoint = accelerator.load(config.token_manipulation.checkpoint_path)
        token_manipulation_network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
     for epoch in range(config.token_manipulation.num_epochs):
        for batch in dataloader:
            x_current, x_reference, editing_condition = batch
            
            # Forward pass
            x_edited, w, edited_w, F_k, F_k_edited = combined_model(x_current, x_reference, editing_condition)
            
            # Compute losses
            reconstruction_loss = compute_reconstruction_loss(x_current, x_edited)
            editing_loss = compute_editing_loss(x_edited, editing_condition)
            feature_consistency_loss = compute_feature_consistency_loss(F_k, F_k_edited)
            
            total_loss = (
                config.loss_weights.reconstruction * reconstruction_loss +
                config.loss_weights.editing * editing_loss +
                config.loss_weights.feature_consistency * feature_consistency_loss
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            
            
            # # Get the original token
            # with torch.no_grad():
            #     original_token = imf_model.latent_token_encoder(x_current)
            
            # # Edit the token
            # edited_token = token_manipulation_network(original_token, editing_condition)
            
            # # Generate the edited image
            # imf_model.set_token_manipulation_network(token_manipulation_network)
            # edited_image = imf_model(x_current, x_reference, editing_condition)
            
            # # Compute loss (this will depend on your specific editing task)
            # loss = compute_editing_loss(edited_image, x_current, editing_condition)
            
            # # Backpropagate and update
            # accelerator.backward(loss)
            # optimizer.step()
            # optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        progress_bar.close()
        avg_loss = total_loss / len(dataloader)

        # Log to wandb
        if accelerator.is_main_process:
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
            })

        # Save checkpoint
        if (epoch + 1) % config.token_manipulation.save_every == 0:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(token_manipulation_network)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            accelerator.save(checkpoint, f"{config.checkpoints.dir}/token_manipulation_checkpoint_{epoch+1}.pth")

    # Final save
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(token_manipulation_network)
    torch.save(unwrapped_model.state_dict(), f"{config.checkpoints.dir}/token_manipulation_final.pth")

def main():
    config = OmegaConf.load('config.yaml')
    wandb.init(project='IMF-TokenManipulation', config=OmegaConf.to_container(config, resolve=True))
    
    train_token_manipulation_network(config)

if __name__ == "__main__":
    main()

