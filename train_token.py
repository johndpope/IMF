import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf
import wandb
from tqdm.auto import tqdm
from model import TokenManipulationNetwork,IMFModel
from models.methods import FSEFull,FSEInverter
from configs.paths import DefaultPaths
from VideoDataset import VideoDataset,gpu_padded_collate
from torchvision import transforms


# class StyleFeatureEditorWrapper(nn.Module):
#     def __init__(self, sfe_model, token_manipulation_network):
#         super().__init__()
#         self.sfe_model = sfe_model
#         self.token_manipulation_network = token_manipulation_network

#     def forward(self, x_current, x_reference, editing_condition):
#         # SFE inversion
#         w, F_k = self.sfe_model.inverter(x_current)
        
#         # Token manipulation
#         edited_w = self.token_manipulation_network(w, editing_condition)
        
#         # SFE editing
#         delta = self.sfe_model.get_delta(w, edited_w)
#         F_k_edited = self.sfe_model.feature_editor(F_k, delta)
        
#         # SFE reconstruction
#         x_edited = self.sfe_model.frame_decoder(F_k_edited, edited_w[self.sfe_model.k+1:])
        
#         return x_edited, w, edited_w, F_k, F_k_edited

class StyleFeatureEditor(nn.Module):
    def __init__(self, device="cuda:0", paths=DefaultPaths, checkpoint_path=None, inverter_pth=None):
        super(StyleFeatureEditor, self).__init__()
        self.fse_full = FSEFull(device, paths, checkpoint_path, inverter_pth)
        self.fse_inverter = FSEInverter(device, paths, checkpoint_path)

    def forward(self, x, return_latents=False, n_iter=1e5):
        # Use FSEFull for editing
        images, w_recon, fused_feat, predicted_feat = self.fse_full(x, return_latents=True, n_iter=n_iter)
        
        # Use FSEInverter for inversion
        inverted_images, w_recon_inv, fused_feat_inv, w_feat_inv = self.fse_inverter(x, return_latents=True, n_iter=n_iter)
        
        if return_latents:
            return images, w_recon, fused_feat, predicted_feat, inverted_images, w_recon_inv, fused_feat_inv, w_feat_inv
        return images, inverted_images

    def get_delta(self, w, edited_w):
        # Compute delta based on the difference between w and edited_w
        return edited_w - w

class CombinedModel(nn.Module):
    def __init__(self, sfe_model, token_manipulation_network):
        super().__init__()
        self.sfe_model = sfe_model
        self.token_manipulation_network = token_manipulation_network

    def forward(self, x, editing_condition):
        # SFE inversion and editing
        images, w_recon, fused_feat, predicted_feat, inverted_images, w_recon_inv, fused_feat_inv, w_feat_inv = self.sfe_model(x, return_latents=True)
        
        # Token manipulation
        edited_w = self.token_manipulation_network(w_recon, editing_condition)
        
        # SFE editing with manipulated token
        delta = self.sfe_model.get_delta(w_recon, edited_w)
        edited_feat = self.sfe_model.fse_full.encoder(torch.cat([fused_feat, delta], dim=1))
        feats = [None] * 9 + [edited_feat] + [None] * (17 - 9)
        
        edited_images, _ = self.sfe_model.fse_full.decoder(
            [edited_w],
            input_is_latent=True,
            return_features=True,
            new_features=feats,
            feature_scale=1.0,
            is_stylespace=False,
            randomize_noise=False
        )
        
        return edited_images, images, inverted_images, w_recon, edited_w, fused_feat, fused_feat_inv



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
    imf_checkpoint = torch.load(config.training.checkpoint_path, map_location='cpu')
    imf_model.load_state_dict(imf_checkpoint['model_state_dict'])
    imf_model.eval()

    # Create the TokenManipulationNetwork
    sfe_model = StyleFeatureEditor(accelerator.device, DefaultPaths, config.checkpoint_path, config.inverter_pth)
    token_manipulation_network = TokenManipulationNetwork(
        token_dim=512,  # Assuming w_recon has 512 dimensions
        condition_dim=config.condition_dim,
        hidden_dim=config.hidden_dim
    )


    combined_model = CombinedModel(sfe_model, token_manipulation_network)
    

    # Set up optimizer and dataset
    optimizer = optim.Adam(token_manipulation_network.parameters(), lr=config.token_manipulation.learning_rate)
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VideoDataset(config.dataset.root_dir, 
                                transform=transform, 
                                frame_skip=0, 
                                num_frames=300)

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

