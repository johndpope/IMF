import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierFeatures(nn.Module):
    def __init__(self, input_dim, mapping_size=256, scale=10):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale, requires_grad=False)
    
    def forward(self, x):
        print(f"FourierFeatures input shape: {x.shape}")
        x = x.matmul(self.B)
        print(f"FourierFeatures output shape: {x.shape}")
        return torch.sin(x)

class ModulatedFC(nn.Module):
    def __init__(self, in_features, out_features, style_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, 512),
            nn.ReLU(),
            nn.Linear(512, in_features)
        )
        self.scale_mlp = nn.Sequential(
            nn.Linear(style_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_features)
        )
        
    def forward(self, x, style):
        print(f"ModulatedFC input shapes - x: {x.shape}, style: {style.shape}")
        
        batch_size, num_pixels, in_features = x.shape
        
        style = self.style_mlp(style).view(batch_size, in_features, 1)
        scale = self.scale_mlp(style).view(batch_size, 1, -1)
        
        print(f"ModulatedFC processed shapes - style: {style.shape}, scale: {scale.shape}")
        
        x = x.transpose(1, 2)  # [batch_size, in_features, num_pixels]
        x = x * style
        x = self.fc(x.transpose(1, 2))  # [batch_size, num_pixels, out_features]
        x = x * scale
        
        print(f"ModulatedFC output shape: {x.shape}")
        return x
class CIPSFrameDecoder(nn.Module):
    def __init__(self, feature_dims=[128, 256, 512, 512], ngf=64, max_resolution=256, style_dim=512, num_layers=8):
        super(CIPSFrameDecoder, self).__init__()
        
        self.feature_dims = feature_dims
        self.ngf = ngf
        self.max_resolution = max_resolution
        self.style_dim = style_dim * len(feature_dims)  # Adjusted style_dim
        self.num_layers = num_layers
        
        self.feature_projection = nn.ModuleList([
            nn.Conv2d(dim, style_dim, kernel_size=1) for dim in feature_dims
        ])
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_dim = 512 + 256  # Fourier features + coordinate embeddings
        
        for i in range(num_layers):
            self.layers.append(ModulatedFC(current_dim, ngf * 8, self.style_dim))
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(ngf * 8, 3, self.style_dim))
            current_dim = ngf * 8 
        
    def get_coord_grid(self, batch_size, resolution):
        print(f"get_coord_grid input - batch_size: {batch_size}, resolution: {resolution}")
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        print(f"get_coord_grid output shape: {coords.shape}")
        return coords.to(next(self.parameters()).device)
    
    def forward(self, features):
        print(f"Input features shapes: {[f.shape for f in features]}")
        
        batch_size = features[0].shape[0]
        target_resolution = self.max_resolution
        print(f"Batch size: {batch_size}, Target resolution: {target_resolution}")
        
        # Project input features to style vectors
        styles = []
        for i, (proj, feat) in enumerate(zip(self.feature_projection, features)):
            print(f"Feature {i} shape before projection: {feat.shape}")
            style = proj(feat)
            style = style.view(batch_size, -1, style.size(2) * style.size(3)).mean(dim=2)
            print(f"Style {i} shape after projection: {style.shape}")
            styles.append(style)
        
        w = torch.cat(styles, dim=1)
        print(f"Combined style vector shape: {w.shape}")
        
        # Generate coordinate grid
        coords = self.get_coord_grid(batch_size, target_resolution)
        print(f"Coordinate grid shape: {coords.shape}")
        coords_flat = coords.view(batch_size, -1, 2)
        print(f"Flattened coordinate grid shape: {coords_flat.shape}")
        
        # Get Fourier features and coordinate embeddings
        fourier_features = self.fourier_features(coords_flat)
        print(f"Fourier features shape: {fourier_features.shape}")
        
        coord_embeddings = F.grid_sample(
            self.coord_embeddings.expand(batch_size, -1, -1, -1),
            coords,
            mode='bilinear',
            align_corners=True
        )
        print(f"Coordinate embeddings shape after grid_sample: {coord_embeddings.shape}")
        coord_embeddings = coord_embeddings.permute(0, 2, 3, 1).reshape(batch_size, -1, 512)
        print(f"Coordinate embeddings shape after reshape: {coord_embeddings.shape}")
        
        # Concatenate Fourier features and coordinate embeddings
        features = torch.cat([fourier_features, coord_embeddings], dim=-1)
        print(f"Combined features shape: {features.shape}")
        
        rgb = 0
        for i, (layer, to_rgb) in enumerate(zip(self.layers, self.to_rgb)):
            features = layer(features, w)
            print(f"Features shape after layer {i}: {features.shape}")
            features = F.leaky_relu(features, 0.2)
            
            if i % 2 == 0 or i == self.num_layers - 1:
                rgb_out = to_rgb(features, w)
                print(f"RGB output shape at layer {i}: {rgb_out.shape}")
                rgb = rgb + rgb_out
        
        output = torch.sigmoid(rgb).view(batch_size, target_resolution, target_resolution, 3).permute(0, 3, 1, 2)
        print(f"Final output shape: {output.shape}")
        
        # Ensure output is in [-1, 1] range
        output = (output * 2) - 1
        
        return output