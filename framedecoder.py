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
        x = x.matmul(self.B)
        return torch.sin(x)

class ModulatedFC(nn.Module):
    def __init__(self, in_features, out_features, style_dim):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.modulation = nn.Linear(style_dim, in_features)
        
    def forward(self, x, style):
        style = self.modulation(style).unsqueeze(1)
        x = self.fc(x * style)
        return x

class CIPSFrameDecoder(nn.Module):
    def __init__(self, feature_dims, ngf=64, max_resolution=256, style_dim=512, num_layers=8):
        super(CIPSFrameDecoder, self).__init__()
        
        self.feature_dims = feature_dims
        self.ngf = ngf
        self.max_resolution = max_resolution
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        self.feature_projection = nn.ModuleList([
            nn.Linear(dim, style_dim) for dim in feature_dims
        ])
        
        self.fourier_features = FourierFeatures(2, 256)
        self.coord_embeddings = nn.Parameter(torch.randn(1, 512, max_resolution, max_resolution))
        
        self.layers = nn.ModuleList()
        self.to_rgb = nn.ModuleList()
        
        current_dim = 512 + 256  # Fourier features + coordinate embeddings
        
        for i in range(num_layers):
            self.layers.append(ModulatedFC(current_dim, ngf * 8, style_dim))
            if i % 2 == 0 or i == num_layers - 1:
                self.to_rgb.append(ModulatedFC(ngf * 8, 3, style_dim))
            current_dim = ngf * 8
        
    def get_coord_grid(self, batch_size, resolution):
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        x, y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack((x, y), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        return coords.to(next(self.parameters()).device)
    
    def forward(self, features):
        print(f"Input features shapes: {[f.shape for f in features]}")
        
        batch_size = features[0].shape[0]
        target_resolution = self.max_resolution
        print(f"Batch size: {batch_size}, Target resolution: {target_resolution}")
        
        # Project input features to style vectors
        styles = []
        for proj, feat in zip(self.feature_projection, features):
            print(f"Feature shape before view: {feat.shape}")
            feat_flat = feat.reshape(batch_size, -1)
            print(f"Feature shape after reshape: {feat_flat.shape}")
            style = proj(feat_flat)
            print(f"Style shape after projection: {style.shape}")
            styles.append(style)
        
        w = torch.cat(styles, dim=-1)
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