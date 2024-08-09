import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your model components here
from vit_mlgffn import WindowAttention, SpatialAwareSelfAttention, ChannelAwareSelfAttention, MLGFFN, HSCATB, ImplicitMotionAlignment, CrossAttentionModule

class TestMLGFFNComponents(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 1
        self.channels = 256
        self.height = 64
        self.width = 64
        self.num_heads = 8
        self.window_size = 8
        self.expansion_factor = 4

    def test_mlgffn_structure(self):
        model = MLGFFN(dim=self.channels, expansion_factor=self.expansion_factor).to(self.device)
        
        # Check if the model has the expected components
        self.assertTrue(hasattr(model, 'fc1'), "MLGFFN should have fc1 layer")
        self.assertTrue(hasattr(model, 'dwconv3x3'), "MLGFFN should have dwconv3x3 layer")
        self.assertTrue(hasattr(model, 'dwconv5x5'), "MLGFFN should have dwconv5x5 layer")
        self.assertTrue(hasattr(model, 'fc2'), "MLGFFN should have fc2 layer")
        self.assertTrue(hasattr(model, 'act'), "MLGFFN should have activation function")

        # Check the dimensions of the layers
        self.assertEqual(model.fc1.in_channels, self.channels)
        self.assertEqual(model.fc1.out_channels, self.channels * self.expansion_factor)
        self.assertEqual(model.dwconv3x3.in_channels, self.channels * self.expansion_factor // 2)
        self.assertEqual(model.dwconv3x3.out_channels, self.channels * self.expansion_factor // 2)
        self.assertEqual(model.dwconv5x5.in_channels, self.channels * self.expansion_factor // 2)
        self.assertEqual(model.dwconv5x5.out_channels, self.channels * self.expansion_factor // 2)
        self.assertEqual(model.fc2.in_channels, self.channels * self.expansion_factor * 2)
        self.assertEqual(model.fc2.out_channels, self.channels)

    def test_mlgffn_forward_pass(self):
        model = MLGFFN(dim=self.channels, expansion_factor=self.expansion_factor).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape, "MLGFFN output shape should match input shape")
        
        # Check if the output is different from the input
        self.assertFalse(torch.allclose(output, x), "MLGFFN should transform the input")
        
        # Verify the local feature extraction
        with torch.no_grad():
            fc1_output = model.fc1(x)
            fc1_output1, fc1_output2 = torch.split(fc1_output, fc1_output.shape[1] // 2, dim=1)
            local_features1 = model.act(model.dwconv3x3(fc1_output1))
            local_features2 = model.act(model.dwconv5x5(fc1_output2))
            local_features = torch.cat([local_features1, local_features2], dim=1)
            
            # Check if local features have spatial variation
            self.assertTrue((local_features.std(dim=(2, 3)) > 1e-5).any(), "Local features should have spatial variation")
        
        # Verify the global feature extraction
        with torch.no_grad():
            global_features = F.adaptive_avg_pool2d(fc1_output, (1, 1))
            global_features = global_features.expand(-1, -1, self.height, self.width)
            
            # Check if global features are constant across spatial dimensions
            self.assertTrue((global_features.std(dim=(2, 3)) < 1e-5).all(), "Global features should be constant across spatial dimensions")
        
        # Verify the combination of local and global features
        with torch.no_grad():
            combined_features = torch.cat([local_features, global_features], dim=1)
            final_output = model.fc2(combined_features)
            
            # Check if the final output has both local and global characteristics
            self.assertTrue((final_output.std(dim=(2, 3)) > 1e-5).any(), "Final output should have spatial variation from local features")
            self.assertTrue((final_output.mean(dim=(2, 3)).std() > 1e-5), "Final output should have channel-wise variation from global features")

    def test_mlgffn_gradient_flow(self):
        model = MLGFFN(dim=self.channels, expansion_factor=self.expansion_factor).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width, requires_grad=True).to(self.device)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check if gradients are computed for all parameters
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient for {name} should not be None")
            self.assertFalse(torch.allclose(param.grad, torch.zeros_like(param.grad)), f"Gradient for {name} should not be zero")

    def test_window_attention(self):
        model = WindowAttention(dim=self.channels, window_size=(self.window_size, self.window_size), num_heads=self.num_heads).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape, "WindowAttention output shape should match input shape")
        self.assertFalse(torch.allclose(output, x), "WindowAttention should transform the input")

    def test_spatial_aware_self_attention(self):
        model = SpatialAwareSelfAttention(dim=self.channels, num_heads=self.num_heads, window_size=self.window_size).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape, "SpatialAwareSelfAttention output shape should match input shape")
        self.assertFalse(torch.allclose(output, x), "SpatialAwareSelfAttention should transform the input")

    def test_channel_aware_self_attention(self):
        model = ChannelAwareSelfAttention(dim=self.channels).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape, "ChannelAwareSelfAttention output shape should match input shape")
        self.assertFalse(torch.allclose(output, x), "ChannelAwareSelfAttention should transform the input")

    def test_mlgffn(self):
        model = MLGFFN(dim=self.channels).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape, "MLGFFN output shape should match input shape")
        self.assertFalse(torch.allclose(output, x), "MLGFFN should transform the input")
        
        # Test if the output has both local and global features
        with torch.no_grad():
            # Check if there's any spatial variation (indicative of local features)
            spatial_variation = (output.std(dim=(2, 3)) > 1e-5).any()
            self.assertTrue(spatial_variation, "MLGFFN output should have spatial variation (local features)")
            
            # Check if there's any channel-wise variation (indicative of global features)
            channel_variation = (output.mean(dim=(2, 3)).std() > 1e-5)
            self.assertTrue(channel_variation, "MLGFFN output should have channel-wise variation (global features)")

        

    def test_hscatb(self):
        model = HSCATB(dim=self.channels, num_heads=self.num_heads, window_size=self.window_size).to(self.device)
        x = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(x)
        
        self.assertEqual(output.shape, x.shape, "HSCATB output shape should match input shape")
        self.assertFalse(torch.allclose(output, x), "HSCATB should transform the input")

    def test_cross_attention_module(self):
        model = CrossAttentionModule(feature_dim=self.channels, motion_dim=self.channels, num_heads=self.num_heads).to(self.device)
        ml_c = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        ml_r = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        fl_r = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(ml_c, ml_r, fl_r)
        
        self.assertEqual(output.shape, ml_c.shape, "CrossAttentionModule output shape should match input shape")
        self.assertFalse(torch.allclose(output, ml_c), "CrossAttentionModule should transform the input")

    def test_improved_implicit_motion_alignment(self):
        model = ImplicitMotionAlignment(
            feature_dim=self.channels, 
            motion_dim=self.channels, 
            depth=4, 
            num_heads=self.num_heads, 
            window_size=self.window_size
        ).to(self.device)
        ml_c = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        ml_r = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        fl_r = torch.randn(self.batch_size, self.channels, self.height, self.width).to(self.device)
        output = model(ml_c, ml_r, fl_r)
        
        self.assertEqual(output.shape, ml_c.shape, "ImplicitMotionAlignment output shape should match input shape")
        self.assertFalse(torch.allclose(output, ml_c), "ImplicitMotionAlignment should transform the input")

if __name__ == '__main__':
    unittest.main()