import unittest
import torch
import torch.nn as nn
import sys
import os

# Add the directory containing the module to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vit import PositionalEncoding,  TransformerBlock, ImplicitMotionAlignment, CrossAttentionModule
from model import LatentTokenEncoder, LatentTokenDecoder, DenseFeatureEncoder, FrameDecoder

class TestNeuralNetworkComponents(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.B, self.C_f, self.C_m, self.H, self.W = 2, 256, 256, 64, 64
        self.feature_dim = self.C_f
        self.motion_dim = self.C_m
        self.heads = 8
        self.dim_head = 64
        self.mlp_dim = 1024
        self.dm = 32
        self.input_size = 256
        self.input_channels = 3

    def test_positional_encoding(self):
        pe = PositionalEncoding(d_model=self.motion_dim).to(self.device)
        x = torch.randn(100, 1, self.motion_dim).to(self.device)
        output = pe(x)
        self.assertEqual(output.shape, (100, 1, self.motion_dim))




    def test_latent_token_encoder(self):
        et = LatentTokenEncoder(dm=self.dm).to(self.device)
        x = torch.randn(self.B, self.input_channels, self.input_size, self.input_size).to(self.device)
        output = et(x)
        
        self.assertEqual(len(output.shape), 2, f"ET output should be a 2D tensor (batch_size, dm), but got shape {output.shape}")
        self.assertEqual(output.shape[0], self.B, f"First dimension should be batch_size ({self.B}), but got {output.shape[0]}")
        self.assertTrue(32 <= output.shape[1] <= 1024, f"dm should be between 32 and 1024, got {output.shape[1]}")

    def test_dense_feature_encoder(self):
        ef = DenseFeatureEncoder().to(self.device)
        x = torch.randn(self.B, self.input_channels, self.input_size, self.input_size).to(self.device)
        outputs = ef(x)
        
        self.assertEqual(len(outputs), 4, f"EF should produce 4 outputs, but got {len(outputs)}")
        
        expected_shapes = [
            (self.B, 128, 64, 64),
            (self.B, 256, 32, 32),
            (self.B, 512, 16, 16),
            (self.B, 512, 8, 8)
        ]
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            self.assertEqual(output.shape, expected_shape, f"fr{i+1} should be {expected_shape}, but got {output.shape}")

    def test_latent_token_decoder(self):
        et = LatentTokenEncoder(dm=self.dm).to(self.device)
        latent_token_decoder = LatentTokenDecoder(latent_dim=self.dm).to(self.device)
        
        x_current = torch.randn(self.B, self.input_channels, self.input_size, self.input_size).to(self.device)
        x_ref = torch.randn(self.B, self.input_channels, self.input_size, self.input_size).to(self.device)
        
        t_c = et(x_current)
        t_r = et(x_ref)
        
        m_r = latent_token_decoder(t_r)
        m_c = latent_token_decoder(t_c)
        
        self.assertEqual(len(m_r), len(m_c), "Number of outputs from LatentTokenDecoder should be the same for reference and current")
        
        for m_r_x, m_c_x in zip(m_r, m_c):
            self.assertEqual(m_r_x.shape, m_c_x.shape, "Shapes of reference and current outputs should match")


if __name__ == '__main__':
    unittest.main()