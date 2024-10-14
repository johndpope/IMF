import unittest
import torch
import tensorflow as tf
import numpy as np
from model import (DenseFeatureEncoder as PyTorchDenseFeatureEncoder,
                   LatentTokenEncoder as PyTorchLatentTokenEncoder,
                   LatentTokenDecoder as PyTorchLatentTokenDecoder,
                   FrameDecoder as PyTorchFrameDecoder,
                   DownConvResBlock as PyTorchDownConvResBlock,
                   UpConvResBlock as PyTorchUpConvResBlock,
                   FeatResBlock as PyTorchFeatResBlock,
                   StyledConv as PyTorchStyledConv)
from tf.model import (DenseFeatureEncoder as TFDenseFeatureEncoder,
                      LatentTokenEncoder as TFLatentTokenEncoder,
                      LatentTokenDecoder as TFLatentTokenDecoder,
                      FrameDecoder as TFFrameDecoder,
                      DownConvResBlock as TFDownConvResBlock,
                      UpConvResBlock as TFUpConvResBlock,
                      FeatResBlock as TFFeatResBlock,
                      StyledConv as TFStyledConv)

def debug_print(framework, message, is_pass=None):
    prefix = f"[{framework.upper()}] "
    if is_pass is not None:
        prefix += "✅ " if is_pass else "❌ "
    print(prefix + message)

class TestIMFComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.input_channels = 3
        self.height = 256
        self.width = 256
        self.latent_dim = 32
        self.epsilon = 1e-5  # Tolerance for floating-point comparisons

    def test_dense_feature_encoder(self):
        pytorch_model = PyTorchDenseFeatureEncoder()
        tf_model = TFDenseFeatureEncoder()

        input_tensor = np.random.rand(self.batch_size, self.input_channels, self.height, self.width).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

        debug_print("pytorch", f"DenseFeatureEncoder input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"DenseFeatureEncoder output shapes: {[f.shape for f in pytorch_output]}")

        debug_print("tf", f"DenseFeatureEncoder input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"DenseFeatureEncoder output shapes: {[f.shape for f in tf_output]}")

        all_match = True
        for i, (pytorch_feat, tf_feat) in enumerate(zip(pytorch_output, tf_output)):
            tf_feat_nhwc = tf.transpose(tf_feat, [0, 3, 1, 2])
            try:
                np.testing.assert_allclose(pytorch_feat.detach().numpy(), tf_feat_nhwc.numpy(), rtol=self.epsilon, atol=self.epsilon)
                debug_print("test", f"DenseFeatureEncoder output {i} matches within epsilon", True)
            except AssertionError:
                debug_print("test", f"DenseFeatureEncoder output {i} does not match within epsilon", False)
                all_match = False
        
        self.assertTrue(all_match, "Not all DenseFeatureEncoder outputs match within epsilon")

    def test_latent_token_encoder(self):
        pytorch_model = PyTorchLatentTokenEncoder()
        tf_model = TFLatentTokenEncoder()

        input_tensor = np.random.rand(self.batch_size, self.input_channels, self.height, self.width).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

        debug_print("pytorch", f"LatentTokenEncoder input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"LatentTokenEncoder output shape: {pytorch_output.shape}")

        debug_print("tf", f"LatentTokenEncoder input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"LatentTokenEncoder output shape: {tf_output.shape}")

        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "LatentTokenEncoder outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "LatentTokenEncoder outputs do not match within epsilon", False)
            raise

    def test_latent_token_decoder(self):
        pytorch_model = PyTorchLatentTokenDecoder()
        tf_model = TFLatentTokenDecoder()

        latent_tensor = np.random.rand(self.batch_size, self.latent_dim).astype(np.float32)
        pytorch_input = torch.from_numpy(latent_tensor)
        tf_input = tf.convert_to_tensor(latent_tensor)

        debug_print("pytorch", f"LatentTokenDecoder input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"LatentTokenDecoder output shapes: {[f.shape for f in pytorch_output]}")

        debug_print("tf", f"LatentTokenDecoder input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"LatentTokenDecoder output shapes: {[f.shape for f in tf_output]}")

        all_match = True
        for i, (pytorch_feat, tf_feat) in enumerate(zip(pytorch_output, tf_output)):
            tf_feat_nchw = tf.transpose(tf_feat, [0, 3, 1, 2])
            try:
                np.testing.assert_allclose(pytorch_feat.detach().numpy(), tf_feat_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
                debug_print("test", f"LatentTokenDecoder output {i} matches within epsilon", True)
            except AssertionError:
                debug_print("test", f"LatentTokenDecoder output {i} does not match within epsilon", False)
                all_match = False
        
        self.assertTrue(all_match, "Not all LatentTokenDecoder outputs match within epsilon")

    def test_frame_decoder(self):
        pytorch_model = PyTorchFrameDecoder()
        tf_model = TFFrameDecoder()

        feature_shapes = [(self.batch_size, 128, 64, 64),
                          (self.batch_size, 256, 32, 32),
                          (self.batch_size, 512, 16, 16),
                          (self.batch_size, 512, 8, 8)]
        
        pytorch_features = [torch.rand(shape) for shape in feature_shapes]
        tf_features = [tf.transpose(tf.convert_to_tensor(feat.numpy()), [0, 2, 3, 1]) for feat in pytorch_features]

        debug_print("pytorch", f"FrameDecoder input shapes: {[f.shape for f in pytorch_features]}")
        pytorch_output = pytorch_model(pytorch_features)
        debug_print("pytorch", f"FrameDecoder output shape: {pytorch_output.shape}")

        debug_print("tf", f"FrameDecoder input shapes: {[f.shape for f in tf_features]}")
        tf_output = tf_model(tf_features)
        debug_print("tf", f"FrameDecoder output shape: {tf_output.shape}")

        tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "FrameDecoder outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "FrameDecoder outputs do not match within epsilon", False)
            raise


    def test_down_conv_res_block(self):
        in_channels = 64
        out_channels = 128
        pytorch_model = PyTorchDownConvResBlock(in_channels, out_channels)
        tf_model = TFDownConvResBlock(in_channels, out_channels)

        input_tensor = np.random.rand(self.batch_size, in_channels, self.height, self.width).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

        debug_print("pytorch", f"DownConvResBlock input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"DownConvResBlock output shape: {pytorch_output.shape}")

        debug_print("tf", f"DownConvResBlock input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"DownConvResBlock output shape: {tf_output.shape}")

        tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "DownConvResBlock outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "DownConvResBlock outputs do not match within epsilon", False)
            raise

    def test_up_conv_res_block(self):
        in_channels = 128
        out_channels = 64
        pytorch_model = PyTorchUpConvResBlock(in_channels, out_channels)
        tf_model = TFUpConvResBlock(in_channels, out_channels)

        input_tensor = np.random.rand(self.batch_size, in_channels, self.height // 2, self.width // 2).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

        debug_print("pytorch", f"UpConvResBlock input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"UpConvResBlock output shape: {pytorch_output.shape}")

        debug_print("tf", f"UpConvResBlock input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"UpConvResBlock output shape: {tf_output.shape}")

        tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "UpConvResBlock outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "UpConvResBlock outputs do not match within epsilon", False)
            raise

    def test_feat_res_block(self):
        channels = 64
        pytorch_model = PyTorchFeatResBlock(channels)
        tf_model = TFFeatResBlock(channels)

        input_tensor = np.random.rand(self.batch_size, channels, self.height, self.width).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

        debug_print("pytorch", f"FeatResBlock input shape: {pytorch_input.shape}")
        pytorch_output = pytorch_model(pytorch_input)
        debug_print("pytorch", f"FeatResBlock output shape: {pytorch_output.shape}")

        debug_print("tf", f"FeatResBlock input shape: {tf_input.shape}")
        tf_output = tf_model(tf_input)
        debug_print("tf", f"FeatResBlock output shape: {tf_output.shape}")

        tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "FeatResBlock outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "FeatResBlock outputs do not match within epsilon", False)
            raise

    def test_styled_conv(self):
        in_channel = 64
        out_channel = 128
        kernel_size = 3
        style_dim = 32
        pytorch_model = PyTorchStyledConv(in_channel, out_channel, kernel_size, style_dim)
        tf_model = TFStyledConv(in_channel, out_channel, kernel_size, style_dim)

        input_tensor = np.random.rand(self.batch_size, in_channel, self.height, self.width).astype(np.float32)
        style_tensor = np.random.rand(self.batch_size, style_dim).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        pytorch_style = torch.from_numpy(style_tensor)
        tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))
        tf_style = tf.convert_to_tensor(style_tensor)

        debug_print("pytorch", f"StyledConv input shape: {pytorch_input.shape}, style shape: {pytorch_style.shape}")
        pytorch_output = pytorch_model(pytorch_input, pytorch_style)
        debug_print("pytorch", f"StyledConv output shape: {pytorch_output.shape}")

        debug_print("tf", f"StyledConv input shape: {tf_input.shape}, style shape: {tf_style.shape}")
        tf_output = tf_model(tf_input, tf_style)
        debug_print("tf", f"StyledConv output shape: {tf_output.shape}")

        tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
        try:
            np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
            debug_print("test", "StyledConv outputs match within epsilon", True)
        except AssertionError:
            debug_print("test", "StyledConv outputs do not match within epsilon", False)
            raise

if __name__ == '__main__':
    unittest.main()