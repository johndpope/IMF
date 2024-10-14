import unittest
import torch
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import platform
import subprocess

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

import warnings
import logging


# Disable all TensorFlow logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to show errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable warnings
warnings.filterwarnings('ignore')

# Disable Python logging
logging.getLogger('tensorflow').disabled = True
logging.getLogger('matplotlib').disabled = True  # If you're using matplotlib

# Disable CUDA warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Disable deprecation warnings
tf.get_logger().setLevel(logging.ERROR)

# Disable NUMA warnings
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Disable CUDA device initialization messages
tf.autograph.set_verbosity(0)

# If you're using Keras
tf.keras.utils.disable_interactive_logging()

def debug_print(framework, message, is_pass=None):
    prefix = f"[{framework.upper()}] "
    if is_pass is not None:
        prefix += "✅ " if is_pass else "❌ "
    print(prefix + message)

def get_cuda_version():
    try:
        return subprocess.check_output(['nvcc', '--version']).decode('utf-8').split('release ')[-1].split(',')[0]
    except:
        return "CUDA not found"

def get_cudnn_version():
    try:
        return tf.sysconfig.get_build_info()["cudnn_version"]
    except:
        return "cuDNN version not found"

def get_tensorflow_gpu_devices():
    return tf.config.list_physical_devices('GPU')

def get_pytorch_gpu_devices():
    return torch.cuda.device_count()

print(f"Python version: {platform.python_version()}")
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {get_cuda_version()}")
print(f"cuDNN version: {get_cudnn_version()}")
print(f"TensorFlow GPU devices: {get_tensorflow_gpu_devices()}")
print(f"PyTorch GPU devices: {get_pytorch_gpu_devices()}")

if tf.test.is_built_with_cuda():
    print("TensorFlow is built with CUDA")
else:
    print("TensorFlow is not built with CUDA")

if torch.cuda.is_available():
    print("PyTorch CUDA is available")
else:
    print("PyTorch CUDA is not available")
class BaseModelTest(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-5

    def _to_numpy(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        else:
            return tensor

    def assert_tensors_close(self, pytorch_tensor, tf_tensor, msg=None):
        pytorch_np = self._to_numpy(pytorch_tensor)
        tf_np = self._to_numpy(tf_tensor)

        if tf_np.ndim == 4:
            tf_np = np.transpose(tf_np, (0, 3, 1, 2))

        np.testing.assert_allclose(pytorch_np, tf_np, rtol=self.epsilon, atol=self.epsilon, err_msg=msg)

    def compare_model_outputs(self, pytorch_model, tf_model, input_shape, input_data=None):
        if input_data is None:
            input_data = np.random.randn(*input_shape).astype(np.float32)

        pytorch_input = torch.from_numpy(input_data)
        tf_input = tf.convert_to_tensor(input_data)

        if tf_input.shape[1] == 3:  # Assume it's an image input
            tf_input = tf.transpose(tf_input, (0, 2, 3, 1))

        with torch.no_grad():
            pytorch_output = pytorch_model(pytorch_input)

        tf_output = tf_model(tf_input)

        self.assert_tensors_close(pytorch_output, tf_output, "Model outputs do not match")


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

        # Set both models to evaluation mode
        pytorch_model.eval()
        tf_model.trainable = False

        input_tensor = np.random.rand(self.batch_size, self.input_channels, self.height, self.width).astype(np.float32)
        pytorch_input = torch.from_numpy(input_tensor)
        tf_input = tf.convert_to_tensor(input_tensor)

        # PyTorch forward pass with intermediate outputs
        pytorch_intermediates = []
        x = pytorch_input
        x = pytorch_model.initial_conv(x)
        debug_print("pytorch", f"After initial conv: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
        for i, block in enumerate(pytorch_model.down_blocks):
            x = block(x)
            pytorch_intermediates.append(x)
            debug_print("pytorch", f"After down_block {i}: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
        pytorch_output = pytorch_intermediates[-4:]  # Assuming last 4 are the output

        # TensorFlow forward pass with intermediate outputs
        tf_intermediates = []
        x = tf_input
        x = tf_model.initial_conv(x)
        debug_print("tf", f"After initial conv: {x.shape}, Mean: {tf.reduce_mean(x).numpy():.6f}, Std: {tf.math.reduce_std(x).numpy():.6f}")
        for i, block in enumerate(tf_model.down_blocks):
            x = block(x)
            tf_intermediates.append(x)
            debug_print("tf", f"After down_block {i}: {x.shape}, Mean: {tf.reduce_mean(x).numpy():.6f}, Std: {tf.math.reduce_std(x).numpy():.6f}")
        tf_output = tf_intermediates[-4:]  # Assuming last 4 are the output

        debug_print("pytorch", f"DenseFeatureEncoder output shapes: {[out.shape for out in pytorch_output]}")
        debug_print("tf", f"DenseFeatureEncoder output shapes: {[out.shape for out in tf_output]}")

        # Compare shapes, sizes, and values for each output
        for i, (pytorch_out, tf_out) in enumerate(zip(pytorch_output, tf_output)):
            # Compare shapes
            self.assertEqual(pytorch_out.shape, tf_out.shape, f"Output {i} shapes do not match")
            debug_print("test", f"DenseFeatureEncoder output {i} shapes match: {pytorch_out.shape}")

            # Compare sizes
            self.assertEqual(pytorch_out.numel(), tf_out.numpy().size, f"Output {i} sizes do not match")
            debug_print("test", f"DenseFeatureEncoder output {i} sizes match: {pytorch_out.numel()}")

            # Compare values
            try:
                np.testing.assert_allclose(pytorch_out.detach().numpy(), tf_out.numpy(), rtol=self.epsilon, atol=self.epsilon)
                debug_print("test", f"DenseFeatureEncoder output {i} values match within epsilon", True)
            except AssertionError as e:
                debug_print("test", f"DenseFeatureEncoder output {i} values do not match within epsilon", False)
                print(f"Max absolute difference: {np.max(np.abs(pytorch_out.detach().numpy() - tf_out.numpy()))}")
                print(f"Max relative difference: {np.max(np.abs((pytorch_out.detach().numpy() - tf_out.numpy()) / tf_out.numpy()))}")
                raise

            # Additional check: Print mean and standard deviation of outputs
            pytorch_mean = torch.mean(pytorch_out).item()
            pytorch_std = torch.std(pytorch_out).item()
            tf_mean = tf.reduce_mean(tf_out).numpy()
            tf_std = tf.math.reduce_std(tf_out).numpy()

            debug_print("pytorch", f"DenseFeatureEncoder output {i} mean: {pytorch_mean:.6f}, std: {pytorch_std:.6f}")
            debug_print("tf", f"DenseFeatureEncoder output {i} mean: {tf_mean:.6f}, std: {tf_std:.6f}")
 

    # def test_latent_token_encoder(self):
    #     pytorch_model = PyTorchLatentTokenEncoder()
    #     tf_model = TFLatentTokenEncoder()

    #     input_tensor = np.random.rand(self.batch_size, self.input_channels, self.height, self.width).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

    #     debug_print("pytorch", f"LatentTokenEncoder input shape: {pytorch_input.shape}")
    #     pytorch_output = pytorch_model(pytorch_input)
    #     debug_print("pytorch", f"LatentTokenEncoder output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"LatentTokenEncoder input shape: {tf_input.shape}")
    #     tf_output = tf_model(tf_input)
    #     debug_print("tf", f"LatentTokenEncoder output shape: {tf_output.shape}")

    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "LatentTokenEncoder outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "LatentTokenEncoder outputs do not match within epsilon", False)
    #         raise

    # def test_latent_token_decoder(self):
    #     pytorch_model = PyTorchLatentTokenDecoder()
    #     tf_model = TFLatentTokenDecoder()

    #     latent_tensor = np.random.rand(self.batch_size, self.latent_dim).astype(np.float32)
    #     pytorch_input = torch.from_numpy(latent_tensor)
    #     tf_input = tf.convert_to_tensor(latent_tensor)

    #     debug_print("pytorch", f"LatentTokenDecoder input shape: {pytorch_input.shape}")
    #     pytorch_output = pytorch_model(pytorch_input)
    #     debug_print("pytorch", f"LatentTokenDecoder output shapes: {[f.shape for f in pytorch_output]}")

    #     debug_print("tf", f"LatentTokenDecoder input shape: {tf_input.shape}")
    #     tf_output = tf_model(tf_input)
    #     debug_print("tf", f"LatentTokenDecoder output shapes: {[f.shape for f in tf_output]}")

    #     all_match = True
    #     for i, (pytorch_feat, tf_feat) in enumerate(zip(pytorch_output, tf_output)):
    #         tf_feat_nchw = tf.transpose(tf_feat, [0, 3, 1, 2])
    #         try:
    #             np.testing.assert_allclose(pytorch_feat.detach().numpy(), tf_feat_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #             debug_print("test", f"LatentTokenDecoder output {i} matches within epsilon", True)
    #         except AssertionError:
    #             debug_print("test", f"LatentTokenDecoder output {i} does not match within epsilon", False)
    #             all_match = False
        
    #     self.assertTrue(all_match, "Not all LatentTokenDecoder outputs match within epsilon")

    # def test_frame_decoder(self):
    #     pytorch_model = PyTorchFrameDecoder()
    #     tf_model = TFFrameDecoder()

    #     feature_shapes = [(self.batch_size, 128, 64, 64),
    #                       (self.batch_size, 256, 32, 32),
    #                       (self.batch_size, 512, 16, 16),
    #                       (self.batch_size, 512, 8, 8)]
        
    #     pytorch_features = [torch.rand(shape) for shape in feature_shapes]
    #     tf_features = [tf.transpose(tf.convert_to_tensor(feat.numpy()), [0, 2, 3, 1]) for feat in pytorch_features]

    #     debug_print("pytorch", f"FrameDecoder input shapes: {[f.shape for f in pytorch_features]}")
    #     pytorch_output = pytorch_model(pytorch_features)
    #     debug_print("pytorch", f"FrameDecoder output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"FrameDecoder input shapes: {[f.shape for f in tf_features]}")
    #     tf_output = tf_model(tf_features)
    #     debug_print("tf", f"FrameDecoder output shape: {tf_output.shape}")

    #     tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "FrameDecoder outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "FrameDecoder outputs do not match within epsilon", False)
    #         raise


    # def test_down_conv_res_block(self):
    #     in_channels = 64
    #     out_channels = 128
    #     pytorch_model = PyTorchDownConvResBlock(in_channels, out_channels)
    #     tf_model = TFDownConvResBlock(in_channels, out_channels)

    #     input_tensor = np.random.rand(self.batch_size, in_channels, self.height, self.width).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

    #     debug_print("pytorch", f"DownConvResBlock input shape: {pytorch_input.shape}")
    #     pytorch_output = pytorch_model(pytorch_input)
    #     debug_print("pytorch", f"DownConvResBlock output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"DownConvResBlock input shape: {tf_input.shape}")
    #     tf_output = tf_model(tf_input)
    #     debug_print("tf", f"DownConvResBlock output shape: {tf_output.shape}")

    #     tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "DownConvResBlock outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "DownConvResBlock outputs do not match within epsilon", False)
    #         raise

    # def test_up_conv_res_block(self):
    #     in_channels = 128
    #     out_channels = 64
    #     pytorch_model = PyTorchUpConvResBlock(in_channels, out_channels)
    #     tf_model = TFUpConvResBlock(in_channels, out_channels)

    #     input_tensor = np.random.rand(self.batch_size, in_channels, self.height // 2, self.width // 2).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

    #     debug_print("pytorch", f"UpConvResBlock input shape: {pytorch_input.shape}")
    #     pytorch_output = pytorch_model(pytorch_input)
    #     debug_print("pytorch", f"UpConvResBlock output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"UpConvResBlock input shape: {tf_input.shape}")
    #     tf_output = tf_model(tf_input)
    #     debug_print("tf", f"UpConvResBlock output shape: {tf_output.shape}")

    #     tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "UpConvResBlock outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "UpConvResBlock outputs do not match within epsilon", False)
    #         raise

    # def test_feat_res_block(self):
    #     channels = 64
    #     pytorch_model = PyTorchFeatResBlock(channels)
    #     tf_model = TFFeatResBlock(channels)

    #     input_tensor = np.random.rand(self.batch_size, channels, self.height, self.width).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))

    #     debug_print("pytorch", f"FeatResBlock input shape: {pytorch_input.shape}")
    #     pytorch_output = pytorch_model(pytorch_input)
    #     debug_print("pytorch", f"FeatResBlock output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"FeatResBlock input shape: {tf_input.shape}")
    #     tf_output = tf_model(tf_input)
    #     debug_print("tf", f"FeatResBlock output shape: {tf_output.shape}")

    #     tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "FeatResBlock outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "FeatResBlock outputs do not match within epsilon", False)
    #         raise

    # def test_styled_conv(self):
    #     in_channel = 64
    #     out_channel = 128
    #     kernel_size = 3
    #     style_dim = 32
    #     pytorch_model = PyTorchStyledConv(in_channel, out_channel, kernel_size, style_dim)
    #     tf_model = TFStyledConv(in_channel, out_channel, kernel_size, style_dim)

    #     input_tensor = np.random.rand(self.batch_size, in_channel, self.height, self.width).astype(np.float32)
    #     style_tensor = np.random.rand(self.batch_size, style_dim).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     pytorch_style = torch.from_numpy(style_tensor)
    #     tf_input = tf.convert_to_tensor(np.transpose(input_tensor, (0, 2, 3, 1)))
    #     tf_style = tf.convert_to_tensor(style_tensor)

    #     debug_print("pytorch", f"StyledConv input shape: {pytorch_input.shape}, style shape: {pytorch_style.shape}")
    #     pytorch_output = pytorch_model(pytorch_input, pytorch_style)
    #     debug_print("pytorch", f"StyledConv output shape: {pytorch_output.shape}")

    #     debug_print("tf", f"StyledConv input shape: {tf_input.shape}, style shape: {tf_style.shape}")
    #     tf_output = tf_model(tf_input, tf_style)
    #     debug_print("tf", f"StyledConv output shape: {tf_output.shape}")

    #     tf_output_nchw = tf.transpose(tf_output, [0, 3, 1, 2])
    #     try:
    #         np.testing.assert_allclose(pytorch_output.detach().numpy(), tf_output_nchw.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #         debug_print("test", "StyledConv outputs match within epsilon", True)
    #     except AssertionError:
    #         debug_print("test", "StyledConv outputs do not match within epsilon", False)
    #         raise

if __name__ == '__main__':
    unittest.main()