import unittest
import torch
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import os
import platform
import subprocess
import numpy as np
from torch import nn
from tensorflow import keras


from resblocks import ConvBlock,NormLayer,DownConvResBlock,UpConvResBlock
from model import (DenseFeatureEncoder as PyTorchDenseFeatureEncoder,
                   LatentTokenEncoder as PyTorchLatentTokenEncoder,
                   LatentTokenDecoder as PyTorchLatentTokenDecoder,
                   FrameDecoder as PyTorchFrameDecoder,
                   FeatResBlock,
                    )
from tf.model import (DenseFeatureEncoder as TFDenseFeatureEncoder,
                      LatentTokenEncoder as TFLatentTokenEncoder,
                      LatentTokenDecoder as TFLatentTokenDecoder,
                      FrameDecoder as TFFrameDecoder,
                      DownConvResBlock as TFDownConvResBlock,
                      UpConvResBlock as TFUpConvResBlock,
                      FeatResBlock as TFFeatResBlock,
                      StyledConv as TFStyledConv)

from tf.resblocks import (ConvBlock as TFConvBlock,NormLayer as TFNormLayer,DownConvResBlock as TFDownConvResBlock,UpConvResBlock as TFUpConvResBlock)
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

def assert_close(pytorch_tensor, tf_tensor, name):
    pytorch_np = pytorch_tensor.detach().cpu().numpy()
    tf_np = tf_tensor.numpy()
    
    assert pytorch_np.shape == tf_np.shape, f"{name} shapes do not match. PyTorch: {pytorch_np.shape}, TensorFlow: {tf_np.shape}"
    
    try:
        np.testing.assert_allclose(pytorch_np, tf_np, rtol=1e-5, atol=1e-5)
        print(f"{name} values match within tolerance")
    except AssertionError as e:
        print(f"{name} values do not match within tolerance")
        print(f"Max absolute difference: {np.max(np.abs(pytorch_np - tf_np))}")
        print(f"Max relative difference: {np.max(np.abs((pytorch_np - tf_np) / (tf_np + 1e-7)))}")
        raise e
    
def manual_conv2d(input, weight, stride, padding):
    # Implement a simple 2D convolution using NumPy
    # This is a basic implementation and doesn't handle all cases
    out_h = (input.shape[2] - weight.shape[2] + 2 * padding) // stride + 1
    out_w = (input.shape[3] - weight.shape[3] + 2 * padding) // stride + 1
    out = np.zeros((input.shape[0], weight.shape[0], out_h, out_w))
    
    padded_input = np.pad(input, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    
    for i in range(out_h):
        for j in range(out_w):
            out[:, :, i, j] = np.sum(
                padded_input[:, :, i*stride:i*stride+weight.shape[2], j*stride:j*stride+weight.shape[3], None] *
                weight[None, :, :, :, :],
                axis=(1, 2, 3)
            )
    return out

class TestIMFComponents(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.input_channels = 3
        self.height = 256
        self.width = 256
        self.latent_dim = 32
        self.epsilon = 1e-5  # Tolerance for floating-point comparisons



    def test_conv_block(self):
        print("\n🧪 Testing test_conv_block")

        in_channels, out_channels = 64, 128
        kernel_size, stride, padding = 3, 1, 1
        
        np.random.seed(0)
        input_np = np.random.rand(1, in_channels, 32, 32).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        input_tf = tf.convert_to_tensor(input_np)
        
        torch.manual_seed(0)
        conv_torch = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        tf.random.set_seed(0)
        conv_tf = tf.keras.layers.Conv2D(out_channels, kernel_size, strides=stride, padding='same', 
                                        use_bias=False, data_format='channels_first')
        conv_tf.build((None, in_channels, None, None))
        
        # Copy weights from PyTorch to TensorFlow
        tf_weights = conv_torch.weight.detach().numpy().transpose(2, 3, 1, 0)
        conv_tf.set_weights([tf_weights])
        
        output_torch = conv_torch(input_torch).detach().numpy()
        output_tf = conv_tf(input_tf).numpy()
        
        print("Input shape:", input_np.shape)
        print("PyTorch output shape:", output_torch.shape)
        print("TensorFlow output shape:", output_tf.shape)
        print("\nPyTorch Conv1 output first few values:", output_torch.flatten()[:10])
        print("TensorFlow Conv1 output first few values:", output_tf.flatten()[:10])
        
        try:
            np.testing.assert_allclose(output_torch, output_tf, rtol=1e-5, atol=1e-5)
            print("\nOutputs match!")
        except AssertionError as e:
            print("\nOutputs do not match.")
            print("Max absolute difference:", np.max(np.abs(output_torch - output_tf)))
            print("Max relative difference:", np.max(np.abs((output_torch - output_tf) / (output_tf + 1e-7))))

        print("\nPyTorch weights (first few values):", conv_torch.weight.data.numpy().flatten()[:10])
        print("TensorFlow weights (first few values):", conv_tf.get_weights()[0].transpose(3, 2, 0, 1).flatten()[:10])

    def test_normlayer():
        print("\n🧪 Testing NormLayer")
        
        num_features = 64
        input_tensor = torch.randn(2, num_features, 32, 32)
        
        for norm_type in ['batch', 'instance', 'layer']:
            norm_layer = NormLayer(num_features, norm_type)
            output = norm_layer(input_tensor)
            
            print(f"{norm_type.capitalize()} Norm output shape:", output.shape)
            print(f"{norm_type.capitalize()} Norm output first few values:", output.flatten()[:5])
            
            assert output.shape == input_tensor.shape, f"{norm_type.capitalize()} Norm output shape mismatch"

    def test_convblock(self):
        print("\n🧪 Testing ConvBlock")
        
        in_channels, out_channels = 3, 64
        input_tensor = torch.randn(2, in_channels, 32, 32)
        
        conv_block = ConvBlock(in_channels, out_channels)
        output = conv_block(input_tensor)
        
        print("ConvBlock output shape:", output.shape)
        print("ConvBlock output first few values:", output.flatten()[:5])
        
        assert output.shape == (2, out_channels, 32, 32), "ConvBlock output shape mismatch"

    def test_featresblock(self):
        print("\n🧪 Testing FeatResBlock")
        
        channels = 64
        input_tensor = torch.randn(2, channels, 32, 32)
        
        feat_res_block = FeatResBlock(channels)
        output = feat_res_block(input_tensor)
        
        print("FeatResBlock output shape:", output.shape)
        print("FeatResBlock output first few values:", output.flatten()[:5])
        
        assert output.shape == input_tensor.shape, "FeatResBlock output shape mismatch"

    def test_downconvresblock(self):
        print("\n🧪 Testing DownConvResBlock")

        in_channels, out_channels = 64, 128
        batch_size, height, width = 2, 32, 32
        
        np.random.seed(0)
        input_np = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        input_tf = tf.convert_to_tensor(input_np)
        
        torch.manual_seed(0)
        block_torch = DownConvResBlock(in_channels, out_channels)
        
        tf.random.set_seed(0)
        block_tf = TFDownConvResBlock(in_channels, out_channels)
        
        # Copy weights from PyTorch to TensorFlow
        for torch_layer, tf_layer in zip(block_torch.modules(), block_tf.layers):
            if isinstance(torch_layer, nn.Conv2d) and isinstance(tf_layer, tf.keras.layers.Conv2D):
                tf_weights = torch_layer.weight.detach().numpy().transpose(2, 3, 1, 0)
                tf_layer.set_weights([tf_weights])
        
        output_torch = block_torch(input_torch).detach().numpy()
        output_tf = block_tf(input_tf).numpy()
        
        print("Input shape:", input_np.shape)
        print("PyTorch output shape:", output_torch.shape)
        print("TensorFlow output shape:", output_tf.shape)
        print("\nPyTorch output first few values:", output_torch.flatten()[:10])
        print("TensorFlow output first few values:", output_tf.flatten()[:10])
        
        try:
            np.testing.assert_allclose(output_torch, output_tf, rtol=1e-5, atol=1e-5)
            print("\nOutputs match!")
        except AssertionError as e:
            print("\nOutputs do not match.")
            print("Max absolute difference:", np.max(np.abs(output_torch - output_tf)))
            print("Max relative difference:", np.max(np.abs((output_torch - output_tf) / (output_tf + 1e-7))))

        # Test gradient flow
        input_torch.requires_grad_(True)
        input_tf = tf.Variable(input_np)

        with tf.GradientTape() as tape:
            output_tf = block_tf(input_tf)
        
        output_torch = block_torch(input_torch)
        output_torch.sum().backward()
        
        grad_torch = input_torch.grad.numpy()
        grad_tf = tape.gradient(output_tf, input_tf).numpy()
        
        print("\nPyTorch input gradient first few values:", grad_torch.flatten()[:10])
        print("TensorFlow input gradient first few values:", grad_tf.flatten()[:10])
        
        try:
            np.testing.assert_allclose(grad_torch, grad_tf, rtol=1e-5, atol=1e-5)
            print("\nGradients match!")
        except AssertionError as e:
            print("\nGradients do not match.")
            print("Max absolute difference:", np.max(np.abs(grad_torch - grad_tf)))
            print("Max relative difference:", np.max(np.abs((grad_torch - grad_tf) / (grad_tf + 1e-7))))

    def test_upconvresblock(self):
        print("\n🧪 Testing UpConvResBlock")

        in_channels, out_channels = 128, 64
        batch_size, height, width = 2, 16, 16
        
        np.random.seed(0)
        input_np = np.random.rand(batch_size, in_channels, height, width).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        input_tf = tf.convert_to_tensor(input_np)
        
        torch.manual_seed(0)
        block_torch = UpConvResBlock(in_channels, out_channels)
        
        tf.random.set_seed(0)
        block_tf = TFUpConvResBlock(in_channels, out_channels)
        
        # Copy weights from PyTorch to TensorFlow
        for torch_layer, tf_layer in zip(block_torch.modules(), block_tf.layers):
            if isinstance(torch_layer, nn.Conv2d) and isinstance(tf_layer, tf.keras.layers.Conv2D):
                tf_weights = torch_layer.weight.detach().numpy().transpose(2, 3, 1, 0)
                tf_layer.set_weights([tf_weights])
        
        output_torch = block_torch(input_torch).detach().numpy()
        output_tf = block_tf(input_tf).numpy()
        
        print("Input shape:", input_np.shape)
        print("PyTorch output shape:", output_torch.shape)
        print("TensorFlow output shape:", output_tf.shape)
        print("\nPyTorch output first few values:", output_torch.flatten()[:10])
        print("TensorFlow output first few values:", output_tf.flatten()[:10])
        
        try:
            np.testing.assert_allclose(output_torch, output_tf, rtol=1e-5, atol=1e-5)
            print("\nOutputs match!")
        except AssertionError as e:
            print("\nOutputs do not match.")
            print("Max absolute difference:", np.max(np.abs(output_torch - output_tf)))
            print("Max relative difference:", np.max(np.abs((output_torch - output_tf) / (output_tf + 1e-7))))

        # Test gradient flow
        input_torch.requires_grad_(True)
        input_tf = tf.Variable(input_np)

        with tf.GradientTape() as tape:
            output_tf = block_tf(input_tf)
        
        output_torch = block_torch(input_torch)
        output_torch.sum().backward()
        
        grad_torch = input_torch.grad.numpy()
        grad_tf = tape.gradient(output_tf, input_tf).numpy()
        
        print("\nPyTorch input gradient first few values:", grad_torch.flatten()[:10])
        print("TensorFlow input gradient first few values:", grad_tf.flatten()[:10])
        
        try:
            np.testing.assert_allclose(grad_torch, grad_tf, rtol=1e-5, atol=1e-5)
            print("\nGradients match!")
        except AssertionError as e:
            print("\nGradients do not match.")
            print("Max absolute difference:", np.max(np.abs(grad_torch - grad_tf)))
            print("Max relative difference:", np.max(np.abs((grad_torch - grad_tf) / (grad_tf + 1e-7))))
            

    def test_downconvresblock_components(self):
        in_channels = 64
        out_channels = 128
        pytorch_block =  DownConvResBlock(in_channels, out_channels)
        tf_block = TFDownConvResBlock(in_channels, out_channels)

        input_tensor = np.random.rand(self.batch_size, in_channels, self.height, self.width).astype(np.float32)
        
        print("\n🧪 Testing DownConvResBlock components")
        
        # Convert input tensor to both PyTorch and TensorFlow tensors
        pytorch_input = torch.from_numpy(input_tensor).float()
        tf_input = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        # Test Conv1
        print("\nTesting Conv1")
        pytorch_conv1 = pytorch_block.conv1(pytorch_input)
        tf_conv1 = tf_block.conv1(tf_input)

        print("PyTorch Conv1 output first few values:", pytorch_conv1.flatten()[:10])
        print("TensorFlow Conv1 output first few values:", tf_conv1.numpy().flatten()[:10])

        self.assert_tensors_close(pytorch_conv1, tf_conv1, "Conv1 output")
        
        # Test Conv2
        print("\nTesting Conv2")
        pytorch_conv2 = pytorch_block.conv2(pytorch_conv1)
        tf_conv2 = tf_block.conv2(tf_conv1)
        self.assert_tensors_close(pytorch_conv2, tf_conv2, "Conv2 output")
        
        # Test Shortcut
        print("\nTesting Shortcut")
        pytorch_shortcut = pytorch_block.skip(pytorch_input)
        tf_shortcut = tf_block.skip(tf_input)
        self.assert_tensors_close(pytorch_shortcut, tf_shortcut, "Shortcut output")
        
        # Test Residual Addition
        print("\nTesting Residual Addition")
        pytorch_residual = pytorch_conv2 + pytorch_shortcut
        tf_residual = tf_conv2 + tf_shortcut
        self.assert_tensors_close(pytorch_residual, tf_residual, "Residual addition")
        
        # Test Final Activation
        print("\nTesting Final Activation")
        pytorch_output = pytorch_block.activation(pytorch_residual)
        tf_output = tf_block.activation(tf_residual)
        self.assert_tensors_close(pytorch_output, tf_output, "Final output")
        
        # Test FeatResBlock1
        print("\nTesting FeatResBlock1")
        pytorch_featres1 = pytorch_block.feat_res_block1(pytorch_output)
        tf_featres1 = tf_block.feat_res_block1(tf_output)
        self.assert_tensors_close(pytorch_featres1, tf_featres1, "FeatResBlock1 output")
        
        # Test FeatResBlock2
        print("\nTesting FeatResBlock2")
        pytorch_featres2 = pytorch_block.feat_res_block2(pytorch_featres1)
        tf_featres2 = tf_block.feat_res_block2(tf_featres1)
        self.assert_tensors_close(pytorch_featres2, tf_featres2, "FeatResBlock2 output")
        
        print("All DownConvResBlock component tests passed successfully!")

    def assert_tensors_close(self, pytorch_tensor, tf_tensor, name):
        pytorch_np = pytorch_tensor.detach().cpu().numpy()
        tf_np = tf_tensor.numpy()
        
        self.assertEqual(pytorch_np.shape, tf_np.shape, f"{name} shapes do not match. PyTorch: {pytorch_np.shape}, TensorFlow: {tf_np.shape}")
        
        try:
            np.testing.assert_allclose(pytorch_np, tf_np, rtol=self.epsilon, atol=self.epsilon)
            print(f"{name} values match within tolerance")
        except AssertionError as e:
            print(f"{name} values do not match within tolerance")
            print(f"Max absolute difference: {np.max(np.abs(pytorch_np - tf_np))}")
            print(f"Max relative difference: {np.max(np.abs((pytorch_np - tf_np) / (tf_np + 1e-7)))}")
            raise e
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

    # def test_dense_feature_encoder(self):
    #     pytorch_model = PyTorchDenseFeatureEncoder()
    #     tf_model = TFDenseFeatureEncoder()

    #     # Set both models to evaluation mode
    #     pytorch_model.eval()
    #     tf_model.trainable = False

    #     input_tensor = np.random.rand(self.batch_size, self.input_channels, self.height, self.width).astype(np.float32)
    #     pytorch_input = torch.from_numpy(input_tensor)
    #     tf_input = tf.convert_to_tensor(input_tensor)

    #     # PyTorch forward pass with intermediate outputs
    #     pytorch_intermediates = []
    #     x = pytorch_input
    #     x = pytorch_model.initial_conv(x)
    #     debug_print("pytorch", f"After initial conv: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
    #     for i, block in enumerate(pytorch_model.down_blocks):
    #         x = block(x)
    #         pytorch_intermediates.append(x)
    #         debug_print("pytorch", f"After down_block {i}: {x.shape}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
    #     pytorch_output = pytorch_intermediates[-4:]  # Assuming last 4 are the output

    #     # TensorFlow forward pass with intermediate outputs and error handling
    #     tf_intermediates = []
    #     x = tf_input
    #     try:
    #         x = tf_model.initial_conv(x)
    #         debug_print("tf", f"After initial conv: {x.shape}, Mean: {tf.reduce_mean(x).numpy():.6f}, Std: {tf.math.reduce_std(x).numpy():.6f}")
    #         for i, block in enumerate(tf_model.down_blocks):
    #             x = block(x)
    #             tf_intermediates.append(x)
    #             debug_print("tf", f"After down_block {i}: {x.shape}, Mean: {tf.reduce_mean(x).numpy():.6f}, Std: {tf.math.reduce_std(x).numpy():.6f}")
    #     except Exception as e:
    #         print(f"Error in TensorFlow forward pass: {str(e)}")
    #         print("GPU Info:")
    #         print(tf.config.list_physical_devices('GPU'))
    #         print("TensorFlow version:", tf.__version__)
    #         print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    #         print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    #         raise

    #     tf_output = tf_intermediates[-4:]  # Assuming last 4 are the output

    #     debug_print("pytorch", f"DenseFeatureEncoder output shapes: {[out.shape for out in pytorch_output]}")
    #     debug_print("tf", f"DenseFeatureEncoder output shapes: {[out.shape for out in tf_output]}")

    #     # Compare shapes, sizes, and values for each output
    #     for i, (pytorch_out, tf_out) in enumerate(zip(pytorch_output, tf_output)):
    #         # Compare shapes
    #         self.assertEqual(pytorch_out.shape, tf_out.shape, f"Output {i} shapes do not match")
    #         debug_print("test", f"DenseFeatureEncoder output {i} shapes match: {pytorch_out.shape}")

    #         # Compare sizes
    #         self.assertEqual(pytorch_out.numel(), tf_out.numpy().size, f"Output {i} sizes do not match")
    #         debug_print("test", f"DenseFeatureEncoder output {i} sizes match: {pytorch_out.numel()}")

    #         # Compare values
    #         try:
    #             np.testing.assert_allclose(pytorch_out.detach().numpy(), tf_out.numpy(), rtol=self.epsilon, atol=self.epsilon)
    #             debug_print("test", f"DenseFeatureEncoder output {i} values match within epsilon", True)
    #         except AssertionError as e:
    #             debug_print("test", f"DenseFeatureEncoder output {i} values do not match within epsilon", False)
    #             print(f"Max absolute difference: {np.max(np.abs(pytorch_out.detach().numpy() - tf_out.numpy()))}")
    #             print(f"Max relative difference: {np.max(np.abs((pytorch_out.detach().numpy() - tf_out.numpy()) / tf_out.numpy()))}")
    #             raise

    #         # Additional check: Print mean and standard deviation of outputs
    #         pytorch_mean = torch.mean(pytorch_out).item()
    #         pytorch_std = torch.std(pytorch_out).item()
    #         tf_mean = tf.reduce_mean(tf_out).numpy()
    #         tf_std = tf.math.reduce_std(tf_out).numpy()

    #         debug_print("pytorch", f"DenseFeatureEncoder output {i} mean: {pytorch_mean:.6f}, std: {pytorch_std:.6f}")
    #         debug_print("tf", f"DenseFeatureEncoder output {i} mean: {tf_mean:.6f}, std: {tf_std:.6f}")

if __name__ == '__main__':
    unittest.main()