import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

DEBUG = False
def debugPrint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        
# Custom NormLayer
class NormLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, norm_type='batch'):
        super(NormLayer, self).__init__()
        if norm_type == 'batch':
            self.norm = layers.BatchNormalization()
        elif norm_type == 'instance':
            self.norm = tfa.layers.InstanceNormalization(axis=-1)
        elif norm_type == 'layer':
            self.norm = layers.LayerNormalization(axis=-1)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def call(self, x):
        return self.norm(x)

# ConvBlock
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, padding='same', 
                 activation=tf.keras.activations.relu, norm_type='batch'):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.norm = NormLayer(out_channels, norm_type)
        self.activation = activation

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

# FeatResBlock
class FeatResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, dropout_rate=0, activation=tf.keras.activations.relu, norm_type='batch'):
        super(FeatResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, activation=activation, norm_type=norm_type)
        self.conv2 = ConvBlock(channels, channels, activation=None, norm_type=norm_type)
        self.activation = activation
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out += residual
        if self.activation:
            out = self.activation(out)
        return out

# DownConvResBlock
class DownConvResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, dropout_rate=0, activation=tf.keras.activations.relu, 
                 norm_type='batch', use_residual_scaling=False):
        super(DownConvResBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, strides=2, activation=activation, norm_type=norm_type)
        self.conv2 = ConvBlock(out_channels, out_channels, activation=None, norm_type=norm_type)
        self.activation = activation
        self.dropout = layers.Dropout(dropout_rate)
        self.feat_res_block1 = FeatResBlock(out_channels, dropout_rate, activation, norm_type)
        self.feat_res_block2 = FeatResBlock(out_channels, dropout_rate, activation, norm_type)
        
        self.shortcut = ConvBlock(in_channels, out_channels, kernel_size=1, strides=2, padding='valid', 
                                  activation=None, norm_type=norm_type)
        self.use_residual_scaling = use_residual_scaling
        if use_residual_scaling:
            self.residual_scaling = self.add_weight(shape=(1,), initializer='ones', trainable=True)

    def call(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.use_residual_scaling:
            out = out * self.residual_scaling
        out += residual
        if self.activation:
            out = self.activation(out)
        out = self.feat_res_block1(out)
        out = self.feat_res_block2(out)
        return out

# UpConvResBlock
class UpConvResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, dropout_rate=0, activation=tf.keras.activations.relu, 
                 norm_type='batch', upsample_mode='nearest', use_residual_scaling=False):
        super(UpConvResBlock, self).__init__()
        self.upsampling = layers.UpSampling2D(size=(2, 2), interpolation=upsample_mode)
        self.conv1 = ConvBlock(in_channels, out_channels, activation=activation, norm_type=norm_type)
        self.conv2 = ConvBlock(out_channels, out_channels, activation=None, norm_type=norm_type)
        self.activation = activation
        self.dropout = layers.Dropout(dropout_rate)
        self.feat_res_block1 = FeatResBlock(out_channels, dropout_rate, activation, norm_type)
        self.feat_res_block2 = FeatResBlock(out_channels, dropout_rate, activation, norm_type)
        
        self.shortcut = tf.keras.Sequential([
            layers.UpSampling2D(size=(2, 2), interpolation=upsample_mode),
            ConvBlock(in_channels, out_channels, kernel_size=1, padding='valid', activation=None, norm_type=norm_type)
        ])
        self.use_residual_scaling = use_residual_scaling
        if use_residual_scaling:
            self.residual_scaling = self.add_weight(shape=(1,), initializer='ones', trainable=True)

    def call(self, x):
        residual = self.shortcut(x)
        out = self.upsampling(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.dropout(out)
        if self.use_residual_scaling:
            out = out * self.residual_scaling
        out += residual
        if self.activation:
            out = self.activation(out)
        out = self.feat_res_block1(out)
        out = self.feat_res_block2(out)
        return out

# Visualization and testing functions remain largely unchanged, but adjusted for TensorFlow
def visualize_feature_maps(block, input_data, num_channels=4):
    x = input_data
    output = block(x)
    outputs = [output]
    titles = ['Output']

    num_outputs = len(outputs)
    fig, axs = plt.subplots(num_outputs, min(num_channels, output.shape[-1]), figsize=(20, 5 * num_outputs))
    if num_outputs == 1 and min(num_channels, output.shape[-1]) == 1:
        axs = np.array([[axs]])
    elif num_outputs == 1 or min(num_channels, output.shape[-1]) == 1:
        axs = np.array([axs])

    for i, out in enumerate(outputs):
        for j in range(min(num_channels, out.shape[-1])):
            ax = axs[i, j] if num_outputs > 1 and min(num_channels, output.shape[-1]) > 1 else axs[i]
            feature_map = out[0, :, :, j].numpy()
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            ax.imshow(feature_map, cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_title(f'{titles[i]}\nChannel {j}')
            else:
                ax.set_title(f'Channel {j}')
    plt.tight_layout()
    plt.show()

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension
    return img_tensor

def test_block_with_image(block, image_tensor):
    debugPrint("\nTesting Block with Image")
    input_shape = image_tensor.shape
    debugPrint(f"Input shape: {input_shape}")

    # Pass the image through the block
    output = block(image_tensor)
    debugPrint(f"Output shape: {output.shape}")

    # Visualize input and output
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Display input image
    input_img = image_tensor[0].numpy()
    ax1.imshow(input_img)
    ax1.set_title("Input Image")
    ax1.axis('off')

    # Display output feature map (first channel)
    output_img = output[0, :, :, 0].numpy()
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    ax2.imshow(output_img, cmap='viridis')
    ax2.set_title("Output Feature Map (First Channel)")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

    debugPrint("Block test with image passed successfully!")

if __name__ == "__main__":
    # Load the image
    image_path = "/path/to/your/image.png"
    image_tensor = load_and_preprocess_image(image_path)

    # Create a ResBlock (you can adjust the channels accordingly)
    resblock = DownConvResBlock(3, 64, downsample=True)

    # Test the block with the image
    test_block_with_image(resblock, image_tensor)
