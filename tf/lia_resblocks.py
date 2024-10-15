import math
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tf.grouped_conv import grouped_convolution2D
from tensorflow.keras import layers


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    out = tf.nn.leaky_relu(input + bias, alpha=negative_slope) * scale
    print(f"fused_leaky_relu - input shape: {input.shape}, output shape: {out.shape}")
    return out

# NCHW
class PyConv2D(tf.keras.layers.Conv2D):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding='valid', 
                 data_format='channels_first', 
                 dilation_rate=(1, 1), 
                 groups=1,
                 activation=None, 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None, 
                 activity_regularizer=None, 
                 kernel_constraint=None, 
                 bias_constraint=None, 
                 **kwargs):
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

class FusedLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super(FusedLeakyReLU, self).__init__()
        self.bias = self.add_weight(shape=(channel,), initializer='zeros', trainable=True, name='bias')
        self.negative_slope = negative_slope
        self.scale = scale
        self.data_format = 'channels_first'

    def call(self, input):
        print(f"FusedLeakyReLU - input shape: {input.shape}")
        if self.data_format == 'channels_first':
            bias = tf.reshape(self.bias, [1, -1, 1, 1])
        else:
            bias = tf.reshape(self.bias, [1, 1, 1, -1])
        out = tf.nn.leaky_relu(input + bias, alpha=self.negative_slope) * self.scale
        print(f"FusedLeakyReLU - output shape: {out.shape}")
        return out


@tf.function
def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    print(f"⛽ upfirdn2d_native - input shape: {input.shape}, kernel shape: {kernel.shape}")

    # Get input and kernel shapes
    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    minor = input_shape[1]  # Number of channels
    in_h = input_shape[2]
    in_w = input_shape[3]
    kernel_h = tf.shape(kernel)[0]
    kernel_w = tf.shape(kernel)[1]

    # Convert input to 'NHWC' format for TensorFlow
    input = tf.transpose(input, [0, 2, 3, 1])  # Shape: [batch_size, in_h, in_w, channels]

    # Upsample: Insert zeros between pixels
    input = tf.reshape(input, [batch_size, in_h, 1, in_w, 1, minor])
    padding = [[0, 0], [0, 0], [0, up_y - 1], [0, 0], [0, up_x - 1], [0, 0]]
    input = tf.pad(input, padding)
    out = tf.reshape(input, [batch_size, in_h * up_y, in_w * up_x, minor])

    # Pad the output
    pad_y0_pos = max(pad_y0, 0)
    pad_y1_pos = max(pad_y1, 0)
    pad_x0_pos = max(pad_x0, 0)
    pad_x1_pos = max(pad_x1, 0)
    padding = [[0, 0], [pad_y0_pos, pad_y1_pos], [pad_x0_pos, pad_x1_pos], [0, 0]]
    out = tf.pad(out, padding)

    # Crop if padding values are negative
    start_y = tf.maximum(-pad_y0, 0)
    end_y = tf.shape(out)[1] - tf.maximum(-pad_y1, 0)
    start_x = tf.maximum(-pad_x0, 0)
    end_x = tf.shape(out)[2] - tf.maximum(-pad_x1, 0)
    out = out[:, start_y:end_y, start_x:end_x, :]

    # Reshape for convolution
    out = tf.reshape(out, [-1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1, 1])

    # Prepare the kernel
    kernel = tf.reverse(kernel, axis=[0, 1])
    kernel = tf.reshape(kernel, [kernel_h, kernel_w, 1, 1])

    # Perform convolution
    out = tf.nn.conv2d(out, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format='NHWC')

    # Reshape back to original dimensions
    out = tf.reshape(out, [batch_size, minor, tf.shape(out)[1], tf.shape(out)[2]])
    out = tf.transpose(out, [0, 2, 3, 1])  # Convert back to 'NCHW' format

    # Downsample
    out = out[:, ::down_y, ::down_x, :]
    out = tf.transpose(out, [0, 3, 1, 2])  # Final output in 'NCHW' format

    print(f"upfirdn2d_native - output shape: {out.shape}")
    return out


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    print(f"upfirdn2d - input shape: {input.shape}, kernel shape: {kernel.shape}")
    out = upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])
    print(f"upfirdn2d - output shape: {out.shape}")
    return out

def make_kernel(k):
    k = np.array(k, dtype=np.float32)
    if k.ndim == 1:
        k = k[:, None] * k[None, :]
    k /= np.sum(k)
    return k



class Blur(tf.keras.layers.Layer):
    def __init__(self, kernel, pad, upsample_factor=1):
        super(Blur, self).__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.kernel = tf.constant(kernel, dtype=tf.float32)
        self.pad = pad

    def call(self, input):
        print(f"Blur - input shape: {input.shape}")
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        print(f"Blur - output shape: {out.shape}")
        return out
    


class ScaledLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def call(self, input):
        return tf.nn.leaky_relu(input, alpha=self.negative_slope)

import tensorflow as tf
import math
class EqualConv2d(tf.keras.layers.Layer):
    def __init__(self, 
                 filters, 
                 kernel_size, 
                 strides=(1, 1), 
                 padding=0,  # Now padding is always an integer
                 data_format='channels_first', 
                 use_bias=True, 
                 **kwargs):
        super(EqualConv2d, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, tuple) else (strides, strides)
        self.padding = padding  # Integer padding
        self.data_format = data_format
        self.use_bias = use_bias

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            in_channels = input_shape[1]
        else:
            in_channels = input_shape[-1]

        # Initialize weights with normal distribution
        kernel_shape = self.kernel_size + (in_channels, self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=tf.random_normal_initializer(),
            trainable=True,
            name='kernel'
        )

        # Calculate the scaling factor
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.scale = 1 / math.sqrt(fan_in)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer='zeros',
                trainable=True,
                name='bias'
            )
        else:
            self.bias = None

        super(EqualConv2d, self).build(input_shape)

    def call(self, inputs):
        # Scale the kernel
        scaled_kernel = self.kernel * self.scale

        # Manually pad the input if padding is greater than zero
        if self.padding > 0:
            if self.data_format == 'channels_first':
                paddings = [[0, 0], [0, 0], [self.padding, self.padding], [self.padding, self.padding]]
            else:
                paddings = [[0, 0], [self.padding, self.padding], [self.padding, self.padding], [0, 0]]
            inputs = tf.pad(inputs, paddings)
        
        # Prepare strides
        if self.data_format == 'channels_first':
            strides = [1, 1] + list(self.strides)
        else:
            strides = [1] + list(self.strides) + [1]
        
        outputs = tf.nn.conv2d(
            inputs,
            scaled_kernel,
            strides=strides,
            padding='VALID',  # We handled padding manually
            data_format='NCHW' if self.data_format == 'channels_first' else 'NHWC'
        )

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW' if self.data_format == 'channels_first' else 'NHWC')

        return outputs

class EqualLinear(tf.keras.layers.Layer):
    def __init__(self, out_dim, use_bias=True, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.lr_mul = lr_mul
        self.activation = activation
        print(f"EqualLinear.__init__: out_dim={out_dim}, use_bias={use_bias}, lr_mul={lr_mul}, activation={activation}")

    def build(self, input_shape):
        in_dim = input_shape[-1]
        self.scale = (1 / math.sqrt(in_dim)) * self.lr_mul
        print(f"EqualLinear.build: input_shape={input_shape}, in_dim={in_dim}, scale={self.scale}")
        
        self.weight = self.add_weight(
            name="weight",
            shape=(in_dim, self.out_dim),
            initializer="random_normal",
            trainable=True,
        )
        print(f"Weight shape: {self.weight.shape}")
        
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.out_dim,),
                initializer="zeros",
                trainable=True,
            )
            print(f"Bias shape: {self.bias.shape}")
        else:
            print("No bias")
        
        super(EqualLinear, self).build(input_shape)

    def call(self, input):
        print(f"🍟 EqualLinear.call: input shape = {input.shape}")
        
        out = tf.matmul(input, self.weight * self.scale)
        print(f"After matmul: shape = {out.shape}")
        
        if self.use_bias:
            out = out + self.bias * self.lr_mul
            print(f"After adding bias: shape = {out.shape}")
        
        if self.activation == 'fused_lrelu':
            out = fused_leaky_relu(out, None)
            print(f"After fused_leaky_relu: shape = {out.shape}")
        
        return out

    def get_config(self):
        config = super(EqualLinear, self).get_config()
        config.update({
            "out_dim": self.out_dim,
            "use_bias": self.use_bias,
            "lr_mul": self.lr_mul,
            "activation": self.activation
        })
        return config

import tensorflow as tf
import math
import numpy as np
class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True):
        super(ConvLayer, self).__init__()
        self.downsample = downsample
        self.activate = activate
        self.use_bias = bias

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
            stride = 2
            self.padding = 0  # No padding for downsampling
        else:
            self.blur = None
            stride = 1
            self.padding = kernel_size // 2  # Integer padding

        self.conv = EqualConv2d(
            filters=out_channel,
            kernel_size=kernel_size,
            strides=(stride, stride),
            padding=self.padding,  # Pass the integer padding
            use_bias=bias and not activate,
            data_format='channels_first'
        )

        if activate:
            if bias:
                self.activation = FusedLeakyReLU(out_channel)
            else:
                self.activation = ScaledLeakyReLU(0.2)
        else:
            self.activation = None

    def call(self, x):
        if self.downsample and self.blur is not None:
            print(f"ConvLayer - module: {type(self.blur).__name__}")
            x = self.blur(x)

        print(f"ConvLayer - module: {type(self.conv).__name__}")
        x = self.conv(x)

        if self.activate and self.activation is not None:
            print(f"ConvLayer - module: {type(self.activation).__name__}")
            x = self.activation(x)

        return x



    
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)

    def call(self, input):
        print(f"ResBlock - input shape: {input.shape}")
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        print(f"ResBlock - output shape: {out.shape}")
        return out


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def call(self, input):
        return input * tf.math.rsqrt(tf.reduce_mean(input ** 2, axis=-1, keepdims=True) + 1e-8)

class StyledConv(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super(StyledConv, self).__init__()
        self.upsample = upsample
        self.conv = ModulatedConv2d(
            filters=out_channel,
            kernel_size=kernel_size,
            demod=demodulate,
            padding='same'
        )
        self.noise = NoiseInjection()
        self.activation = FusedLeakyReLU(out_channel)

    def call(self, input, style, noise=None):
        print(f"StyledConv call - input shape: {input.shape}, style shape: {style.shape}")
        if self.upsample:
            input = tf.image.resize(input, [tf.shape(input)[1]*2, tf.shape(input)[2]*2], method='nearest')
            print(f"StyledConv call - after upsample, input shape: {input.shape}")
        out = self.conv([input, style])
        print(f"StyledConv call - after conv, output shape: {out.shape}")
        out = self.noise(out, noise=noise)
        print(f"StyledConv call - after noise, output shape: {out.shape}")
        out = self.activation(out)
        print(f"StyledConv call - final output shape: {out.shape}")
        return out
    
import tensorflow as tf
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import layers, initializers, regularizers, constraints
import keras.backend as K


# Custom mod-demod convolutional layer

class ModulatedConv2d(layers.Layer):
    def __init__(self, filters, kernel_size,
                 strides=1, padding='same', dilation_rate=(1,1),
                 kernel_initializer='glorot_uniform', kernel_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None,
                 demod=True, **kwargs):
        super(ModulatedConv2d, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = 'same' if isinstance(padding, int) else conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.demod = demod
        self.epsilon = 1e-8

    def build(self, input_shape):
        input_shape, style_shape = input_shape
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        
        self.style_scale = self.add_weight(
            shape=(style_shape[-1], self.filters),
            initializer='ones',
            name='style_scale',
            trainable=True
        )

        print(f"ModulatedConv2D build - kernel shape: {self.kernel.shape}, style_scale shape: {self.style_scale.shape}")
        self.built = True

    def call(self, inputs):
        x, style = inputs
        print(f"⛳ ModulatedConv2D call - input x shape: {x.shape}, style shape: {style.shape}")
        batch_size = tf.shape(x)[0]

        # Apply style scaling
        style = tf.matmul(style, self.style_scale)
        style = tf.reshape(style, [batch_size, 1, 1, self.filters])
        print(f"🍟 Style shape after scaling: {style.shape}")
        
        # Modulate
        w = self.kernel
        w = w[tf.newaxis, :, :, :, :]  # [1, kh, kw, in_ch, out_ch]
        w = w * (style[:, tf.newaxis, tf.newaxis, tf.newaxis, :] + 1)
        print(f"Weight shape after style modulation: {w.shape}")

        # Demodulate
        if self.demod:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(w), axis=[1,2,3]) + self.epsilon)
            w = w * d[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
        print(f"Weight shape after demodulation: {w.shape}")

        # Reshape for convolution
        w = tf.transpose(tf.reshape(w, [batch_size * self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2], self.filters]), [1, 2, 3, 0])
        print(f"Weight shape before convolution: {w.shape}")
        
        # Convolution
        x = tf.transpose(x, [0, 2, 3, 1])  # NHWC
        x = tf.reshape(x, [1, -1, tf.shape(x)[2], tf.shape(x)[3]])  # [1, b*in_ch, h, w]
        print(f"Input shape before convolution: {x.shape}")
        x = tf.nn.conv2d(x, w, strides=self.strides, padding=self.padding.upper(), data_format="NHWC")
        print(f"Output shape after convolution: {x.shape}")
        x = tf.reshape(x, [batch_size, tf.shape(x)[1], tf.shape(x)[2], self.filters])
        x = tf.transpose(x, [0, 3, 1, 2])  # NCHW
        
        print(f"ModulatedConv2D call - output x shape: {x.shape}")
        return x

    def compute_output_shape(self, input_shape):
        input_shape = input_shape[0]
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0],) + tuple(new_space) + (self.filters,)
    
class NoiseInjection(tf.keras.layers.Layer):
    def __init__(self):
        super(NoiseInjection, self).__init__()
        self.weight = self.add_weight(shape=(1,), initializer='zeros', trainable=True)

    def call(self, image, noise=None):
        if noise is None:
            batch, height, width, channels = tf.shape(image)
            noise = tf.random.normal([batch, height, width, 1])
        return image + self.weight * noise

class ConstantInput(tf.keras.layers.Layer):
    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.constant = self.add_weight(shape=(1, size, size, channel), initializer='random_normal', trainable=True)

    def call(self, input):
        batch_size = tf.shape(input)[0]
        out = tf.tile(self.constant, [batch_size, 1, 1, 1])
        return out

class ToRGB(tf.keras.layers.Layer):
    def __init__(self, in_channel, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.conv = EqualConv2d(in_channel, 3, kernel_size=1, stride=1, padding='same')
        self.bias = self.add_weight(shape=(1, 1, 1, 3), initializer='zeros', trainable=True)

    def call(self, input, skip=None):
        out = self.conv(input)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = tf.image.resize(skip, [tf.shape(skip)[1]*2, tf.shape(skip)[2]*2], method='nearest')
            out = out + skip
        return out

class Synthesis(tf.keras.layers.Layer):
    def __init__(self, size, style_dim, blur_kernel=[1, 3, 3, 1], channel_multiplier=1):
        super(Synthesis, self).__init__()
        self.size = size
        self.style_dim = style_dim
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = []
        self.to_rgbs = []

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel))
            in_channel = out_channel

    def call(self, styles):
        out = self.input(styles)
        out = self.conv1(out, styles[:, 0])

        for i in range(len(self.convs)):
            conv = self.convs[i]
            style = styles[:, i + 1]
            out = conv(out, style)

            if i % 2 == 1:
                to_rgb = self.to_rgbs[i // 2]
                out = to_rgb(out)
        return out

class Generator(tf.keras.Model):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=1):
        super(Generator, self).__init__()
        self.style_dim = style_dim
        self.size = size
        self.style = [EqualLinear(style_dim, style_dim, activation='fused_lrelu') for _ in range(n_mlp)]
        self.synthesis = Synthesis(size, style_dim, channel_multiplier=channel_multiplier)

    def call(self, styles):
        for layer in self.style:
            styles = layer(styles)
        out = self.synthesis(styles)
        return out
