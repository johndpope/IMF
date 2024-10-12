import math
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    out = tf.nn.leaky_relu(input + bias, alpha=negative_slope) * scale
    print(f"fused_leaky_relu - input shape: {input.shape}, output shape: {out.shape}")
    return out

class FusedLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super(FusedLeakyReLU, self).__init__()
        self.bias = self.add_weight(shape=(channel,), initializer='zeros', trainable=True, name='bias')
        self.negative_slope = negative_slope
        self.scale = scale

    def call(self, input):
        print(f"FusedLeakyReLU - input shape: {input.shape}")
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        print(f"FusedLeakyReLU - output shape: {out.shape}")
        return out


@tf.function
def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    _, h, w, c = input.shape
    
    # Upsample
    if up > 1:
        out = tf.reshape(input, [-1, h, 1, w, 1, c])
        out = tf.pad(out, [[0, 0], [0, 0], [0, up - 1], [0, 0], [0, up - 1], [0, 0]])
        out = tf.reshape(out, [-1, h * up, w * up, c])
    else:
        out = input

    # Pad
    if pad[0] > 0 or pad[1] > 0:
        out = tf.pad(out, [[0, 0], [pad[0], pad[1]], [pad[0], pad[1]], [0, 0]])

    # Prepare kernel
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.tile(kernel, [1, 1, c, 1])

    # Apply kernel
    out = tf.nn.depthwise_conv2d(out, kernel, strides=[1, down, down, 1], padding='VALID')

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


class EqualConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding='SAME', bias=True):
        super(EqualConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        self.weight = self.add_weight(
            name='weight',
            shape=(kernel_size, kernel_size, in_channel, out_channel),
            initializer=tf.random_normal_initializer(0, 1),
            trainable=True
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        
        if bias:
            self.bias = self.add_weight(
                name='bias', 
                shape=(out_channel,), 
                initializer='zeros', 
                trainable=True
            )
        else:
            self.bias = None

    def call(self, input):
        print(f"EqualConv2d - input shape: {input.shape}")
        weight = self.weight * self.scale
        out = tf.nn.conv2d(input, weight, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        
        if self.use_bias:
            out = out + self.bias

        print(f"EqualConv2d - output shape: {out.shape}")
        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[2]}, {self.weight.shape[3]},'
            f' {self.kernel_size}, stride={self.stride}, padding={self.padding})'
        )

    

class EqualLinear(tf.keras.layers.Layer):
    def __init__(self, in_dim, out_dim, use_bias=True, lr_mul=1, activation=None):
        super(EqualLinear, self).__init__()
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.use_bias = use_bias
        self.linear = layers.Dense(out_dim, use_bias=use_bias, kernel_initializer='random_normal')

    def build(self, input_shape):
        super(EqualLinear, self).build(input_shape)
        self.linear.kernel.assign(self.linear.kernel * self.scale)

    def call(self, input):
        out = self.linear(input)
        if self.activation == 'fused_lrelu':
            out = fused_leaky_relu(out, 0)
        return out

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], use_bias=True, activate=True):
        super(ConvLayer, self).__init__()
        self.downsample = downsample
        self.activate = activate
        self.use_bias = use_bias

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = Blur(blur_kernel, pad=(pad0, pad1))
            stride = 2
            padding = 'VALID'
        else:
            stride = 1
            padding = 'SAME'

        self.conv = EqualConv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=use_bias and not activate)
        
        if activate:
            if use_bias:
                self.activation = FusedLeakyReLU(out_channel)
            else:
                self.activation = ScaledLeakyReLU(0.2)

    def call(self, input):
        print(f"ConvLayer - input shape: {input.shape}")
        if self.downsample:
            input = self.blur(input)
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)
        print(f"ConvLayer - output shape: {out.shape}")
        return out
  


    
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, use_bias=False)

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
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        self.noise = NoiseInjection()
        self.activation = FusedLeakyReLU(out_channel)

    def call(self, input, style, noise=None):
        print(f"StyledConv - input shape: {input.shape}, style shape: {style.shape}")
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activation(out)
        print(f"StyledConv - output shape: {out.shape}")
        return out
class ModulatedConv2d(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super(ModulatedConv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample
        self.demodulate = demodulate
        self.eps = 1e-8
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        if upsample:
            self.blur = Blur(blur_kernel, pad=(self.padding, self.padding))

        if downsample:
            self.blur = Blur(blur_kernel, pad=(self.padding, self.padding))

        self.weight = self.add_weight(shape=(1, out_channel, in_channel, kernel_size, kernel_size), initializer='random_normal', trainable=True)
        self.modulation = EqualLinear(style_dim, in_channel, use_bias=True, activation=None)

    def call(self, input, style):
        batch, height, width, in_channel = tf.shape(input)
        style = self.modulation(style)
        style = tf.reshape(style, [batch, 1, in_channel, 1, 1])

        weight = self.scale * self.weight * style
        if self.demodulate:
            demod = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[2, 3, 4]) + self.eps)
            weight = weight * tf.reshape(demod, [batch, self.out_channel, 1, 1, 1])

        weight = tf.reshape(weight, [batch * self.out_channel, self.in_channel, self.kernel_size, self.kernel_size])

        input = tf.reshape(input, [1, batch * in_channel, height, width])

        if self.upsample:
            weight = tf.transpose(weight, [0, 1, 3, 2])  # Transpose for depthwise_conv2d_transpose
            out = tf.nn.depthwise_conv2d_transpose(
                input, weight, output_shape=[batch, self.out_channel, height * 2, width * 2], strides=[1, 2, 2, 1], padding='SAME'
            )
            out = tf.reshape(out, [batch, height * 2, width * 2, self.out_channel])
            out = self.blur(out)
        elif self.downsample:
            input = self.blur(input)
            out = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='VALID')
            out = tf.reshape(out, [batch, height // 2, width // 2, self.out_channel])
        else:
            out = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME')
            out = tf.reshape(out, [batch, height, width, self.out_channel])

        return out

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
