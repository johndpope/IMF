import tensorflow as tf
import numpy as np

class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, up=False, down=False, demodulate=True, fused_modconv=True, gain=1.0, lrmul=1.0, **kwargs):
        super(ModulatedConv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.style_fmaps = style_fmaps
        self.kernel = kernel
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.fused_modconv = fused_modconv
        self.gain = gain
        self.lrmul = lrmul

        self.mod_dense = tf.keras.layers.Dense(self.fmaps, use_bias=True, kernel_initializer='he_uniform')
      
    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        in_fmaps = x_shape[1]  # Assuming NCHW format
        weight_shape = [self.kernel, self.kernel, in_fmaps, self.fmaps]
        init_std = 1.0 / self.lrmul
        self.w = self.add_weight(
            name='w',
            shape=weight_shape,
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=init_std),
            trainable=True
        )

    def scale_conv_weights(self, w):
        weight = self.w * (self.gain / (self.kernel * self.kernel * tf.cast(tf.shape(self.w)[2], tf.float32)) ** 0.5)

        style = self.mod_dense(w)  # [batch_size, fmaps]
        style = tf.reshape(style, [-1, 1, 1, self.fmaps])  # [batch_size, 1, 1, fmaps]
        
        weight = weight[tf.newaxis]  # [1, kernel, kernel, in_fmaps, out_fmaps]
        weight *= style[:, tf.newaxis, tf.newaxis, tf.newaxis, :]  # [batch_size, kernel, kernel, in_fmaps, out_fmaps]

        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)
            weight *= d[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        return weight

    def call(self, inputs):
        x, y = inputs
        weight = self.scale_conv_weights(y)

        if self.up:
            x = tf.nn.depth_to_space(x, 2)
        
        if self.fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
            w = tf.reshape(weight, [-1, weight.shape[2], weight.shape[3], weight.shape[4]])
            x = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')
            x = tf.reshape(x, [-1, self.fmaps, x.shape[2], x.shape[3]])
        else:
            x *= tf.cast(weight[:, :, 0, 0], x.dtype)
            x = tf.nn.conv2d(x, weight[0], strides=[1,1,1,1], padding='SAME', data_format='NCHW')

        if self.down:
            x = tf.nn.avg_pool2d(x, ksize=[1,1,2,2], strides=[1,1,2,2], padding='VALID', data_format='NCHW')

        return x

class StyledConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, up=False, down=False, demodulate=True, fused_modconv=True, gain=1.0, lrmul=1.0, **kwargs):
        super(StyledConv, self).__init__(**kwargs)
        self.conv = ModulatedConv2D(fmaps, style_fmaps, kernel, up, down, demodulate, fused_modconv, gain, lrmul)
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=lrmul, act='lrelu', name='bias')

    def call(self, inputs):
        x, y = inputs
        x = self.conv([x, y])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)
        return x

class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = self.add_weight(name='w', shape=(), initializer='zeros', trainable=True)

    def call(self, inputs):
        x_shape = tf.shape(inputs)
        noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=inputs.dtype)
        return inputs + noise * self.noise_strength

class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        self.lrmul = lrmul
        self.act = act

    def build(self, input_shape):
        self.b = self.add_weight(name='b', shape=(input_shape[1],), initializer='zeros', trainable=True)

    def call(self, inputs):
        b = self.lrmul * self.b
        x = tf.nn.bias_add(inputs, b, data_format='NCHW')
        if self.act == 'linear':
            return x
        elif self.act == 'lrelu':
            return tf.nn.leaky_relu(x, alpha=0.2)
        else:
            raise ValueError(f"Unsupported activation: {self.act}")

def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])
    he_std = gain / np.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef