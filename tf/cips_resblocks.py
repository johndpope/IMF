from tf.ops import *

class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, resample_kernel, up, down, demodulate, fused_modconv, gain, lrmul, **kwargs):
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

        self.k, self.pad0, self.pad1 = compute_paddings(resample_kernel, self.kernel, up, down, is_conv=True)

        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = BiasAct(lrmul=1.0, act='linear', name='mod_bias')

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        in_fmaps = x_shape[1]
        weight_shape = [self.kernel, self.kernel, in_fmaps, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        weight = self.runtime_coef * self.w
        weight = weight[np.newaxis]

        style = self.mod_dense(w)
        style = self.mod_bias(style) + 1.0
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]

        d = None
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]

        return weight, style, d

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        weight, style, d = self.scale_conv_weights(y)

        if self.fused_modconv:
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
            new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]
            weight = tf.transpose(weight, [1, 2, 3, 0, 4])
            weight = tf.reshape(weight, shape=new_weight_shape)
        else:
            x *= style[:, :, tf.newaxis, tf.newaxis]

        if self.up:
            x = upsample_conv_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        elif self.down:
            x = conv_downsample_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        if self.fused_modconv:
            x_shape = tf.shape(x)
            x = tf.reshape(x, [-1, self.fmaps, x_shape[2], x_shape[3]])
        elif self.demodulate:
            x *= d[:, :, tf.newaxis, tf.newaxis]

        return x

class StyledConv(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, resample_kernel=(1, 3, 3, 1), up=False, down=False, demodulate=True, fused_modconv=True, gain=1.0, lrmul=1.0, **kwargs):
        super(StyledConv, self).__init__(**kwargs)
        resample_kernel = list(resample_kernel)
        self.conv = ModulatedConv2D(fmaps, style_fmaps, kernel, resample_kernel, up, down, demodulate, fused_modconv, gain, lrmul)
        self.apply_noise = Noise(name='noise')
        self.apply_bias_act = BiasAct(lrmul=lrmul, act='lrelu', name='bias')

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        x = self.conv([x, y])
        x = self.apply_noise(x)
        x = self.apply_bias_act(x)

        return x

class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name='w')

    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.float32)

        x = inputs + noise * self.noise_strength
        return x

class BiasAct(tf.keras.layers.Layer):
    def __init__(self, lrmul, act, **kwargs):
        super(BiasAct, self).__init__(**kwargs)
        self.lrmul = lrmul
        self.act = act

    def build(self, input_shape):
        b_init = tf.zeros(shape=(input_shape[1],), dtype=tf.float32)
        self.b = tf.Variable(b_init, name='b', trainable=True)

    def call(self, inputs, training=None, mask=None):
        b = self.lrmul * self.b
        x = fused_bias_act(inputs, b=b, act=self.act, alpha=None, gain=None)
        return x

def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = tf.reduce_prod(weight_shape[:-1])
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef
