from tf.cuda.upfirdn_2d import *
from tf.cuda.fused_bias_act import fused_bias_act

##################################################################################
# Layers
##################################################################################

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, kernel, resample_kernel, up, down, gain, lrmul, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.kernel = kernel
        self.gain = gain
        self.lrmul = lrmul
        self.up = up
        self.down = down

        self.k, self.pad0, self.pad1 = compute_paddings(resample_kernel, self.kernel, up, down, is_conv=True)

    def build(self, input_shape):
        weight_shape = [self.kernel, self.kernel, input_shape[1], self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kernel, kernel, fmaps_in, fmaps_out]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        w = self.runtime_coef * self.w

        # actual conv
        if self.up:
            x = upsample_conv_2d(x, w, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        elif self.down:
            x = conv_downsample_2d(x, w, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        else:
            x = tf.nn.conv2d(x, w, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')
        return x

class ModulatedConv2D(tf.keras.layers.Layer):
    def __init__(self, fmaps, style_fmaps, kernel, resample_kernel, up, down, demodulate, fused_modconv, gain, lrmul, **kwargs):
        super(ModulatedConv2D, self).__init__(**kwargs)
        assert not (up and down)

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

        # self.factor = 2
        self.mod_dense = Dense(self.style_fmaps, gain=1.0, lrmul=1.0, name='mod_dense')
        self.mod_bias = BiasAct(lrmul=1.0, act='linear', name='mod_bias')

    def build(self, input_shape):
        x_shape, w_shape = input_shape[0], input_shape[1]
        in_fmaps = x_shape[1]
        weight_shape = [self.kernel, self.kernel, in_fmaps, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        # [kkIO]
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def scale_conv_weights(self, w):
        # convolution kernel weights for fused conv
        weight = self.runtime_coef * self.w  # [kkIO]
        weight = weight[np.newaxis]  # [BkkIO]

        # modulation
        style = self.mod_dense(w)  # [BI]
        style = self.mod_bias(style) + 1.0  # [BI]
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]  # [BkkIO]

        # demodulation
        d = None
        if self.demodulate:
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO]

        return weight, style, d

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        # height, width = tf.shape(x)[2], tf.shape(x)[3]

        # prepare weights: [BkkIO] Introduce minibatch dimension
        # prepare convoultuon kernel weights
        weight, style, d = self.scale_conv_weights(y)

        if self.fused_modconv:
            # Fused => reshape minibatch to convolution groups
            x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])

            # weight: reshape, prepare for fused operation
            new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]  # [kkI(BO)]
            weight = tf.transpose(weight, [1, 2, 3, 0, 4])  # [kkIBO]
            weight = tf.reshape(weight, shape=new_weight_shape)  # [kkI(BO)]
        else:
            # [BIhw] Not fused => scale input activations
            x *= style[:, :, tf.newaxis, tf.newaxis]

        # Convolution with optional up/downsampling.
        if self.up:
            x = upsample_conv_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        elif self.down:
            x = conv_downsample_2d(x, weight, self.kernel, self.kernel, self.pad0, self.pad1, self.k)
        else:
            x = tf.nn.conv2d(x, weight, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # Reshape/scale output
        if self.fused_modconv:
            # Fused => reshape convolution groups back to minibatch
            x_shape = tf.shape(x)
            x = tf.reshape(x, [-1, self.fmaps, x_shape[2], x_shape[3]])
        elif self.demodulate:
            # [BOhw] Not fused => scale output activations
            x *= d[:, :, tf.newaxis, tf.newaxis]

        return x


class Dense(tf.keras.layers.Layer):
    def __init__(self, fmaps, gain=1.0, lrmul=1.0, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.fmaps = fmaps
        self.gain = gain
        self.lrmul = lrmul

    def build(self, input_shape):
        fan_in = tf.reduce_prod(input_shape[1:])
        weight_shape = [fan_in, self.fmaps]
        init_std, self.runtime_coef = compute_runtime_coef(weight_shape, self.gain, self.lrmul)

        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=init_std)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        weight = self.runtime_coef * self.w

        c = tf.reduce_prod(tf.shape(inputs)[1:])
        x = tf.reshape(inputs, shape=[-1, c])
        x = tf.matmul(x, weight)
        return x

class LabelEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super(LabelEmbedding, self).__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        weight_shape = [input_shape[1], self.embed_dim]
        # tf 1.15 mean(0.0), std(1.0) default value of tf.initializers.random_normal()
        w_init = tf.random.normal(shape=weight_shape, mean=0.0, stddev=1.0)
        self.w = tf.Variable(w_init, name='w', trainable=True)

    def call(self, inputs, training=None, mask=None):
        x = tf.matmul(inputs, self.w)
        return x

##################################################################################
# etc
##################################################################################
class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, inputs, training=None, mask=None):
        x = inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True) + 1e-8)
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

class Noise(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Noise, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_strength = tf.Variable(initial_value=0.0, dtype=tf.float32, trainable=True, name='w')


    def call(self, inputs, noise=None, training=None, mask=None):
        x_shape = tf.shape(inputs)

        # noise: [1, 1, x_shape[2], x_shape[3]] or None
        if noise is None:
            noise = tf.random.normal(shape=(x_shape[0], 1, x_shape[2], x_shape[3]), dtype=tf.float32)

        x = inputs + noise * self.noise_strength
        return x

class MinibatchStd(tf.keras.layers.Layer):
    def __init__(self, group_size, num_new_features, **kwargs):
        super(MinibatchStd, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_new_features = num_new_features

    def call(self, inputs, training=None, mask=None):
        s = tf.shape(inputs)
        group_size = tf.minimum(self.group_size, s[0])

        y = tf.reshape(inputs, [group_size, -1, self.num_new_features, s[1] // self.num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, inputs.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])

        x = tf.concat([inputs, y], axis=1)
        return x

def compute_runtime_coef(weight_shape, gain, lrmul):
    fan_in = tf.reduce_prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    fan_in = tf.cast(fan_in, dtype=tf.float32)
    he_std = gain / tf.sqrt(fan_in)
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul
    return init_std, runtime_coef

def lerp(a, b, t):
    out = a + (b - a) * t
    return out

def lerp_clip(a, b, t):
    out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out


def get_coords(batch_size, height, width):

    x = tf.linspace(-1, 1, width)
    x = tf.reshape(x, shape=[1, 1, width, 1])
    x = tf.tile(x, multiples=[batch_size, width, 1, 1])

    y = tf.linspace(-1, 1, height)
    y = tf.reshape(y, shape=[1, height, 1, 1])
    y = tf.tile(y, multiples=[batch_size, 1, height, 1])

    coords = tf.concat([x, y], axis=-1)
    coords = tf.transpose(coords, perm=[0, 3, 1, 2])
    coords = tf.cast(coords, tf.float32)

    return coords

def grid_sample_tf(img, coords, align_corners=False, padding='border'):
    """

    :param img: [B, C, H, W]
    :param coords: [B, C, H, W]
    :return: [B, C, H, W]
    """
    def get_pixel_value(img, x, y):
        """
        Utility function to get pixel value for coordinate
        vectors x and y from a  4D tensor image.
        Input
        -----
        - img: tensor of shape (B, H, W, C)
        - x: flattened tensor of shape (B*H*W,)
        - y: flattened tensor of shape (B*H*W,)
        Returns
        -------
        - output: tensor of shape (B, H, W, C)
        """
        shape = tf.shape(x)
        batch_size = shape[0]
        height = shape[1]
        width = shape[2]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, height, width))

        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(img, indices)

    # rescale x and y to [0, W-1/H-1]
    img = tf.transpose(img, perm=[0, 2, 3, 1]) # -> [N, H, W, C]
    coords = tf.transpose(coords, perm=[0, 2, 3, 1]) # -> [N, H, W, C]

    x, y = coords[:, ..., 0], coords[:, ..., 1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    side = tf.cast(tf.shape(img)[1], tf.int32)
    side_f = tf.cast(side, tf.float32)

    if align_corners:
        x = ((x + 1) / 2) * (side_f - 1)
        y = ((y + 1) / 2) * (side_f - 1)
    else:
        x = 0.5 * ((x + 1.0) * side_f - 1)
        y = 0.5 * ((y + 1.0) * side_f - 1)

    if padding == 'border':
        x = tf.clip_by_value(x, 0, side_f - 1)
        y = tf.clip_by_value(y, 0, side_f - 1)

    # -------------- Changes above --------------------
    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.floor(x)
    x1 = x0 + 1
    y0 = tf.floor(y)
    y1 = y0 + 1

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # recast as int for img boundaries
    x0 = tf.cast(x0, 'int32')
    x1 = tf.cast(x1, 'int32')
    y0 = tf.cast(y0, 'int32')
    y1 = tf.cast(y1, 'int32')

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, side-1)
    x1 = tf.clip_by_value(x1, 0, side-1)
    y0 = tf.clip_by_value(y0, 0, side-1)
    y1 = tf.clip_by_value(y1, 0, side-1)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id # [N, H, W, C]
    out = tf.transpose(out, perm=[0, 3, 1, 2])

    return out

def torch_normalization(x):
    x /= 255.

    r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=x)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    x = tf.concat(axis=-1, values=[
        (r - mean[0]) / std[0],
        (g - mean[1]) / std[1],
        (b - mean[2]) / std[2]
    ])

    return x


def inception_processing(filename):
    x = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
    img = tf.image.resize(img, [256, 256], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
    img = tf.image.resize(img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

    img = torch_normalization(img)
    # img = tf.transpose(img, [2, 0, 1])
    return img