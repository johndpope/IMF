from tf.vit import ImplicitMotionAlignment
from tf.resblocks import  FeatResBlock,UpConvResBlock,DownConvResBlock
from tf.cips_resblocks import StyledConv
from tf.lia_resblocks import EqualConv2d,EqualLinear,ResBlock ,PyConv2D# these are correct https://github.com/hologerry/IMF/issues/4  "You can refer to this repo https://github.com/wyhsirius/LIA/ for StyleGAN2 related code, such as Encoder, Decoder."

# from torchvision.models import efficientnetB0, EfficientNet_B0_Weights
import math
# from common import DownConvResBlock,UpConvResBlock
# import coloredTraceback.auto # makes terminal show color coded output when crash
import random

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import math

DEBUG = True
def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def normalize(x):
    return (x - tf.reduce_min(x)) / (tf.reduce_max(x) - tf.reduce_min(x) + 1e-8)


# keep everything in 1 class to allow copying / pasting into claude / chatgpt

class SpectralNormalization(layers.Wrapper):
    def __init__(self, layer, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.layer = layer

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(shape=(1, self.w.shape[-1]),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=False,
                                 name='sn_u')

    def call(self, inputs):
        w_shape = tf.shape(self.w)
        w_reshaped = tf.reshape(self.w, [-1, w_shape[-1]])
        u_hat = self.u
        v_hat = tf.math.l2_normalize(tf.matmul(u_hat, tf.transpose(w_reshaped)))
        u_hat = tf.math.l2_normalize(tf.matmul(v_hat, w_reshaped))
        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        w_norm = self.w / sigma
        self.u.assign(u_hat)
        self.layer.kernel = w_norm
        return self.layer(inputs)


'''
DenseFeatureEncoder
It starts with a conv-64-k7-s1-p3 layer, followed by BatchNorm and relu, as shown in the first block of the image.
It then uses a series of DownConvResBlocks, which perform downsampling (indicated by ↓2 in the image) while increasing the number of channels:

DownConvResBlock-64
DownConvResBlock-128
DownConvResBlock-256
DownConvResBlock-512
DownConvResBlock-512


It outputs multiple feature maps (f¹ᵣ, f²ᵣ, f³ᵣ, f⁴ᵣ) as shown in the image. These are collected in the features list and returned.

Each DownConvResBlock performs downsampling using a strided convolution, maintains a residual connection, and applies BatchNorm and relu activations, which is consistent with typical ResNet architectures.'''

class InitialConvBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, initial_channels):
        super(InitialConvBlock, self).__init__()
        
        self.conv = tf.keras.layers.Conv2D(
            filters=initial_channels,
            kernel_size=7,
            strides=1,
            padding='same',  # 'same' padding in TF is equivalent to padding=3 in PyTorch for a 7x7 kernel
            data_format='channels_first',
            use_bias=False  # typically, bias is not used when followed by BatchNorm
        )
        
        self.bn = tf.keras.layers.BatchNormalization(axis=1)  # axis=1 for channels_first
        
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DenseFeatureEncoder(tf.keras.Model):
    def __init__(self, in_channels=3, output_channels=[128, 256, 512, 512], initial_channels=64):
        super(DenseFeatureEncoder, self).__init__()
        self.initial_conv = InitialConvBlock(in_channels, initial_channels)

        self.down_blocks = []
        current_channels = initial_channels

        # Add an initial down block that doesn't change the number of channels
        self.down_blocks.append(DownConvResBlock(current_channels, current_channels))
        
        # Add down blocks for each specified output channel
        for out_channels in output_channels:
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

    def call(self, x):
        debug_print(f"⚾  DenseFeatureEncoder input shape: {x.shape}")
        features = []
        x = self.initial_conv(x)
        debug_print(f"After initial conv: {x.shape}")
        
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            debug_print(f"After down_block {i+1}: {x.shape}")
            if i >= 1:  # Start collecting features from the second block
                features.append(x)
        
        debug_print(f"DenseFeatureEncoder output shapes: {[f.shape for f in features]}")
        return features

'''
The upSampling parameter is replaced with downsample to match the diagram.
The first convolution now has a stride of 2 when downsampling.
The shortcut connection now uses a 3x3 convolution with stride 2 when downsampling, instead of a 1x1 convolution.
relu activations are applied both after adding the residual and at the end of the block.
The FeatResBlock is now a subclass of ResBlock with downsample=False, as it doesn't change the spatial dimensions.
'''
class LatentTokenEncoder(tf.keras.Model):
    def __init__(self, initial_channels=64, output_channels=[256, 256, 512, 512, 512, 512], dm=32):
        super(LatentTokenEncoder, self).__init__()

        self.conv1 = PyConv2D(initial_channels, kernel_size=3, strides=1, padding='same', data_format='channels_first')
        self.activation = layers.LeakyReLU(0.2)

        self.res_blocks = []
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.equalconv = EqualConv2d(
            filters=output_channels[-1],
            kernel_size=3,
            strides=1,
            padding=1
        )
        self.linear_layers = [EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)]
        
        # self.final_linear = EqualLinear(output_channels[-1], dm)  # This should output dm (32) channels
        self.final_linear = EqualLinear(dm)  
    def call(self, x):
        debug_print(f"🥊 LatentTokenEncoder input shape: {x.shape}")

        x = self.activation(self.conv1(x))
        debug_print(f"After initial conv and activation: {x.shape}")
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            debug_print(f"After res_block {i+1}: {x.shape}")
        
        x = self.equalconv(x)
        debug_print(f"After equalconv: {x.shape}")
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=[2, 3])
        debug_print(f"After global average pooling: {x.shape}")
        
        for i, linear_layer in enumerate(self.linear_layers):
            x = self.activation(linear_layer(x))
            debug_print(f"After linear layer {i+1}: {x.shape}")
        
        x = self.final_linear(x)
        debug_print(f"Final output: {x.shape}")
        
        return x

'''
LatentTokenDecoder
It starts with a constant input (self.const), as shown at the top of the image.
It uses a series of StyleConv layers, which apply style modulation based on the input latent token t.
Some StyleConv layers perform upsampling (indicated by ↑2 in the image).
The network maintains 512 channels for most of the layers, switching to 256 channels for the final three layers.
It outputs multiple feature maps (m⁴, m³, m², m¹) as shown in the image. These are collected in the features list and returned in reverse order to match the notation in the image.




Latent Token Decoder
Transforms compact tokens (tr, tc) into multiple motion features (mlr and mlc) with specific dimensions, where "l" indicates the layer index.

Style Modulation:

Uses a technique from StyleGAN2 called Weight Modulation and Demodulation.
This technique scales convolution weights with the latent token and normalizes them to have a unit standard deviation.
Benefits:

The latent tokens compress multi-scale information, making them comprehensive representations.
Our representation is fully implicit, allowing flexible adjustment of the latent token dimension for different scenarios.
Advantages Over Keypoint-Based Methods:

Unlike keypoint-based methods that use Gaussian heatmaps converted from keypoints, our design scales better and has more capabilities.
Our latent tokens are directly learned by the encoder, rather than being restricted to coordinates with a limited value range.
'''
import tensorflow as tf

class LatentTokenDecoder(tf.keras.Model):
    def __init__(self, latent_dim=32, const_dim=32):
        super(LatentTokenDecoder, self).__init__()
        print(f"Initializing LatentTokenDecoder with latent_dim={latent_dim}, const_dim={const_dim}")
        
        # Initialize a constant input with shape (1, const_dim, 4, 4)
        self.constant = self.add_weight(shape=(1, const_dim, 4, 4), initializer='random_normal', trainable=True)
        print(f"Constant input shape: {self.constant.shape}")

        self.style_conv_layers = [
            StyledConv(const_dim, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim, upsample=True),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 512, 3, latent_dim),
            StyledConv(512, 256, 3, latent_dim, upsample=True),
            StyledConv(256, 256, 3, latent_dim),
            StyledConv(256, 256, 3, latent_dim)
        ]
        print(f"Number of StyledConv layers: {len(self.style_conv_layers)}")

    def call(self, t):
        print(f"🍩 LatentTokenDecoder call method input shape: {t.shape}")
        
        batch_size = tf.shape(t)[0]
        x = tf.tile(self.constant, [batch_size, 1, 1, 1])
        print(f"Tiled constant input shape: {x.shape}")
        
        m1, m2, m3, m4 = None, None, None, None
        for i, layer in enumerate(self.style_conv_layers):
            x = layer(x, t)
            print(f"Layer {i} output shape: {x.shape}")
            
            if i == 3:
                m1 = x
                print(f"m1 shape: {m1.shape}")
            elif i == 6:
                m2 = x
                print(f"m2 shape: {m2.shape}")
            elif i == 9:
                m3 = x
                print(f"m3 shape: {m3.shape}")
            elif i == 12:
                m4 = x
                print(f"m4 shape: {m4.shape}")
        
        print("LatentTokenDecoder call method completed")
        return m4, m3, m2, m1

    
'''
FrameDecoder FeatResBlock
It uses a series of UpConvResBlock layers that perform upsampling (indicated by ↑2 in the image).
It incorporates FeatResBlock layers, with 3 blocks for each of the first three levels (512, 512, and 256 channels).
It uses concatenation (Concat in the image) to combine the upsampled features with the processed input features.
The channel dimensions decrease as we go up the network: 512 → 512 → 256 → 128 → 64.
It ends with a final convolutional layer (conv-3-k3-s1-p1) followed by a Sigmoid activation.
'''
class FrameDecoder(tf.keras.Model):
    def __init__(self, gradient_scale=1):
        super(FrameDecoder, self).__init__()
        
        self.upconv_blocks = [
            UpConvResBlock(512, 512),
            UpConvResBlock(1024, 512),
            UpConvResBlock(768, 256),
            UpConvResBlock(384, 128),
            UpConvResBlock(128, 64)
        ]
        
        self.feat_blocks = [
            tf.keras.Sequential([FeatResBlock(512) for _ in range(3)]),
            tf.keras.Sequential([FeatResBlock(256) for _ in range(3)]),
            tf.keras.Sequential([FeatResBlock(128) for _ in range(3)])
        ]
        
        self.final_conv = tf.keras.Sequential([
            PyConv2D(3, kernel_size=3, strides=1, padding='same'),
            layers.Activation('sigmoid')
        ])

    def call(self, features):
        debug_print(f"🎒  FrameDecoder input shapes")
        for f in features:
            debug_print(f"f:{f.shape}")
        x = features[-1]
        debug_print(f"Initial x shape: {x.shape}")
        
        for i in range(len(self.upconv_blocks)):
            debug_print(f"\nProcessing upconvBlock {i+1}")
            x = self.upconv_blocks[i](x)
            debug_print(f"After upconvBlock {i+1}: {x.shape}")
            
            if i < len(self.feat_blocks):
                debug_print(f"Processing featBlock {i+1}")
                feat_input = features[-(i+2)]
                debug_print(f"featBlock {i+1} input shape: {feat_input.shape}")
                feat = self.feat_blocks[i](feat_input)
                debug_print(f"featBlock {i+1} output shape: {feat.shape}")
                
                debug_print(f"Concatenating: x {x.shape} and feat {feat.shape}")
                x = tf.concat([x, feat], axis=-1)
                debug_print(f"After concatenation: {x.shape}")
        
        debug_print("\nApplying final convolution")
        x = self.final_conv(x)
        debug_print(f"FrameDecoder final output shape: {x.shape}")

        return x

class TokenManipulationNetwork(tf.keras.Model):
    def __init__(self, token_dim, condition_dim, hidden_dim=256):
        super(TokenManipulationNetwork, self).__init__()
        self.token_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])
        self.condition_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(hidden_dim)
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation='relu'),
            tf.keras.layers.Dense(token_dim)
        ])

    def call(self, token, condition):
        token_encoded = self.token_encoder(token)
        condition_encoded = self.condition_encoder(condition)
        combined = tf.concat([token_encoded, condition_encoded], axis=-1)
        edited_token = self.decoder(combined)
        return edited_token


'''
DenseFeatureEncoder (EF): Encodes the reference frame into multi-scale features.
LatentTokenEncoder (ET): Encodes both the current and reference frames into latent tokens.
LatentTokenDecoder (IMFD): Decodes the latent tokens into motion features.
ImplicitMotionAlignment (IMFA): Aligns the reference features to the current frame using the motion features.

The call pass:

Encodes the reference frame using the dense feature encoder.
Encodes both current and reference frames into latent tokens.
Decodes the latent tokens into motion features.
For each scale, aligns the reference features to the current frame using the ImplicitMotionAlignment module.
'''
class IMFModel(tf.keras.Model):
    def __init__(self, use_resnet_feature=False, use_mlgffn=False, use_skip=False, use_enhanced_generator=False, latent_dim=32, base_channels=64, num_layers=4, noise_level=0.1, style_mix_prob=0.5):
        super(IMFModel, self).__init__()
        
        self.latent_token_encoder = LatentTokenEncoder(
            initial_channels=64,
            output_channels=[256, 256, 512, 512, 512, 512],
            dm=32
        )
        
        self.latent_token_decoder = LatentTokenDecoder()
        self.feature_dims = [128, 256, 512, 512]
        self.spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8)]

        self.dense_feature_encoder = DenseFeatureEncoder(output_channels=self.feature_dims)

        self.motion_dims = [256, 512, 512, 512]
        
        # Initialize ImplicitMotionAlignment modules
        self.implicit_motion_alignment = []
        for i in range(num_layers):
            feature_dim = self.feature_dims[i]
            motion_dim = self.motion_dims[i]
            spatial_dim = self.spatial_dims[i]
            alignment_module = ImplicitMotionAlignment(feature_dim=feature_dim, motion_dim=motion_dim, spatial_dim=spatial_dim)
            self.implicit_motion_alignment.append(alignment_module)
        
        self.frame_decoder = FrameDecoder()
        self.noise_level = noise_level
        self.style_mix_prob = style_mix_prob

    def call(self, x_current, x_reference):

        fR = self.dense_feature_encoder(x_reference)
        tR = self.latent_token_encoder(x_reference)
        tC = self.latent_token_encoder(x_current)

        # Style mixing
        # if tf.random.uniform([]) < self.style_mix_prob:
        #     batch_size = tf.shape(tC)[0]
        #     rand_indices = tf.random.shuffle(tf.range(batch_size))
        #     rand_tC = tf.gather(tC, rand_indices)
        #     rand_tR = tf.gather(tR, rand_indices)
        #     mix_mask = tf.random.uniform([batch_size, 1]) < 0.5
        #     mix_mask = tf.cast(mix_mask, tf.float32)
        #     mix_tC = tC * mix_mask + rand_tC * (1 - mix_mask)
        #     mix_tR = tR * mix_mask + rand_tR * (1 - mix_mask)
        # else:
        mix_tC = tC
        mix_tR = tR

        mC = self.latent_token_decoder(mix_tC)
        mR = self.latent_token_decoder(mix_tR)

        aligned_features = []
        for i in range(len(self.implicit_motion_alignment)):
            fRI = fR[i]
            align_layer = self.implicit_motion_alignment[i]
            mCI = mC[i] 
            mRI = mR[i]
            aligned_feature = align_layer(mCI, mRI, fRI)
            aligned_features.append(aligned_feature)

        x_reconstructed = self.frame_decoder(aligned_features)
        x_reconstructed = normalize(x_reconstructed)
        return x_reconstructed




            



'''
PatchDiscriminator
This implementation of the PatchDiscriminator class follows the architecture described in the supplementary material of the IMF paper. Here are the key features:

Multi-scale: The discriminator operates on two scales. The first scale (scale1) processes the input at its original resolution, while the second scale (scale2) processes a downsampled version of the input.
Spectral Normalization: We use spectral normalization on all convolutional layers to stabilize training, as indicated by the "SN" in the paper's diagram.
Architecture: Each scale follows the structure described in the paper:

4x4 convolutions with stride 2
leakyRelu activation (α=0.2)
Instance Normalization (except for the first and last layers)
Channel progression: 64 -> 128 -> 256 -> 512 -> 1


Output: The call method returns a list containing the outputs from both scales.
Weight Initialization: A helper function initWeights is provided to initialize the weights of the network, which can be applied using the apply method.
'''

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class IMFPatchDiscriminator(tf.keras.Model):
    def __init__(self, input_nc=3, ndf=64):
        super(IMFPatchDiscriminator, self).__init__()
        
        def conv_block(filters, kernel_size=4, strides=2, use_instance_norm=True):
            block = [
                SpectralNormalization(PyConv2D(filters, kernel_size, strides=strides, padding='same')),
                layers.LeakyReLU(0.2)
            ]
            if use_instance_norm:
                block.insert(1, InstanceNormalization())
            return block

        self.scale1 = tf.keras.Sequential([
            *conv_block(ndf, use_instance_norm=False),
            *conv_block(ndf * 2),
            *conv_block(ndf * 4),
            *conv_block(ndf * 8),
            SpectralNormalization(PyConv2D(1, kernel_size=1, strides=1))
        ])
        
        self.scale2 = tf.keras.Sequential([
            *conv_block(ndf, use_instance_norm=False),
            *conv_block(ndf * 2),
            *conv_block(ndf * 4),
            *conv_block(ndf * 8),
            SpectralNormalization(PyConv2D(1, kernel_size=1, strides=1))
        ])

    def call(self, x):
        # Scale 1
        output1 = self.scale1(x)
        
        # Scale 2
        x_downsampled = tf.image.resize(x, [tf.shape(x)[1] // 2, tf.shape(x)[2] // 2], method='bilinear')
        output2 = self.scale2(x_downsampled)
        
        return [output1, output2]


# Helper function to initialize weights
def initWeights(m):
    classname = m._Class__._Name__
    if classname.find('conv') != -1:
        tf.keras.layers.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        tf.keras.layers.init.normal_(m.weight.data, 1.0, 0.02)
        tf.keras.layers.init.constant_(m.bias.data, 0)



        

# class ConvBlock(tf.keras.layers.Module):
#     def _Init__(self, inChannels, outChannels, kernelSize=4, stride=2, padding=1, useInstanceNorm=True):
#         super(ConvBlock, self)._Init__()
#         layers = [
#             spectralNorm(tf.keras.layers.Conv2d(inChannels, outChannels, kernelSize, stride, padding)),
#             tf.keras.layers.leakyRelu(0.2, inplace=True)
#         ]
#         if useInstanceNorm:
#             layers.insert(1, tf.keras.layers.InstanceNorm2d(outChannels))
#         self.block = tf.keras.layers.Sequential(*layers)

#     def call(self, x):
#         return self.block(x)

# class PatchDiscriminator(tf.keras.layers.Module):
#     def _Init__(self, inputNc, ndf=64, nLayers=3):
#         super(PatchDiscriminator, self)._Init__()
#         sequence = [ConvBlock(inputNc, ndf, useInstanceNorm=False)]
        
#         nfMult = 1
#         for n in range(1, nLayers):
#             nfMultPrev = nfMult
#             nfMult = min(2 ** n, 8)
#             sequence += [ConvBlock(ndf * nfMultPrev, ndf * nfMult)]

#         sequence += [
#             ConvBlock(ndf * nfMult, ndf * nfMult, stride=1),
#             spectralNorm(tf.keras.layers.Conv2d(ndf * nfMult, 1, kernelSize=4, stride=1, padding=1))
#         ]

#         self.model = tf.keras.layers.Sequential(*sequence)

#     def call(self, x):
#         return self.model(x)

# class MultiScalePatchDiscriminator(tf.keras.layers.Module):
#     def _Init__(self, inputNc, ndf=64, nLayers=3, num_D=3):
#         super(MultiScalePatchDiscriminator, self)._Init__()
#         self.num_D = num_D
#         self.nLayers = nLayers

#         for i in range(num_D):
#             subnetD = PatchDiscriminator(inputNc, ndf, nLayers)
#             setattr(self, f'scale_{i}', subnetD)

#     def singleDcall(self, model, input):
#         return model(input)

#     def call(self, input):
#         result = []
#         inputDownsampled = input
#         for i in range(self.num_D):
#             model = getattr(self, f'scale_{i}')
#             result.append(self.singleDcall(model, inputDownsampled))
#             if i != self.num_D - 1:
#                 inputDownsampled = tf.nn.avgPool2d(inputDownsampled, kernelSize=3, stride=2, padding=1, countIncludePad=False)
#         return result

#     def getScaleParams(self):
#         return [getattr(self, f'scale_{i}').parameters() for i in range(self.num_D)]

        
        