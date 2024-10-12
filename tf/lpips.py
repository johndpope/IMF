import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as applications

class LPIPS(tf.keras.layers.Layer):
    def __init__(self, net='alex', version='0.1'):
        super(LPIPS, self).__init__()
        self.net = net
        self.version = version
        self.model = self._build_model()

    def _build_model(self):
        if self.net == 'alex':
            base_model = applications.VGG16(weights='imagenet', include_top=False)
        elif self.net == 'vgg':
            base_model = applications.VGG16(weights='imagenet', include_top=False)
        else:
            raise ValueError(f"Unsupported network: {self.net}")

        outputs = [base_model.get_layer(name).output for name in [
            'block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'
        ]]
        return tf.keras.Model(inputs=base_model.input, outputs=outputs)

    def call(self, img0, img1):
        # Assuming img0 and img1 are in range [-1, 1]
        features0 = self.model(img0)
        features1 = self.model(img1)

        diffs = [tf.reduce_mean((f0 - f1) ** 2, axis=[1, 2, 3]) for f0, f1 in zip(features0, features1)]
        return tf.reduce_sum(diffs, axis=0)

def lpips(image0, image1, model='net-lin', net='alex'):
    lpips_model = LPIPS(net=net)
    return lpips_model(image0, image1)