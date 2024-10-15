import tensorflow as tf
from tensorflow.keras import layers
import collections
import tensorflow as tf


def grouped_convolution2D(inputs, filters, padding, num_groups,
                          strides=None,
                          dilation_rate=None,
                          data_format='NHWC'):
    """
    Performs a grouped convolution by applying a normal convolution to each of the separate groups
    """
    if data_format not in ['NHWC', 'NCHW']:
        raise ValueError("data_format must be either 'NHWC' or 'NCHW'")

    # Determine the axis for splitting based on data_format
    split_axis = 1 if data_format == 'NCHW' else -1

    # Split input and outputs along their channel dimension
    input_list = tf.split(inputs, num_groups, axis=split_axis)
    filter_list = tf.split(filters, num_groups, axis=-1)
    output_list = []

    # Perform a normal convolution on each split of the input and filters
    for conv_idx, (input_tensor, filter_tensor) in enumerate(zip(input_list, filter_list)):
        output_list.append(tf.nn.convolution(
            input_tensor,
            filter_tensor,
            padding,
            strides=strides,
            dilations=dilation_rate,
            data_format=data_format,
            name=f"grouped_convolution_{conv_idx}"
        ))
    # Concatenate outputs along the channel dimension
    outputs = tf.concat(output_list, axis=split_axis)

    return outputs

class GroupedConvolution(layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 groups,
                 strides=1,
                 padding='SAME',
                 dilation_rate=1,
                 activation=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 data_format='NHWC',
                 **kwargs):
        super(GroupedConvolution, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.data_format = data_format

        if self.data_format not in ['NHWC', 'NCHW']:
            raise ValueError("data_format must be either 'NHWC' or 'NCHW'")

    def build(self, input_shape):
        if self.data_format == 'NHWC':
            input_channels = input_shape[-1]
        else:  # 'NCHW'
            input_channels = input_shape[1]
        
        if self.groups == 1:
            self.conv = layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding=self.padding.lower(),
                dilation_rate=self.dilation_rate,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                data_format='channels_first' if self.data_format == 'NCHW' else 'channels_last'
            )
        else:
            lowest_channels = min(input_channels, self.filters)
            assert lowest_channels % self.groups == 0, "The remainder of min(input_channels, filters) / groups should be zero"
            assert max(input_channels, self.filters) % self.groups == 0, f"The remainder of max(input_channels, filters) / groups ({self.groups}) should be zero"

            if isinstance(self.kernel_size, collections.abc.Iterable):
                kernel_shape = list(self.kernel_size) + [input_channels // self.groups, self.filters]
            else:
                kernel_shape = [self.kernel_size, self.kernel_size, input_channels // self.groups, self.filters]

            self.kernel = self.add_weight(
                name='kernel',
                shape=kernel_shape,
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                trainable=True
            )

            if self.use_bias:
                self.bias = self.add_weight(
                    name='bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    trainable=True
                )

        super(GroupedConvolution, self).build(input_shape)

    def call(self, inputs):
        if self.groups == 1:
            return self.conv(inputs)

        outputs = grouped_convolution2D(
            inputs,
            self.kernel,
            self.padding,
            self.groups,
            strides=[self.strides, self.strides],
            dilation_rate=[self.dilation_rate, self.dilation_rate],
            data_format=self.data_format
        )

        if self.use_bias:
            if self.data_format == 'NHWC':
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
            else:  # 'NCHW'
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self):
        config = super(GroupedConvolution, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'groups': self.groups,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'data_format': self.data_format
        })
        return config