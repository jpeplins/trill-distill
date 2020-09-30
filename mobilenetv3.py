"""MobileNet v3 models for Keras."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.keras import layers
import tensorflow as tf


def mobile_net_v3(input_shape, alpha=1.0, num_classes=2048, pre_output_length=1024, dropout=0.1, rescaling=True):

    model_in = layers.Input(shape=input_shape, name="log_mel_spectrogram")
    x = model_in

    if rescaling:
        x = layers.experimental.preprocessing.Rescaling(1./255.)(x)

    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, name='Conv')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm')(x)
    x = hard_swish(x)

    # BEGIN LINEAR BOTTLENECKS
    #                       x,  expansion,   num_filters,        k,  s,  se,   activation, block_id
    x = _inverted_res_block(x,  1,           _depth(16 * alpha), 3,  2,  0.25, relu,       0)
    x = _inverted_res_block(x,  72. / 16,    _depth(24 * alpha), 3,  2,  None, relu,       1)
    x = _inverted_res_block(x,  88. / 24,    _depth(24 * alpha), 3,  1,  None, relu,       2)
    x = _inverted_res_block(x,  4,           _depth(40 * alpha), 5,  2,  0.25, hard_swish, 3)
    x = _inverted_res_block(x,  6,           _depth(40 * alpha), 5,  1,  0.25, hard_swish, 4)
    x = _inverted_res_block(x,  6,           _depth(40 * alpha), 5,  1,  0.25, hard_swish, 5)
    x = _inverted_res_block(x,  3,           _depth(48 * alpha), 5,  1,  0.25, hard_swish, 6)
    x = _inverted_res_block(x,  3,           _depth(48 * alpha), 5,  1,  0.25, hard_swish, 7)
    x = _inverted_res_block(x,  6,           _depth(96 * alpha), 5,  2,  0.25, hard_swish, 8)
    x = _inverted_res_block(x,  6,           _depth(96 * alpha), 5,  1,  0.25, hard_swish, 9)
    x = _inverted_res_block(x,  6,           _depth(96 * alpha), 5,  1,  0.25, hard_swish, 10)
    # END LINEAR BOTTLENECKS

    x = layers.Conv2D(576, kernel_size=1, padding='same', use_bias=True, name='Conv_1')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = hard_swish(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(pre_output_length, activation=tf.nn.swish)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_classes, activation=tf.nn.swish, name="embedding")(x)

    return models.Model(model_in, x, name='MobilenetV3')


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(inputs)
    x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(_depth(filters * se_ratio), kernel_size=1, padding='same', name=prefix + 'squeeze_excite/Conv')(x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(filters, kernel_size=1, padding='same', name=prefix + 'squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
    shortcut = x
    prefix = 'expanded_conv/'

    # spectrograms are single channel images
    infilters = backend.int_shape(x)[-1]

    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(_depth(infilters * expansion), kernel_size=1, padding='same', use_bias=False, name=prefix + 'expand')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand/BatchNorm')(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size), name=prefix + 'depthwise/pad')(x)

    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same' if stride == 1 else 'valid', use_bias=False, name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False, name=prefix + 'project')(x)
    x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])

    return x
