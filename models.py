import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input, SeparableConv2D, Conv2D, Dense
from mobilenetv3_block import BottleNeck, h_swish


def mobilenetv2_96x64_1s(alpha=1.0, embedding_size=12288):
    """ Vanilla MobileNet"""

    model_in = Input(shape=(64, 96))
    net = Reshape((64, 96, 1))(model_in)
    net = tf.keras.applications.MobileNetV2(
        input_shape=(64, 96, 1),
        alpha=alpha,
        include_top=True,
        weights=None,
        classifier_activation=tf.nn.swish,
        classes=embedding_size,
    )(net)
    return tf.keras.models.Model(inputs=[model_in], outputs=[net], name='model_v0')


def fred_v0_96x64_1s(out_size=12288):
    model_in = Input(shape=(64, 96))
    net = Reshape((64, 96, 1))(model_in)
    pass


def keras_mobilenetv3_small(num_classes=12288):
    net = tf.keras.Sequential([
        Input(shape=(64, 96)),
        Reshape((64, 96, 1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3),
        BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3),
        BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3),
        BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=96, exp_size=576, out_size=96, s=1, is_se_existing=True, NL="HS", k=5),
        tf.keras.layers.Conv2D(filters=576, kernel_size=(1, 1), strides=1, padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.AveragePooling2D(pool_size=(4, 6), strides=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.swish)
    ])
    return net


if __name__ == "__main__":
    model = keras_mobilenetv3_small()
    model.summary()
