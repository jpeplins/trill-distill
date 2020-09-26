import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input, Conv2D, Dense, Dropout, Flatten, BatchNormalization, AveragePooling2D
from tensorflow.keras import regularizers
from mobilenetv3_block import BottleNeck


def distilled_model(embedding_size=2048, dropout=0.1):
    """ Wrapper model that contains large fully connected layer for distilling to layer19. """
    embedding_model = keras_mobilenetv3_small(num_classes=embedding_size, dropout=dropout)
    distillation_model = tf.keras.Sequential([
        embedding_model,
        Dense(12288, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(1e-6), name="layer19_hat")
    ])
    return embedding_model, distillation_model


def keras_mobilenetv3_small(num_classes=2048, dropout=0.1):
    model = tf.keras.Sequential([
        Input(shape=(64, 96), name="log_mel_spec"),
        Reshape((64, 96, 1)),
        Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same", use_bias=True),
        BatchNormalization(epsilon=1e-3, momentum=0.999),
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
        Conv2D(filters=576, kernel_size=(1, 1), strides=1, padding="same", use_bias=True),
        Dropout(dropout),
        BatchNormalization(),
        AveragePooling2D(pool_size=(4, 6), strides=1),
        Conv2D(filters=1024, kernel_size=(1, 1), strides=1, padding="same"),
        Conv2D(filters=num_classes, kernel_size=(1, 1), strides=1, padding="same", activation=tf.nn.swish),
        Flatten(name="embedding"),
    ])
    return model


def keras_mobilenetv3_tiny(num_classes=12288, dropout=0.1):
    model = tf.keras.Sequential([
        Input(shape=(64, 96)),
        Reshape((64, 96, 1)),
        Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same"),
        Dropout(dropout),
        BatchNormalization(),
        BottleNeck(in_size=16, exp_size=16, out_size=16, s=2, is_se_existing=True, NL="RE", k=3),
        BottleNeck(in_size=16, exp_size=72, out_size=24, s=2, is_se_existing=False, NL="RE", k=3),
        BottleNeck(in_size=24, exp_size=88, out_size=24, s=1, is_se_existing=False, NL="RE", k=3),
        BottleNeck(in_size=24, exp_size=96, out_size=40, s=2, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=240, out_size=40, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=40, exp_size=120, out_size=48, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=48, exp_size=144, out_size=48, s=1, is_se_existing=True, NL="HS", k=5),
        BottleNeck(in_size=48, exp_size=288, out_size=96, s=2, is_se_existing=True, NL="HS", k=5),
        Conv2D(filters=288, kernel_size=(1, 1), strides=1, padding="same"),
        Dropout(dropout),
        BatchNormalization(epsilon=1e-3, momentum=0.999),
        AveragePooling2D(pool_size=(4, 6), strides=1),
        Conv2D(filters=576, kernel_size=(1, 1), strides=1, padding="same"),
        Flatten(name="embedding"),
        Dense(num_classes, activation=tf.nn.swish)
    ])
    return model


if __name__ == "__main__":
    e_model, d_model = distilled_model(embedding_size=2048, dropout=0.1)
    dad = tf.keras.applications.MobileNetV3Small(input_shape=(64, 96, 1), weights=None, include_top=True, classes=2048)
    dad.summary()
    d = 0