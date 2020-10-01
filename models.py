from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from mobilenetv3 import mobile_net_v3
import tensorflow as tf


def distilled_model(embedding_size=2048, pre_output_size=4096, dropout=0.1):
    """ Wrapper model that contains large fully connected layer for distilling to layer19. """
    embedding_model = mnetv2_2048_v0(
        alpha=1.0,
        pre_output_size=pre_output_size,
        embedding_size=embedding_size,
    )
    distillation_model = tf.keras.Sequential([
        embedding_model,
        Dense(12288, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(1e-8), name="layer19_hat")
    ])
    return embedding_model, distillation_model


def mnetv2_2048_v0(alpha=1.0, embedding_size=2048, pre_output_size=4096):
    return tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            (64, 96, 1),
            alpha=alpha,
            include_top=False,
            weights=None,
            pooling='avg'),
        tf.keras.layers.Flatten(),
        Dense(pre_output_size, activation=tf.nn.swish),
        Dense(embedding_size, activation=tf.nn.swish)
    ])


if __name__ == "__main__":
    pass
