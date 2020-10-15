from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import regularizers
from mobilenet_v3 import MobileNetV3Small, MobileNetV3Tiny
import tensorflow as tf


def distilled_model(embedding_size=2048, pre_embedding_size=4096, alpha=1.0, dropout=0.1, gap=True):
    """ Wrapper model that contains large fully connected layer for distilling to layer19. """
    embedding_model = mnetv3_2048_arch(
        embedding_size=embedding_size,
        pre_embedding_size=pre_embedding_size,
        alpha=alpha,
        dropout=dropout,
        gap=gap)
    distillation_model = tf.keras.Sequential([
        embedding_model,
        Dense(12288, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(1e-9), name="layer19_hat")
    ])
    return embedding_model, distillation_model


def mnetv3_2048_arch(embedding_size=2048, pre_embedding_size=2048, alpha=1.0, dropout=0.0, gap=True):
    model_in = Input(shape=(64, 96, 1), name="log_mel_spec")
    x = MobileNetV3Tiny(
            input_shape=(64, 96, 1),
            alpha=alpha,
            minimalistic=False,
            include_top=False,
            weights=None,
            pooling='avg' if gap else None,
            dropout_rate=dropout)(model_in)
    x = Flatten()(x)
    if pre_embedding_size > 0:
        x = Dense(pre_embedding_size, activation=tf.nn.swish, name='pre_embedding')(x)
    if embedding_size > 0:
        x = Dense(embedding_size, activation=tf.nn.swish, name='embedding')(x)
    return tf.keras.Model(inputs=[model_in], outputs=[x])

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
    e, d = distilled_model(
        embedding_size=0,
        pre_embedding_size=0,
        alpha=1.0)
    e.summary()