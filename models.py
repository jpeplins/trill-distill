from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from mobilenetv3 import mobile_net_v3
import tensorflow as tf


def distilled_model(embedding_size=2048, dropout=0.1):
    """ Wrapper model that contains large fully connected layer for distilling to layer19. """
    embedding_model = mobile_net_v3((64, 96, 1))
    distillation_model = tf.keras.Sequential([
        embedding_model,
        Dense(12288, activation=tf.nn.swish, kernel_regularizer=regularizers.l2(1e-6), name="layer19_hat")
    ])
    return embedding_model, distillation_model


if __name__ == "__main__":
    pass
