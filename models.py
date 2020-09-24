import tensorflow as tf
from tensorflow.keras.layers import Reshape, Input


def mobilenetv3_96x64_1s(alpha=1.0, embedding_size=12288):
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
        dropout_rate=0.0
    )(net)
    return tf.keras.models.Model(inputs=[model_in], outputs=[net], name='model_v0')


def fred_v0_96x64_1s():
    pass


if __name__ == "__main__":
    mobilenetv3_96x64_1s()
