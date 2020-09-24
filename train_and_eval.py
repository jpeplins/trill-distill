from absl import flags, logging, app
from input_manager import get_dataset
from tensorflow import keras
from models import mobilenetv3_96x64_1s
import tensorflow as tf
import numpy as np
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
    pass

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', None, 'Path to collection of TFRecords.')
flags.DEFINE_string('output_path', None, 'Store model checkpoints and stuff.')
flags.DEFINE_float('learning_rate', 0.01, 'You know what this does.')
flags.DEFINE_integer('num_epochs', 50, 'You know what this does.')
flags.DEFINE_integer('batch_size', 64, 'You know what this does.')


def main(unused_argvs):
    assert FLAGS.dataset_path
    assert FLAGS.output_path

    train_ds = get_dataset(FLAGS.dataset_path, batch_size=FLAGS.batch_size, n_epochs=FLAGS.num_epochs)

    model = mobilenetv3_96x64_1s()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.output_path,
        save_weights_only=False,
        monitor='loss',
        mode='auto',
        save_best_only=True)

    history = model.fit(
        train_ds,
        epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        callbacks=[model_checkpoint_callback]
    )

    np.savetxt(os.path.join(FLAGS.output_path, "loss.txt"), np.array(history.history['loss']), delimiter=",")


if __name__ == '__main__':
    app.run(main)
