from input_manager import get_dataset
from models import distilled_model
from tensorflow import keras
from absl import flags, app
import tensorflow as tf
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
    pass

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', None, 'Give your model a name.')
flags.DEFINE_string('dataset_path', None, 'Path to collection of TFRecords.')
flags.DEFINE_string('output_path', None, 'Path to save distilled embedding models.')
flags.DEFINE_string('checkpoint_path', None, 'Path to save training checkpoints.')
flags.DEFINE_string('log_path', None, 'Path to store logs for tensorboard')
flags.DEFINE_float('learning_rate', None, 'You know what this does.')
flags.DEFINE_integer('num_epochs', None, 'You know what this does.')
flags.DEFINE_integer('batch_size', None, 'You know what this does.')
flags.DEFINE_integer('embedding_size', None, 'Size of embedding to distill.')
flags.DEFINE_integer('pre_embedding_size', None, 'Size of FC layer right before embedding layer.')
flags.DEFINE_float('dropout', 0.05, 'Dropout.')
flags.DEFINE_float('alpha', 1.0, 'Alpha for mobile net')
flags.DEFINE_bool('gap', True, 'GlobalAveragePool mobile net output')

def main(_):
    assert FLAGS.model_name
    assert FLAGS.dataset_path
    assert FLAGS.output_path
    assert FLAGS.checkpoint_path
    assert FLAGS.log_path
    assert FLAGS.learning_rate
    assert FLAGS.num_epochs
    assert FLAGS.batch_size
    assert FLAGS.embedding_size
    assert FLAGS.pre_embedding_size >= 0
    assert FLAGS.learning_rate

    train_ds, test_ds = get_dataset(FLAGS.dataset_path, batch_size=FLAGS.batch_size)

    embedding_model, distillation_model = distilled_model(
        embedding_size=FLAGS.embedding_size,
        pre_embedding_size=FLAGS.pre_embedding_size,
        alpha=FLAGS.alpha,
        dropout=FLAGS.dropout,
        gap=FLAGS.gap
    )

    schedule = keras.optimizers.schedules.ExponentialDecay(
        FLAGS.learning_rate,
        5000,
        0.97,
        staircase=True
    )

    # TODO(jpeplins): Check for checkpoints and reload if desired.
    distillation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=schedule),
        loss=keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='auto',
        save_best_only=False
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,
        mode='auto',
    )

    board = tf.keras.callbacks.TensorBoard(
        log_dir=FLAGS.log_path,
        histogram_freq=1,
        write_graph=False,
        write_images=False,
        update_freq='epoch',
        profile_batch=2,
    )

    distillation_model.fit(
        x=train_ds,
        validation_data=test_ds,
        epochs=FLAGS.num_epochs,
        batch_size=FLAGS.batch_size,
        callbacks=[model_checkpoint_callback, early_stopping, board]
    )

    tf.keras.models.save_model(
        embedding_model,
        FLAGS.output_path
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.output_path)
    with open(os.path.join(FLAGS.output_path, '%s.tflite' % FLAGS.model_name), 'wb') as f:
        f.write(converter.convert())


if __name__ == '__main__':
    app.run(main)
