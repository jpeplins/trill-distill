from input_manager import get_dataset
from models import distilled_model
from datetime import datetime
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
flags.DEFINE_float('learning_rate', 0.1, 'You know what this does.')
flags.DEFINE_integer('num_epochs', 50, 'You know what this does.')
flags.DEFINE_integer('batch_size', 128, 'You know what this does.')
flags.DEFINE_integer('embedding_size', 2048, 'Size of embedding to distill.')
flags.DEFINE_integer('pre_output_size', 4096, 'Size of FC layer right before embedding layer.')
flags.DEFINE_float('dropout', 0.1, 'Dropout.')


def main(_):
    assert FLAGS.dataset_path
    assert FLAGS.output_path
    assert FLAGS.checkpoint_path
    assert FLAGS.log_path
    assert FLAGS.model_name

    train_ds, test_ds = get_dataset(FLAGS.dataset_path, batch_size=FLAGS.batch_size)

    embedding_model, distillation_model = distilled_model(
        embedding_size=FLAGS.embedding_size,
        pre_output_size=FLAGS.pre_output_size,
        dropout=FLAGS.dropout
    )

    distillation_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=['accuracy']
    )

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=FLAGS.checkpoint_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='auto',
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='auto',
    )

    board = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(FLAGS.log_path, datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq=1000,
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
