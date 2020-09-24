import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
SPEC_HEIGHT = 64
SPEC_WIDTH = 96
TARGET_SIZE = 12288


def _parse_batch(record_batch):
    """ Parses a TFExample from a record. """

    # Create a description of the features
    feature_description = {
        'audio': tf.io.FixedLenFeature([int(SPEC_HEIGHT * SPEC_WIDTH)], tf.float32),
        'label': tf.io.FixedLenFeature([TARGET_SIZE], tf.float32),
    }

    example = tf.io.parse_example(record_batch, feature_description)
    return tf.reshape(example['audio'], (64, 96)), example['label']


def reshape_tensor(tensor):
    return tf.reshape(tensor, (64, 96, 1))


def get_dataset(data_dir, batch_size=32, n_epochs=10):

    files_ds = tf.data.Dataset.list_files(os.path.join(data_dir, '*.tfrecord'))

    # DATASET OPTIONS:
    ignore_order = tf.data.Options()
    # Disregard data order in favor of reading speed
    ignore_order.experimental_deterministic = False
    # APPLY DATASET OPTIONS
    files_ds = files_ds.with_options(ignore_order)

    # Read records and prepare batches
    ds = tf.data.TFRecordDataset(files_ds)

    # Parse a batch into a dataset of [audio, label] pairs
    ds = ds.map(lambda x: _parse_batch(x))

    # Repeat the training data for n_epochs.
    # ds = ds.repeat(n_epochs)

    # Shuffle batches each epoch.
    ds = ds.shuffle(int(1e3), reshuffle_each_iteration=True)

    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    # train_ds = get_dataset("/Users/jacob/code/lab/trill-distill/test_output")
    pass
