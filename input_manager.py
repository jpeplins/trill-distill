import tensorflow as tf
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
SPEC_HEIGHT = 64
SPEC_WIDTH = 96
TARGET_SIZE = 12288
NUM_EXAMPLES = 902523


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


def get_dataset(data_dir, batch_size=32, train_percent=0.8):

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

    # train/test split
    test_size = int(NUM_EXAMPLES * (1 - train_percent))
    ds_test = ds.take(test_size)
    ds_train = ds.skip(test_size)

    # Shuffle batches each epoch.
    ds_train = ds_train.shuffle(int(1e3), reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(test_size)

    return ds_train.prefetch(buffer_size=AUTOTUNE), ds_test.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    # train_ds = get_dataset("/Users/jacob/code/lab/trill-distill/test_output")
    pass
