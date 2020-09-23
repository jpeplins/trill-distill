from absl import flags, logging, app
from frontend import log_mel_spec
from scipy.io import wavfile
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import traceback
import os

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
    tf.config.experimental.set_memory_growth(physical_devices[2], True)
except:
    pass

SAMPLING_RATE = 16000
WIN_SIZE_SEC = 0.96
CONTEXT_SIZE_SAMPLES = int(WIN_SIZE_SEC*SAMPLING_RATE)
EXAMPLES_PER_SHARD = 200  # 68kb per example -> 140mb per shard

FLAGS = flags.FLAGS

flags.DEFINE_string('file_glob', None, 'Glob of input wav files to process.')
flags.DEFINE_string('output_dir', None, 'Directory to store processed dataset.')
flags.DEFINE_string('dataset_version', None, 'Give your dataset a version number.')


def _float_feature(list_of_floats):  # float32
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(spectrogram, trill_feature):
    feature = {
        'audio': _float_feature(spectrogram),
        'label': _float_feature(trill_feature),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_wav(path):
    fs, x = wavfile.read(path)
    x = np.asarray(x, dtype=float)
    x = x / pow(2, 15)
    return x, fs


def main(unused_argv):
    assert FLAGS.file_glob
    assert FLAGS.output_dir
    assert FLAGS.dataset_version

    # Load ResNet50 version of Trill
    module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')

    # files to prepare for distillation training
    wav_list = os.listdir(FLAGS.file_glob)
    num_files = len(wav_list)
    num_shards = int(num_files / EXAMPLES_PER_SHARD) + (1 if num_files % EXAMPLES_PER_SHARD != 0 else 0)

    for shard in range(num_shards):

        shard_fn = "distill_training_v0_%.5d-of-%.5d.tfrecord" % (shard, num_shards-1)
        shard_path = os.path.join(FLAGS.output_dir, shard_fn)
        start_idx = shard * EXAMPLES_PER_SHARD

        logging.info("Processing %3.3f-percent complete." % (100 * (start_idx / num_files)))

        with tf.io.TFRecordWriter(shard_path) as out:
            for idx in range(start_idx, start_idx+EXAMPLES_PER_SHARD):

                try:
                    fn = wav_list[idx]
                    x, _ = load_wav(os.path.join(FLAGS.file_glob, fn))

                    if len(x) < CONTEXT_SIZE_SAMPLES:
                        continue
                    start_idx = np.random.randint(0, len(x)-CONTEXT_SIZE_SAMPLES)
                    x_samp = x[start_idx:start_idx+CONTEXT_SIZE_SAMPLES]

                    # spectrogram input
                    lms = log_mel_spec(x_samp, SAMPLING_RATE).numpy()
                    # trill target
                    target = module(samples=x_samp, sample_rate=SAMPLING_RATE)['layer19'].numpy()

                    # Write serialized example to TFRecord file
                    example = to_tfrecord(lms.flatten().tolist(), target.flatten().tolist())
                    out.write(example.SerializeToString())

                except Exception as e:
                    print(traceback.format_exc())



if __name__ == '__main__':
    app.run(main)
