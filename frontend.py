import tensorflow as tf
import numpy as np

# Such that each 0.96 audio context produces a 96x64 log mel spec.
WIN_SIZE_SEC = 0.025
WIN_STEP_SEC = 0.01
NUM_MEL_BINS = 64
MEL_LB_HZ = 80.0
MEL_UB_HZ = 7600.0


def stabilized_log(x, additive_offset, floor):
  """TF version of mfcc_mel.StabilizedLog."""
  return tf.math.log(tf.math.maximum(x, floor) + additive_offset)

def log_mel_spectrogram(data,
                        audio_sample_rate,
                        log_additive_offset=0.001,
                        log_floor=1e-12,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        fft_length=None):
  """TF version of mfcc_mel.LogMelSpectrogram."""
  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  if not fft_length:
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

  spectrogram = tf.abs(
      tf.signal.stft(
          tf.cast(data, tf.dtypes.float64),
          frame_length=window_length_samples,
          frame_step=hop_length_samples,
          fft_length=fft_length,
          window_fn=tf.signal.hann_window,
      )
  )

  to_mel = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=64,
      num_spectrogram_bins=fft_length // 2 + 1,
      sample_rate=audio_sample_rate,
      lower_edge_hertz=125.0,
      upper_edge_hertz=7500.0,
      dtype=tf.dtypes.float64
  )

  mel = spectrogram @ to_mel
  log_mel = stabilized_log(mel, log_additive_offset, log_floor)
  return log_mel


def compute_frontend_features(samples, sr, overlap_seconds, tflite=False):
  """Compute features."""
  if tflite:
    raise ValueError("TFLite frontend unsupported")
  if samples.dtype == np.int16:
    samples = tf.cast(samples, np.float32) / np.iinfo(np.int16).max
  if samples.dtype == np.float64:
    samples = tf.cast(samples, np.float32)
  assert samples.dtype == np.float32, samples.dtype
  n = tf.size(samples)
  n_required = 16000
  samples = tf.cond(
      n < n_required,
      lambda: tf.pad(samples, [(0, n_required - n)]),
      lambda: samples
  )
  mel = log_mel_spectrogram(samples, sr)
  # Frame to ~.96 seconds per chunk (96 frames) with ~.0.793 second overlap.
  step = 96 - overlap_seconds
  mel = tf.signal.frame(mel, frame_length=96, frame_step=step, axis=0)
  return mel