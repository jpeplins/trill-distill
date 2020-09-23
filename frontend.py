import tensorflow as tf

# Such that each 0.96 audio context produces a 96x64 log mel spec.
WIN_SIZE_SEC = 0.1
WIN_STEP_SEC = 0.009
NUM_MEL_BINS = 64
MEL_LB_HZ = 80.0
MEL_UB_HZ = 7600.0


def stabilized_log(x):
    """ Log with an offset. x is a tensor."""
    return tf.math.log(x + 1e-6)


def mel_spectrogram(x, fs):
    """ TF mel spectrogram implementation """

    spec = tf.abs(tf.signal.stft(
        x,
        frame_length=int(fs * WIN_SIZE_SEC),
        frame_step=int(fs * WIN_STEP_SEC),
        fft_length=int(fs * WIN_SIZE_SEC),
    ))

    mel_weight_matrix = tf.cast(tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS,
        int(fs * WIN_SIZE_SEC//2 + 1),
        fs,
        MEL_LB_HZ,
        MEL_UB_HZ
    ), tf.double)

    mel_spec = tf.tensordot(spec, mel_weight_matrix, 1)
    return tf.transpose(mel_spec)


def log_mel_spec(x, fs):
    """ frontend! """
    return stabilized_log(mel_spectrogram(x, fs))
