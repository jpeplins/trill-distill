from dataset import load_wav
import tensorflow_hub as hub
from frontend import log_mel_spec
import time

WIN_SIZE_SEC = 0.96

if __name__ == "__main__":

    module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')
    start = time.time()
    x, fs = load_wav("/Users/jacob/code/lab/trill-distill/test_data/test.wav")
    x_samp = x[0:int(WIN_SIZE_SEC * fs)]
    target = module(samples=x_samp, sample_rate=fs)['layer19'].numpy()
    spec = log_mel_spec(x_samp, fs).numpy()

    print(target.shape, time.time() - start)
