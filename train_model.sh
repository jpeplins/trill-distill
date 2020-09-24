#!/bin/bash
set -e
set -x

python -m train_and_eval -dataset_path "/data2/audioset-speech/tfrecords" -output_path "/data2/audioset-speech/models"


