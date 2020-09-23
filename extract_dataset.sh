#!/bin/bash
set -e
set -x

python -m dataset -file_glob "/data2/audioset-speech/files" -output_dir "/data2/audioset-speech/tfrecords" -dataset_version "0"

