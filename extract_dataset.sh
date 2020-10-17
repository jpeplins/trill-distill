#!/bin/bash
set -e
set -x
export CUDA_VISIBLE_DEVICES=1

SOURCE_DIR="/data2/audioset-speech/files"
DEST_DIR="/data2/audioset-speech/tfrecords/v1"

python -m dataset \
-file_glob "${SOURCE_DIR}" \
-output_dir "${DEST_DIR}" \
-dataset_version "1"

