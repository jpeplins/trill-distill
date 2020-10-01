#!/bin/bash
set -e
set -x

#########################################
# A driver script for model distillation.
#########################################

# PARAMS
MODEL_VERSION=4
BATCH_SIZE=128
NUM_EPOCHS=80
EMBEDDING_SIZE=2048
PRE_OUTPUT_SIZE=12288
LEARNING_RATE=0.1

# INPUTS
DATASET_PATH="/data2/audioset-speech/tfrecords"
MODEL_NAME="mnetv3_small_${EMBEDDING_SIZE}_v${MODEL_VERSION}"

# OUTPUTS
LOGDIR="/data2/audioset-speech/tensorboard/${MODEL_NAME}"
OUTPUT_PATH="/data2/audioset-speech/models/${MODEL_NAME}"
CHECKPOINT_PATH="/data2/audioset-speech/checkpoints/${MODEL_NAME}"

python -m train_and_eval \
-model_name "${MODEL_NAME}" \
-dataset_path "${DATASET_PATH}" \
-output_path "${OUTPUT_PATH}" \
-checkpoint_path "${CHECKPOINT_PATH}" \
-log_path "${LOGDIR}" \
-learning_rate "${LEARNING_RATE}" \
-num_epochs "${NUM_EPOCHS}" \
-batch_size "${BATCH_SIZE}" \
-embedding_size "${EMBEDDING_SIZE}" \
-pre_output_size "${PRE_OUTPUT_SIZE}"
