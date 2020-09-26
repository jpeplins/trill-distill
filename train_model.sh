#!/bin/bash
set -e
set -x

#########################################
# A driver script for model distillation.
#########################################

# PARAMS
BATCH_SIZE=64
NUM_EPOCHS=100
EMBEDDING_SIZE=2048
LEARNING_RATE=0.1
TENSORBOARD_PORT=6006

# INPUTS
DATASET_PATH="/data2/audioset-speech/tfrecords"
MODEL_NAME="mnetv3_small_${EMBEDDING_SIZE}_v0"

# OUTPUTS
LOGDIR="/data2/audioset-speech/tensorboard/"
OUTPUT_PATH="/data2/audioset-speech/models"
CHECKPOINT_PATH="/data2/audioset-speech/checkpoints"

tensorboard --logdir "${LOGDIR}" --port "${TENSORBOARD_PORT}"

python -m train_and_eval \
-model_name "${MODEL_NAME}" \
-dataset_path "${DATASET_PATH}" \
-output_path "${OUTPUT_PATH}" \
-checkpoint_path "${CHECKPOINT_PATH}" \
-log_path "${LOGDIR}" \
-learning_rate "${LEARNING_RATE}" \
-num_epochs "${NUM_EPOCHS}" \
-batch_size "${BATCH_SIZE}" \
-embedding_size "${BATCH_SIZE}"
