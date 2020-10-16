#!/bin/bash
set -e
set -x
export CUDA_VISIBLE_DEVICES=1


#########################################
# A driver script for model distillation.
#########################################

# PARAMS
LEARNING_RATE=0.0001
BATCH_SIZE=128
NUM_EPOCHS=50

# INPUTS
DATASET_PATH="/data2/audioset-speech/tfrecords"

# OUTPUTS
LOGDIR="/data2/audioset-speech/tensorboard"
OUTPUT_PATH="/data2/audioset-speech/models"
CHECKPOINT_PATH="/data2/audioset-speech/checkpoints"

# local debug
#DATASET_PATH="${PWD}/test_output"
#LOGDIR="${PWD}/logs"
#OUTPUT_PATH="${PWD}/trained_models"
#CHECKPOINT_PATH="${PWD}/checkpoints"

function train_student() {
  local model_name=$1
  local embedding_size=$2
  local pre_embedding_size=$3
  local alpha=$4
  local gap=$5

  if [ "$gap" = true ] ; then
    model_id="${model_name}_${embedding_size}_${pre_embedding_size}_${alpha}_gap"
  else
    model_id="${model_name}_${embedding_size}_${pre_embedding_size}_${alpha}"
  fi

  python -m train_and_eval \
    -model_name "${model_id}" \
    -dataset_path "${DATASET_PATH}" \
    -output_path "${OUTPUT_PATH}/${model_id}" \
    -checkpoint_path "${CHECKPOINT_PATH}/${model_id}/${model_id}" \
    -log_path "${LOGDIR}/${model_id}" \
    -learning_rate "${LEARNING_RATE}" \
    -num_epochs "${NUM_EPOCHS}" \
    -batch_size "${BATCH_SIZE}" \
    -embedding_size "${embedding_size}" \
    -pre_embedding_size "${pre_embedding_size}" \
    -alpha "${alpha}" \
    -gap "${gap}"
}


train_student "mnetv3tinyv2" 0 0 1.0 true
train_student "mnetv3tinyv2" 0 0 0.5 true
train_student "mnetv3tinyv2" 0 0 0.25 true






