#!/bin/bash
set -e
set -x

TENSORBOARD_PORT=6006
LOGDIR="/data2/audioset-speech/tensorboard/"
tensorboard --logdir "${LOGDIR}" --port "${TENSORBOARD_PORT}"
