#!/bin/bash
set -e
set -x

TENSORBOARD_PORT=6006
tensorboard --logdir "${LOGDIR}" --port "${TENSORBOARD_PORT}"
