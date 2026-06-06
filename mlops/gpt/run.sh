#!/bin/bash
# Running command:        bash mlops/gpt/run.sh
# Debug (attach VSCode):  DEBUG=1 bash mlops/gpt/run.sh
set -euo pipefail

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# Hparam
EPOCHS=20
BATCH_SIZE=2 # default: 2
LR=1e-1 # default: 1e-4

# MLFLOW
EXP_NAME="GPT (Decoder Only)"
RUN_NAME="Test: mlflow logging ${BATCH_SIZE}bs"

CKPT_ARGS=()
# CKPT_ARGS=(--checkpoint-dir "run/checkpoints/gpt/${TIMESTAMP}")


DEBUG=${DEBUG:-1}

if [[ "$DEBUG" == "1" ]]; then
    echo "Waiting for VSCode debugger to attach on port 5678..."
    export PYDEVD_DISABLE_FILE_VALIDATION=1
    PYRUN="python -m debugpy --listen 5678 --wait-for-client"
else
    PYRUN="python"
fi

$PYRUN mlops/gpt/train.py \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --experiment-name "$EXP_NAME" \
    --run-name "$RUN_NAME" \
    --batch-size "$BATCH_SIZE" \
    "${CKPT_ARGS[@]}"