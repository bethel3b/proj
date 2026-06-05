#!/bin/bash
# Running command:        bash mlops/gpt/run.sh
# Debug (attach VSCode):  DEBUG=1 bash mlops/gpt/run.sh
set -euo pipefail

EPOCHS=5
RUN_NAME="Test: grad norm"

DEBUG=${DEBUG:-0}

if [[ "$DEBUG" == "1" ]]; then
    echo "Waiting for VSCode debugger to attach on port 5678..."
    export PYDEVD_DISABLE_FILE_VALIDATION=1
    PYRUN="python -m debugpy --listen 5678 --wait-for-client"
else
    PYRUN="python"
fi

$PYRUN mlops/gpt/train.py \
    --epochs "$EPOCHS" \
    --run-name "$RUN_NAME"
