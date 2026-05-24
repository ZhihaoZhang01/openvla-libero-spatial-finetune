#!/bin/bash
# 断点续传 openvla-7b
# 用法: bash scripts/download/resume_model_download.sh
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
source "${AUTODL_ROOT}/scripts/env/activate_openvla.sh"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

pip install -q huggingface_hub accelerate

MAX_TRIES="${MAX_TRIES:-20}"
try=1
while [ "$try" -le "$MAX_TRIES" ]; do
  echo "=== Attempt $try/$MAX_TRIES ($(date)) ==="
  if python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('openvla/openvla-7b', resume_download=True, max_workers=1)
print('Done:', path)
"; then
    echo "完成。运行: python run/test_final.py"
    exit 0
  fi
  sleep 10
  try=$((try + 1))
done
exit 1
