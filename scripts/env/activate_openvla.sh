#!/bin/bash
# 用法: source /root/autodl-tmp/scripts/env/activate_openvla.sh
export AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
export HF_HOME="${HF_HOME:-${AUTODL_ROOT}/hf_cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTHONPATH="${AUTODL_ROOT}/dlimp:${AUTODL_ROOT}/LIBERO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla
cd "${AUTODL_ROOT}" || exit 1
mkdir -p "${HF_HOME}" "${AUTODL_ROOT}/datasets" "${AUTODL_ROOT}/checkpoints" \
  "${AUTODL_ROOT}/outputs/videos" "${AUTODL_ROOT}/logs"
