#!/bin/bash
# 安装 flash-attn 2.5.5（OpenVLA 官方版本）
# 用法: bash scripts/setup/install_flash_attn.sh 2>&1 | tee logs/install_flash_attn.log
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
mkdir -p "${AUTODL_ROOT}/logs"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') install_flash_attn.sh =========="

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla

python -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
"

if ! conda list -n openvla cuda-nvcc 2>/dev/null | grep -q cuda-nvcc; then
  conda install -y -n openvla -c "nvidia/label/cuda-12.1.0" cuda-nvcc cuda-cudart-dev 2>/dev/null || \
  conda install -y -n openvla -c nvidia cuda-nvcc=12.1 2>/dev/null || true
fi

NVCC_CONDA="$(find "${CONDA_PREFIX}" -name nvcc -type f 2>/dev/null | head -1)"
if [ -n "${NVCC_CONDA}" ]; then
  export CUDA_HOME="$(dirname "$(dirname "${NVCC_CONDA}")")"
  export PATH="${CUDA_HOME}/bin:${PATH}"
else
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi

pip install -q packaging ninja wheel
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-4}"

pip install "flash-attn==2.5.5" --no-build-isolation

python -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'OK')"
echo "========== DONE =========="
