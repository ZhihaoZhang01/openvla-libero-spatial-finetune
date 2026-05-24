#!/bin/bash
# 新机器一键安装：openvla + dlimp + flash-attn + LIBERO
# 用法: bash scripts/setup/setup_env.sh 2>&1 | tee logs/setup_env.log
set -euo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
SCRIPT_DIR="${AUTODL_ROOT}/scripts/setup"
OPENVLA_DIR="${AUTODL_ROOT}/openvla"
DLIMP_DIR="${AUTODL_ROOT}/dlimp"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') setup_env.sh =========="

grep -q 'AUTODL_ROOT=' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'EOF'
export AUTODL_ROOT=/root/autodl-tmp
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/root/autodl-tmp/dlimp:/root/autodl-tmp/LIBERO:${PYTHONPATH:-}
EOF

export AUTODL_ROOT HF_HOME="${AUTODL_ROOT}/hf_cache" HF_ENDPOINT="https://hf-mirror.com"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "${HF_HOME}" "${AUTODL_ROOT}/datasets" "${AUTODL_ROOT}/checkpoints" \
  "${AUTODL_ROOT}/outputs/videos" "${AUTODL_ROOT}/logs"

apt-get update -qq && apt-get install -y -qq \
  libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf git vim \
  libegl1 libegl1-mesa libgl1-mesa-dri ffmpeg unzip || true

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla 2>/dev/null || { conda create -n openvla python=3.10 -y; conda activate openvla; }

python -c "import torch" 2>/dev/null || \
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

export GIT_HTTP_VERSION=1.1
clone_or_skip() {
  local url="$1" dest="$2"
  if [ -d "${dest}/.git" ]; then echo "已存在 ${dest}，跳过"; return 0; fi
  rm -rf "${dest}"
  git clone --depth 1 "${url}" "${dest}" || \
    git clone --depth 1 "https://ghproxy.net/${url}" "${dest}"
}

clone_or_skip "https://github.com/openvla/openvla.git" "${OPENVLA_DIR}"
cd "${OPENVLA_DIR}"
pip install -r requirements-min.txt
pip install draccus==0.8.0 peft==0.11.1 bitsandbytes==0.43.1 wandb packaging ninja
pip install -e .

clone_or_skip "https://github.com/moojink/dlimp_openvla.git" "${DLIMP_DIR}"
pip install -e "${DLIMP_DIR}"
SITE_PKGS="$(python -c "import site; print(site.getsitepackages()[0])")"
echo "${DLIMP_DIR}" > "${SITE_PKGS}/dlimp_path.pth"

bash "${SCRIPT_DIR}/install_flash_attn.sh" || echo "WARN: flash-attn 失败，推理可用 sdpa"
bash "${SCRIPT_DIR}/install_libero.sh"

echo ""
echo "完成。验证: source scripts/env/activate_openvla.sh && python run/verify_phase01.py"
