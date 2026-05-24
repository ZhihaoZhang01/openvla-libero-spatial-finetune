#!/bin/bash
# 安装 LIBERO（Git 镜像 + numpy<2 + editable 修复）
# 用法: bash scripts/setup/install_libero.sh 2>&1 | tee logs/install_libero.log
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
LIBERO_DIR="${AUTODL_ROOT}/LIBERO"
OPENVLA_DIR="${AUTODL_ROOT}/openvla"
mkdir -p "${AUTODL_ROOT}/logs"

echo "========== $(date '+%Y-%m-%d %H:%M:%S') install_libero.sh =========="

source /root/miniconda3/etc/profile.d/conda.sh
conda activate openvla
export AUTODL_ROOT MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GIT_HTTP_VERSION=1.1

LIBERO_MIRRORS=(
  "https://gitclone.com/github.com/Lifelong-Robot-Learning/LIBERO.git"
  "https://mirror.ghproxy.com/https://github.com/Lifelong-Robot-Learning/LIBERO.git"
  "https://ghfast.top/https://github.com/Lifelong-Robot-Learning/LIBERO.git"
  "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
)

clone_libero() {
  if [ -d "${LIBERO_DIR}/.git" ] && { [ -f "${LIBERO_DIR}/setup.py" ] || [ -f "${LIBERO_DIR}/pyproject.toml" ]; }; then
    echo "LIBERO 已存在: ${LIBERO_DIR}，跳过 clone"
    return 0
  fi
  rm -rf "${LIBERO_DIR}"
  for url in "${LIBERO_MIRRORS[@]}"; do
    echo "尝试克隆: ${url}"
    if git clone --depth 1 "${url}" "${LIBERO_DIR}"; then
      return 0
    fi
    rm -rf "${LIBERO_DIR}"
  done
  return 1
}

pip install 'numpy<2' -q
clone_libero

touch "${LIBERO_DIR}/libero/__init__.py"
cd "${LIBERO_DIR}"
pip uninstall -y libero 2>/dev/null || true
pip install -e .

cd "${OPENVLA_DIR}"
if ! python -c "import robosuite" 2>/dev/null; then
  pip install -r experiments/robot/libero/libero_requirements.txt
  pip install "gym<0.26" mujoco "imageio[ffmpeg]"
else
  echo "robosuite 已安装，跳过 libero_requirements"
fi

pip install "numpy>=1.23.5,<2.0" --force-reinstall -q
python -c "
import numpy; print('numpy', numpy.__version__)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
print('libero OK')
"
echo "========== DONE =========="
