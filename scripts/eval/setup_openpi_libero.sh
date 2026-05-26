#!/bin/bash
# One-time setup: openpi (uv) + LIBERO client venv for examples/libero/main.py
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
OPENPI_ROOT="${OPENPI_ROOT:-${AUTODL_ROOT}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${AUTODL_ROOT}/openpi_cache}"
MARKER="${AUTODL_ROOT}/.openpi_libero_setup_done"

export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -d "${OPENPI_ROOT}/.git" ]]; then
  echo "缺少 openpi 仓库: ${OPENPI_ROOT}"
  echo "  git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git ${OPENPI_ROOT}"
  exit 1
fi

# Apply our patched eval client (num_tasks, logging, video names)
PATCH_SRC="${AUTODL_ROOT}/patches/openpi/examples_libero_main.py"
PATCH_DST="${OPENPI_ROOT}/examples/libero/main.py"
if [[ -f "${PATCH_SRC}" ]]; then
  cp -a "${PATCH_SRC}" "${PATCH_DST}"
  echo "[setup] Applied patch → ${PATCH_DST}"
fi

if [[ -f "${MARKER}" ]]; then
  echo "[setup] Already done (${MARKER}). Skip."
  exit 0
fi

echo "[setup] openpi install (PyPI mirror + local lerobot)…"
cd "${OPENPI_ROOT}"

# Avoid slow/blocked git fetch during uv sync on AutoDL
LEROBOT_SRC="${AUTODL_ROOT}/vendor/lerobot"
if [[ ! -d "${LEROBOT_SRC}/.git" ]]; then
  mkdir -p "${AUTODL_ROOT}/vendor"
  git clone --depth 1 "https://ghfast.top/https://github.com/huggingface/lerobot" "${LEROBOT_SRC}"
  git -C "${LEROBOT_SRC}" fetch --depth 1 origin 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
  git -C "${LEROBOT_SRC}" checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
fi
if ! grep -q "path = \"${LEROBOT_SRC}\"" pyproject.toml; then
  sed -i "s|lerobot = { git = \"https://github.com/huggingface/lerobot\", rev = \"0cf864870cf29f4738d3ade893e6fd13fbd7cdb5\" }|lerobot = { path = \"${LEROBOT_SRC}\" }|" pyproject.toml
fi

export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-${AUTODL_ROOT}/.cache/uv}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-8}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
mkdir -p "${UV_CACHE_DIR}"

rm -rf .venv
uv venv --python 3.11
uv sync
uv pip install -e .

echo "[setup] LIBERO client venv (Python 3.8)…"
uv venv --python 3.8 examples/libero/.venv
# shellcheck source=/dev/null
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate

mkdir -p "${OPENPI_DATA_HOME}"
date -Iseconds > "${MARKER}"
echo "[setup] Complete. OPENPI_DATA_HOME=${OPENPI_DATA_HOME}"
