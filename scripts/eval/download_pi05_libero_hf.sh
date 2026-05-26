#!/bin/bash
# Download OpenPI-compatible pi05_libero (JAX/Orbax) via HuggingFace 国内镜像.
# Same layout as gs://openpi-assets/checkpoints/pi05_libero → OPENPI_DATA_HOME.
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
OPENPI_ROOT="${OPENPI_ROOT:-${AUTODL_ROOT}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${AUTODL_ROOT}/openpi_cache}"
REPO="${HF_REPO:-bf-jeon/pi05_libero}"
DEST="${OPENPI_DATA_HOME}/openpi-assets/checkpoints/pi05_libero"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "${DEST}"
cd "${OPENPI_ROOT}"
source .venv/bin/activate

echo "Repo:  ${REPO}"
echo "Mirror: ${HF_ENDPOINT}"
echo "Dest:  ${DEST}"

huggingface-cli download "${REPO}" \
  --local-dir "${DEST}" \
  --local-dir-use-symlinks False

echo "Done. Verify:"
ls -la "${DEST}/params" "${DEST}/assets" 2>/dev/null | head -5
echo ""
echo "Start server:"
echo "  OPENPI_DATA_HOME=${OPENPI_DATA_HOME} uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=${DEST}"
