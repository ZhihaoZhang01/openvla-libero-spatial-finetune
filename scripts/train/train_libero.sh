#!/bin/bash
# 用法: bash scripts/train/train_libero.sh
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
source "${AUTODL_ROOT}/scripts/env/activate_openvla.sh"

# W&B（与 wandb.ai 上可写权限的 entity/project 一致）
export WANDB_ENTITY="${WANDB_ENTITY:-zhihaozhang321-george-mason-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-openvla-LoRA}"
# 无法联网时取消下一行注释：export WANDB_MODE=offline

DATA_ROOT="${AUTODL_ROOT}/datasets/openvla-libero-spatial"
RUN_DIR="${AUTODL_ROOT}/checkpoints/openvla-7b+libero_spatial+lora"

if [ ! -d "${DATA_ROOT}" ]; then
  echo "缺少数据: ${DATA_ROOT}"
  echo "先运行: python scripts/download/download_data.py"
  exit 1
fi

cd "${AUTODL_ROOT}/openvla"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "${DATA_ROOT}" \
  --dataset_name "libero_spatial_no_noops" \
  --run_root_dir "${AUTODL_ROOT}/checkpoints" \
  --adapter_tmp_dir "${AUTODL_ROOT}/checkpoints/adapter_tmp" \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --max_steps 6500 \
  --save_steps 1000  \
  --image_aug True \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --save_latest_checkpoint_only True

echo "Checkpoint: ${RUN_DIR}"
