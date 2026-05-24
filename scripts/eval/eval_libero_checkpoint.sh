#!/bin/bash
# LIBERO 评测：加载 LoRA checkpoint，在仿真里 rollout 并统计成功率
#
# 用法:
#   # 更快冒烟（10 任务 × 2 次 ≈ 20 个 episode）
#   NUM_TRIALS=2 bash scripts/eval/eval_libero_checkpoint.sh
#
#   # 默认评测（3 任务 × 10 次 = 30 episodes）
#   bash scripts/eval/eval_libero_checkpoint.sh
#
#   # 全套 10 任务
#   NUM_TASKS=10 bash scripts/eval/eval_libero_checkpoint.sh
#
#   # 更稳统计（10 任务 × 50 trials）
#   NUM_TASKS=10 NUM_TRIALS=50 bash scripts/eval/eval_libero_checkpoint.sh
#
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
source "${AUTODL_ROOT}/scripts/env/activate_openvla.sh"

CKPT="${CKPT:-${AUTODL_ROOT}/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug}"
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
CENTER_CROP="${CENTER_CROP:-True}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_TASKS="${NUM_TASKS:-3}"
RUN_NOTE="${RUN_NOTE:-manual-eval}"
USE_WANDB="${USE_WANDB:-False}"

if [[ ! -f "${CKPT}/adapter_model.safetensors" ]]; then
  echo "缺少 LoRA 权重: ${CKPT}/adapter_model.safetensors"
  exit 1
fi

echo "Checkpoint:  ${CKPT}"
echo "Task suite:  ${TASK_SUITE}"
echo "center_crop: ${CENTER_CROP}"
echo "tasks:       ${NUM_TASKS} (of 10 in libero_spatial)"
echo "trials/task: ${NUM_TRIALS}"

cd "${AUTODL_ROOT}/openvla"
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "${CKPT}" \
  --task_suite_name "${TASK_SUITE}" \
  --center_crop "${CENTER_CROP}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --num_tasks "${NUM_TASKS}" \
  --run_id_note "${RUN_NOTE}" \
  --use_wandb "${USE_WANDB}"

echo ""
echo "日志: ${AUTODL_ROOT}/openvla/experiments/logs/EVAL-${TASK_SUITE}-openvla--*${RUN_NOTE}*.txt"
echo "视频: ${AUTODL_ROOT}/openvla/rollouts/"
