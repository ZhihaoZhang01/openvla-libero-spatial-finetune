#!/bin/bash
# Shared helpers for LIBERO LoRA experiment sweep. Source, do not execute directly.
[[ -n "${_LIBERO_EXP_COMMON_LOADED:-}" ]] && return 0
_LIBERO_EXP_COMMON_LOADED=1

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
EXP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_PY="${EXP_SCRIPT_DIR}/ckpt_dir.py"
RESULTS_CSV="${AUTODL_ROOT}/experiments/results.csv"
CKPT_ARCHIVE_ROOT="${AUTODL_ROOT}/checkpoints/_archive"

# Set by run_libero_sweep.sh before each batch
SWEEP_RUN_ID="${SWEEP_RUN_ID:-}"
SWEEP_RUN_ROOT="${AUTODL_ROOT}/experiments/runs/${SWEEP_RUN_ID}"

# Fixed hyperparameters (rank / lr not swept)
LORA_RANK="${LORA_RANK:-32}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASET_NAME="${DATASET_NAME:-libero_spatial_no_noops}"
DATA_ROOT="${AUTODL_ROOT}/datasets/openvla-libero-spatial"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_TASKS="${NUM_TASKS:-3}"
WANDB_PROJECT="${WANDB_PROJECT:-openvla-LoRA-ablation}"

exp_run_dir() {
  local exp_id="$1"
  echo "${SWEEP_RUN_ROOT}/${exp_id}"
}

exp_ckpt_dir() {
  local dropout="$1" image_aug="$2" run_note="$3"
  python3 "${CKPT_PY}" \
    --lora-dropout "${dropout}" \
    --image-aug "${image_aug}" \
    --run-id-note "${run_note}"
}

exp_has_ckpt() {
  local ckpt="$1"
  [[ -f "${ckpt}/adapter_model.safetensors" ]] || [[ -f "${ckpt}/adapter_model.bin" ]]
}

# Move existing canonical checkpoint aside so retrain never overwrites in-place.
exp_archive_canonical_ckpt() {
  local ckpt="$1" exp_id="$2" reason="${3:-pre-train}"
  if [[ ! -d "${ckpt}" ]] || ! exp_has_ckpt "${ckpt}"; then
    return 0
  fi
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  local dest="${CKPT_ARCHIVE_ROOT}/${exp_id}/${ts}-${reason}"
  mkdir -p "$(dirname "${dest}")"
  echo "  [archive ckpt] ${ckpt} → ${dest}"
  mv "${ckpt}" "${dest}"
  echo "${dest}" > "${CKPT_ARCHIVE_ROOT}/${exp_id}/LATEST_ARCHIVE.txt"
}

# Per-sweep snapshot: copy weights + metadata into experiments/runs/<sweep>/<exp_id>/
exp_snapshot_run() {
  local exp_id="$1" ckpt="$2" desc="$3" train_skipped="$4" success_rate="$5" eval_log="${6:-}"
  local run_dir
  run_dir="$(exp_run_dir "${exp_id}")"
  mkdir -p "${run_dir}"

  if [[ -d "${ckpt}" ]] && exp_has_ckpt "${ckpt}"; then
    rm -rf "${run_dir}/checkpoint"
    mkdir -p "${run_dir}/checkpoint"
    cp -a "${ckpt}/." "${run_dir}/checkpoint/"
    echo "${ckpt}" > "${run_dir}/canonical_ckpt_path.txt"
  fi

  [[ -n "${eval_log}" && -f "${eval_log}" ]] && cp -a "${eval_log}" "${run_dir}/eval_libero_spatial.log"

  local desc_escaped="${desc//\"/\\\"}"
  cat > "${run_dir}/meta.json" <<EOF
{
  "sweep_run_id": "${SWEEP_RUN_ID}",
  "exp_id": "${exp_id}",
  "description": "${desc_escaped}",
  "canonical_ckpt": "${ckpt}",
  "train_skipped": ${train_skipped},
  "success_rate": "${success_rate}",
  "num_trials": ${NUM_TRIALS},
  "archived_at": "$(date -Iseconds)"
}
EOF
  echo "  [snapshot] ${run_dir}"
}

exp_train() {
  local exp_id="$1" dropout="$2" image_aug="$3" max_steps="$4" run_note="$5"
  local run_dir
  run_dir="$(exp_run_dir "${exp_id}")"
  mkdir -p "${run_dir}"
  local log="${run_dir}/train.log"

  echo "[${exp_id}] Training → ${log}"
  cd "${AUTODL_ROOT}/openvla"
  local note_args=()
  [[ -n "${run_note}" ]] && note_args=(--run_id_note "${run_note}")

  # save_steps=max_steps → only write checkpoint once at end (avoid overwriting same files 6×)
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir "${DATA_ROOT}" \
    --dataset_name "${DATASET_NAME}" \
    --run_root_dir "${AUTODL_ROOT}/checkpoints" \
    --adapter_tmp_dir "${AUTODL_ROOT}/checkpoints/adapter_tmp" \
    --lora_rank "${LORA_RANK}" \
    --lora_dropout "${dropout}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accumulation_steps 1 \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${max_steps}" \
    --save_steps "${max_steps}" \
    --image_aug "${image_aug}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY:-zhihaozhang321-george-mason-university}" \
    --save_latest_checkpoint_only True \
    "${note_args[@]}" \
    2>&1 | tee "${log}"
}

exp_eval() {
  local exp_id="$1" ckpt="$2" center_crop="$3" run_note="$4"
  local run_dir
  run_dir="$(exp_run_dir "${exp_id}")"
  mkdir -p "${run_dir}"
  local log="${run_dir}/eval_console.log"
  local eval_note="sweep-${SWEEP_RUN_ID}-${exp_id}"
  [[ -n "${run_note}" ]] && eval_note="${eval_note}--${run_note}"

  echo "[${exp_id}] Eval → ${log}"
  cd "${AUTODL_ROOT}/openvla"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "${ckpt}" \
    --task_suite_name libero_spatial \
    --center_crop "${center_crop}" \
    --num_trials_per_task "${NUM_TRIALS}" \
    --num_tasks "${NUM_TASKS}" \
    --run_id_note "${eval_note}" \
    2>&1 | tee "${log}"

  # Copy the matching structured eval log (timestamped filename) into run dir
  local eval_log
  eval_log="$(ls -t "${AUTODL_ROOT}/openvla/experiments/logs/"EVAL-libero_spatial-openvla--"${eval_note}"*.txt 2>/dev/null | head -1)"
  if [[ -n "${eval_log}" && -f "${eval_log}" ]]; then
    cp -a "${eval_log}" "${run_dir}/eval_libero_spatial.log"
    echo "${eval_log}" > "${run_dir}/eval_log_source_path.txt"
  fi
  echo "${run_dir}/eval_libero_spatial.log"
}

exp_parse_success_rate() {
  local eval_log="$1"
  if [[ -z "${eval_log}" || ! -f "${eval_log}" ]]; then
    echo ""
    return 1
  fi
  grep "Current total success rate:" "${eval_log}" | tail -1 | awk '{print $NF}'
}

exp_append_results() {
  local line="$1"
  mkdir -p "$(dirname "${RESULTS_CSV}")"
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "sweep_run_id,timestamp,exp_id,dropout,image_aug,max_steps,ckpt_dir,run_snapshot_dir,num_trials,success_rate,train_skipped,eval_log,notes" > "${RESULTS_CSV}"
  fi
  echo "${line}" >> "${RESULTS_CSV}"
}

exp_init_sweep() {
  SWEEP_RUN_ID="${SWEEP_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
  SWEEP_RUN_ROOT="${AUTODL_ROOT}/experiments/runs/${SWEEP_RUN_ID}"
  mkdir -p "${SWEEP_RUN_ROOT}"
  cp -a "${AUTODL_ROOT}/scripts/exp/experiments.conf" "${SWEEP_RUN_ROOT}/experiments.conf" 2>/dev/null || true
  echo "${SWEEP_RUN_ID}" > "${SWEEP_RUN_ROOT}/sweep_run_id.txt"
  echo "Sweep run: ${SWEEP_RUN_ID}"
  echo "Records:   ${SWEEP_RUN_ROOT}/"
}
