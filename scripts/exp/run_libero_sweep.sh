#!/bin/bash
# Automate: train (optional) → LIBERO eval → snapshot per experiment (no overwrite across runs)
#
# Saving policy:
#   - Each experiment uses a separate checkpoints/<exp_id>/ directory (from finetune exp_id).
#   - Re-train with FORCE_RETRAIN=1 moves old weights to checkpoints/_archive/<exp_id>/<timestamp>/.
#   - Each sweep batch writes experiments/runs/<SWEEP_RUN_ID>/<exp_id>/ (train.log, eval, checkpoint copy, meta.json).
#   - results.csv appends one row per experiment (never truncates).
#
# Usage:
#   bash scripts/exp/run_libero_sweep.sh
#   SWEEP_RUN_ID=my-batch-01 bash scripts/exp/run_libero_sweep.sh D1
#   FORCE_RETRAIN=1 bash scripts/exp/run_libero_sweep.sh D1
#
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
source "${AUTODL_ROOT}/scripts/env/activate_openvla.sh"
source "${AUTODL_ROOT}/scripts/exp/libero_exp_common.sh"

CONF="${AUTODL_ROOT}/scripts/exp/experiments.conf"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "缺少数据: ${DATA_ROOT}"
  exit 1
fi

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]] && pgrep -f "vla-scripts/finetune.py" >/dev/null 2>&1; then
  echo "检测到 finetune 正在运行。请等当前训练结束后再跑 sweep，或: SKIP_GPU_CHECK=1 bash ..."
  exit 1
fi

exp_init_sweep

BASELINE_CKPT="${AUTODL_ROOT}/checkpoints/openvla-7b+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"

should_run_exp() {
  local id="$1"
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  local want
  for want in "$@"; do
    [[ "${want}" == "${id}" ]] && return 0
  done
  return 1
}

run_one() {
  local exp_id="$1" dropout="$2" image_aug="$3" max_steps="$4" run_note="$5" do_train="$6" do_eval="$7" desc="$8"

  local aug_flag="true"
  [[ "${image_aug}" == "False" || "${image_aug}" == "false" ]] && aug_flag="false"

  local ckpt
  ckpt="$(exp_ckpt_dir "${dropout}" "${aug_flag}" "${run_note}")"

  if [[ "${exp_id}" == "D0" && ! -d "${ckpt}" && -d "${BASELINE_CKPT}" ]]; then
    ckpt="${BASELINE_CKPT}"
  fi

  local center_crop="True"
  [[ "${aug_flag}" == "false" ]] && center_crop="False"

  local train_skipped=0
  local success_rate=""
  local eval_log=""
  local snapshot_dir
  snapshot_dir="$(exp_run_dir "${exp_id}")"

  echo ""
  echo "========== ${exp_id}: ${desc} =========="
  echo "  sweep_run_id=${SWEEP_RUN_ID}"
  echo "  snapshot_dir=${snapshot_dir}"
  echo "  ckpt=${ckpt}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY_RUN] train=${do_train} eval=${do_eval}"
    return 0
  fi

  if [[ "${EVAL_ONLY}" != "1" && "${do_train}" == "1" ]]; then
    if exp_has_ckpt "${ckpt}"; then
      if [[ "${FORCE_RETRAIN}" == "1" ]]; then
        exp_archive_canonical_ckpt "${ckpt}" "${exp_id}" "force-retrain"
      else
        echo "  [skip train] checkpoint exists (use FORCE_RETRAIN=1 to archive & retrain)"
        train_skipped=1
      fi
    fi
    if [[ "${train_skipped}" != "1" ]]; then
      exp_train "${exp_id}" "${dropout}" "${image_aug}" "${max_steps}" "${run_note}"
    fi
  elif [[ "${do_train}" != "1" ]]; then
    train_skipped=1
    if [[ "${exp_id}" == "D0" ]]; then
      if ! exp_has_ckpt "${ckpt}" && ! exp_has_ckpt "${BASELINE_CKPT}"; then
        echo "  [error] D0 baseline checkpoint not found."
        return 1
      fi
    fi
  fi

  if [[ "${TRAIN_ONLY}" != "1" && "${do_eval}" == "1" ]]; then
    if ! exp_has_ckpt "${ckpt}"; then
      echo "  [error] No checkpoint for eval: ${ckpt}"
      return 1
    fi
    eval_log="$(exp_eval "${exp_id}" "${ckpt}" "${center_crop}" "${run_note}")"
    success_rate="$(exp_parse_success_rate "${eval_log}" || true)"
    echo "  [eval] success_rate=${success_rate:-N/A}"
  fi

  exp_snapshot_run "${exp_id}" "${ckpt}" "${desc}" "${train_skipped}" "${success_rate}" "${eval_log}"

  exp_append_results "${SWEEP_RUN_ID},$(date -Iseconds),${exp_id},${dropout},${image_aug},${max_steps},${ckpt},${snapshot_dir},${NUM_TRIALS},${success_rate},${train_skipped},${eval_log},${desc}"
}

FILTER=("$@")
ran=0

while IFS= read -r line || [[ -n "${line}" ]]; do
  case "${line}" in ""|\#*) continue ;; esac
  IFS='|' read -r exp_id dropout image_aug max_steps run_note do_train do_eval desc <<< "${line}"
  exp_id="$(echo "${exp_id}" | xargs)"
  if ! should_run_exp "${exp_id}" "${FILTER[@]}"; then
    continue
  fi
  run_one "${exp_id}" "${dropout}" "${image_aug}" "${max_steps}" "${run_note}" "${do_train}" "${do_eval}" "${desc}"
  ((ran++)) || true
done < "${CONF}"

echo ""
echo "Done. Ran ${ran} experiment(s)."
echo "  CSV:      ${RESULTS_CSV}"
echo "  Snapshots: ${SWEEP_RUN_ROOT}/"
