#!/bin/bash
# π₀.₅-LIBERO 官方 checkpoint 评测 — 协议对齐 OpenVLA eval_libero_checkpoint.sh
#
# 默认: libero_spatial 前 3 任务 × 10 trials（与 20260522-eval 一致）
#
# 用法:
#   bash scripts/eval/setup_openpi_libero.sh          # 首次
#   bash scripts/eval/eval_pi05_libero.sh
#
#   NUM_TASKS=10 NUM_TRIALS=50 bash scripts/eval/eval_pi05_libero.sh   # 全套
#
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
OPENPI_ROOT="${OPENPI_ROOT:-${AUTODL_ROOT}/openpi}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${AUTODL_ROOT}/openpi_cache}"
export PATH="${HOME}/.local/bin:${PATH}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-10}"
NUM_TASKS="${NUM_TASKS:-3}"
SEED="${SEED:-7}"
POLICY_PORT="${POLICY_PORT:-8000}"
SWEEP_RUN_ID="${SWEEP_RUN_ID:-$(date +%Y%m%d-%H%M%S)-pi05}"
RUN_NOTE="${RUN_NOTE:-${SWEEP_RUN_ID}}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"

RESULTS_CSV="${AUTODL_ROOT}/experiments/results.csv"
EVAL_LOG_DIR="${AUTODL_ROOT}/experiments/eval_logs"
VIDEO_DIR="${AUTODL_ROOT}/experiments/rollouts_pi05/$(date +%Y_%m_%d)"
RUN_DIR="${AUTODL_ROOT}/experiments/runs/${SWEEP_RUN_ID}/pi05-libero"
LOG_FILE="${RUN_DIR}/eval_libero_spatial.log"
SERVER_LOG="${RUN_DIR}/policy_server.log"

CKPT_REF="gs://openpi-assets/checkpoints/pi05_libero"

mkdir -p "${EVAL_LOG_DIR}" "${VIDEO_DIR}" "${RUN_DIR}" "${OPENPI_DATA_HOME}"

if [[ ! -d "${OPENPI_ROOT}/.git" ]]; then
  echo "请先克隆 openpi 到 ${OPENPI_ROOT}"
  exit 1
fi

bash "${AUTODL_ROOT}/scripts/eval/setup_openpi_libero.sh"

# --- Start policy server (official pi05_libero via --env LIBERO) ---
if ss -ltn 2>/dev/null | grep -q ":${POLICY_PORT} "; then
  echo "端口 ${POLICY_PORT} 已被占用，请先结束旧 policy server"
  exit 1
fi

echo "Starting policy server (pi05_libero) → ${SERVER_LOG}"
cd "${OPENPI_ROOT}"
nohup env OPENPI_DATA_HOME="${OPENPI_DATA_HOME}" \
  uv run scripts/serve_policy.py --env LIBERO --port "${POLICY_PORT}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${RUN_DIR}/policy_server.pid"

cleanup() {
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping policy server (pid ${SERVER_PID})"
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 2
  fi
}
trap cleanup EXIT

POLICY_WAIT_MAX="${POLICY_WAIT_MAX:-720}"  # 720×5s ≈ 60min (first-time ckpt ~12GB)
echo "Waiting for policy server on port ${POLICY_PORT} (max ${POLICY_WAIT_MAX} attempts)…"
for i in $(seq 1 "${POLICY_WAIT_MAX}"); do
  if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',${POLICY_PORT})); s.close()" 2>/dev/null; then
    echo "Policy server ready."
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Policy server exited early. Log:"
    tail -30 "${SERVER_LOG}"
    exit 1
  fi
  if [[ "${i}" -eq "${POLICY_WAIT_MAX}" ]]; then
    echo "Timeout waiting for policy server."
    tail -30 "${SERVER_LOG}"
    exit 1
  fi
  sleep 5
done

# --- LIBERO eval client ---
echo "Checkpoint:  ${CKPT_REF} (official)"
echo "Task suite:  ${TASK_SUITE}"
echo "tasks:       ${NUM_TASKS}"
echo "trials/task: ${NUM_TRIALS}"
echo "Videos:      ${VIDEO_DIR}"
echo "Log:         ${LOG_FILE}"

cd "${OPENPI_ROOT}"
# shellcheck source=/dev/null
source examples/libero/.venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:${OPENPI_ROOT}/third_party/libero"

set +e
python examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.port "${POLICY_PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS}" \
  --args.num-tasks "${NUM_TASKS}" \
  --args.seed "${SEED}" \
  --args.video-out-path "${VIDEO_DIR}" \
  --args.log-path "${LOG_FILE}" \
  --args.run-id-note "${RUN_NOTE}" \
  2>&1 | tee "${RUN_DIR}/eval_console.log"
EVAL_RC=${PIPESTATUS[0]}
set -e
deactivate

cp -a "${LOG_FILE}" "${EVAL_LOG_DIR}/EVAL-${TASK_SUITE}-pi05-${RUN_NOTE}.txt" 2>/dev/null || true

# --- Parse & append results.csv ---
SUCCESS_RATE=""
if [[ -f "${LOG_FILE}" ]]; then
  SUCCESS_RATE=$(grep "Current total success rate:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
  [[ -z "${SUCCESS_RATE}" ]] && SUCCESS_RATE=$(grep "^Total success rate:" "${LOG_FILE}" | tail -1 | awk '{print $NF}')
fi

TS="$(date -Iseconds)"
NOTES="${NUM_TASKS} tasks x ${NUM_TRIALS} trials official pi05_libero dual-cam"
LINE="${SWEEP_RUN_ID},${TS},pi05-libero,,,0,${CKPT_REF},${RUN_DIR},${NUM_TRIALS},${SUCCESS_RATE},1,experiments/eval_logs/EVAL-${TASK_SUITE}-pi05-${RUN_NOTE}.txt,${NOTES}"

if [[ ! -f "${RESULTS_CSV}" ]]; then
  echo "sweep_run_id,timestamp,exp_id,dropout,image_aug,max_steps,ckpt_dir,run_snapshot_dir,num_trials,success_rate,train_skipped,eval_log,notes" > "${RESULTS_CSV}"
fi
echo "${LINE}" >> "${RESULTS_CSV}"

cat > "${RUN_DIR}/meta.json" <<EOF
{
  "sweep_run_id": "${SWEEP_RUN_ID}",
  "exp_id": "pi05-libero",
  "model": "pi05_libero",
  "checkpoint": "${CKPT_REF}",
  "task_suite": "${TASK_SUITE}",
  "num_tasks": ${NUM_TASKS},
  "num_trials": ${NUM_TRIALS},
  "success_rate": "${SUCCESS_RATE}",
  "eval_log": "${LOG_FILE}",
  "videos": "${VIDEO_DIR}"
}
EOF

echo ""
echo "=========================================="
echo "π₀.₅-LIBERO eval finished (exit=${EVAL_RC})"
echo "  success_rate: ${SUCCESS_RATE}"
echo "  log:          ${LOG_FILE}"
echo "  videos:       ${VIDEO_DIR}"
echo "  results.csv:  appended"
echo "=========================================="

exit "${EVAL_RC}"
