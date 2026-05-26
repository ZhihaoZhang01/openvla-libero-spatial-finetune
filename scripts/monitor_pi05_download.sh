#!/bin/bash
# Monitor pi05_libero checkpoint download; append status to log every INTERVAL sec.
set -eo pipefail

AUTODL_ROOT="${AUTODL_ROOT:-/root/autodl-tmp}"
LOG="${POLICY_LOG:-${AUTODL_ROOT}/experiments/runs/20260526-192106-pi05/pi05-libero/policy_server.log}"
OUT="${MONITOR_LOG:-${AUTODL_ROOT}/experiments/runs/pi05-download-monitor.log}"
INTERVAL="${INTERVAL:-300}"  # 5 min

mkdir -p "$(dirname "$OUT")"

while true; do
  TS="$(date -Iseconds)"
  PROGRESS="$(grep 'Progress on:' "$LOG" 2>/dev/null | tail -1 | sed 's/.*Progress on: //' | sed 's/ postfix:.*//')"
  CACHE="$(du -sh "${AUTODL_ROOT}/openpi_cache" 2>/dev/null | awk '{print $1}')"
  DISK="$(df -h "${AUTODL_ROOT}" 2>/dev/null | tail -1 | awk '{print $4" free ("$5" used)"}')"
  SERVER="$(ps aux | grep -c '[s]erve_policy.py' || true)"
  PORT="down"
  python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',8000)); s.close()" 2>/dev/null && PORT="up"

  if [[ -z "$PROGRESS" ]]; then
    PROGRESS="(no progress line yet)"
  fi

  echo "[$TS] progress=${PROGRESS} cache=${CACHE} disk=${DISK} server_procs=${SERVER} port8000=${PORT}" >> "$OUT"
  sleep "$INTERVAL"
done
