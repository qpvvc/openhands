#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPLICAS="${REPLICAS:-8}"
TP_SIZE="${TP_SIZE:-8}"
MODEL_PATH="${MODEL_PATH:-/mnt/zj-data/models/001688/rl_models/Qwen3-Coder-30B-A3B-Instruct}"
WORKER_PORT="${WORKER_PORT:-18080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SGLANG_DIR="${SGLANG_DIR:-${SCRIPT_DIR}/eval_outputs/qwen3-Coder-30B/swe-gym/sglang_replicas${REPLICAS}_${RUN_STAMP}}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://$(hostname -I | awk '{print $1}'):8265}"
SUBMISSION_ID="${SUBMISSION_ID:-swegym-sglang-${REPLICAS}-${RUN_STAMP}}"
WAIT_FOR_READY="${WAIT_FOR_READY:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
SGLANG_PYTHON_ROOT="${SGLANG_PYTHON_ROOT:-}"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/lib64:/usr/lib:/usr/lib/x86_64-linux-gnu:${MUSA_HOME}/lib:/usr/local/openmpi/lib}"

[[ "${REPLICAS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: REPLICAS must be a positive integer: ${REPLICAS}" >&2
  exit 2
}
[[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: TP_SIZE must be a positive integer: ${TP_SIZE}" >&2
  exit 2
}
(( WORKER_PORT + REPLICAS - 1 <= 65535 )) || {
  echo "ERROR: worker port range exceeds 65535" >&2
  exit 2
}

mkdir -p "${SGLANG_DIR}"
printf '%s\n' "${SUBMISSION_ID}" > "${SGLANG_DIR}/submission_id.txt"

extra_args=(--tool-call-parser "${TOOL_CALL_PARSER}")
if [[ -n "${SGLANG_PYTHON_ROOT}" ]]; then
  [[ -d "${SGLANG_PYTHON_ROOT}/python/sglang" ]] || {
    echo "ERROR: invalid SGLANG_PYTHON_ROOT: ${SGLANG_PYTHON_ROOT}" >&2
    exit 2
  }
  extra_args+=(--sglang-python-root "${SGLANG_PYTHON_ROOT}")
fi

ray job submit \
  --address "${RAY_DASHBOARD_ADDRESS}" \
  --submission-id "${SUBMISSION_ID}" \
  --no-wait \
  -- \
  env MATE_FORCE_JIT=1 MUSA_HOME="${MUSA_HOME}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  python3 "${SCRIPT_DIR}/launch_sglang_cluster.py" \
    --model-path "${MODEL_PATH}" \
    --replicas "${REPLICAS}" \
    --tp-size "${TP_SIZE}" \
    --worker-port "${WORKER_PORT}" \
    --startup-timeout "${STARTUP_TIMEOUT}" \
    --log-dir "${SGLANG_DIR}" \
    --ready-file "${SGLANG_DIR}/ready.json" \
    --llm-config-template "${SCRIPT_DIR}/configs/sglang-cluster-eval.json" \
    "${extra_args[@]}"

echo "Ray submission: ${SUBMISSION_ID}"
echo "SGLang output: ${SGLANG_DIR}"

if [[ "${WAIT_FOR_READY}" != 1 ]]; then
  exit 0
fi

deadline=$((SECONDS + STARTUP_TIMEOUT + 120))
until [[ -s "${SGLANG_DIR}/ready.json" ]]; do
  if (( SECONDS >= deadline )); then
    echo "ERROR: timed out waiting for ${SGLANG_DIR}/ready.json" >&2
    ray job logs --address "${RAY_DASHBOARD_ADDRESS}" "${SUBMISSION_ID}" | tail -200 >&2 || true
    exit 1
  fi
  status="$(ray job status --address "${RAY_DASHBOARD_ADDRESS}" "${SUBMISSION_ID}" 2>&1 || true)"
  if grep -Eq 'FAILED|STOPPED|SUCCEEDED' <<<"${status}"; then
    echo "ERROR: SGLang Ray job stopped before readiness: ${status}" >&2
    ray job logs --address "${RAY_DASHBOARD_ADDRESS}" "${SUBMISSION_ID}" | tail -200 >&2 || true
    exit 1
  fi
  sleep 5
done

jq . "${SGLANG_DIR}/ready.json"
