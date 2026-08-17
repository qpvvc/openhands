#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${BENCHMARKS_ROOT}/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-021-32B-Agentic}"
DATA_NAME="${DATA_NAME:-scale-swe-random1000-smoke8}"
DATASET="${DATASET:-${WORKSPACE_ROOT}/data/Scale-SWE/scale-swe-random1000-final-slime.jsonl}"
SELECTION="${SELECTION:-${WORKSPACE_ROOT}/data/Scale-SWE/scale-swe-random1000-smoke8.instances}"
SGLANG_DIR="${SGLANG_DIR:?Set SGLANG_DIR to the ready single-replica cluster directory}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-7200}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-10800}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_smoke8_w${NUM_WORKERS}}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARKS_ROOT}/eval_outputs/${MODEL_NAME}/${DATA_NAME}/${RUN_ID}}"
UV="${UV_BIN:-${WORKSPACE_ROOT}/.tools/bin/uv}"
CONFIG="${SGLANG_DIR}/configs/worker-0.json"

: "${SANDBOX_API_KEY:?Set SANDBOX_API_KEY before running this script}"
export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-10.124.120.15:30003}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-10.124.120.15:30001}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.uv-cache}"

[[ -x "${UV}" ]] || { echo "ERROR: uv is not executable: ${UV}" >&2; exit 2; }
[[ -s "${DATASET}" ]] || { echo "ERROR: dataset is missing: ${DATASET}" >&2; exit 2; }
[[ -s "${SELECTION}" ]] || { echo "ERROR: selection is missing: ${SELECTION}" >&2; exit 2; }
[[ "$(wc -l < "${SELECTION}")" -eq 8 ]] || {
  echo "ERROR: smoke selection must contain exactly 8 rows" >&2
  exit 2
}
[[ -s "${SGLANG_DIR}/ready.json" ]] || {
  echo "ERROR: SGLang ready file is missing: ${SGLANG_DIR}/ready.json" >&2
  exit 2
}
[[ -s "${CONFIG}" ]] || { echo "ERROR: worker config is missing: ${CONFIG}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results"
sha256sum "${DATASET}" > "${RUN_ROOT}/dataset.sha256"

cd "${BENCHMARKS_ROOT}"
set +e
"${UV}" run swegym-infer "${CONFIG}" \
  --dataset "${DATASET}" \
  --split train \
  --workspace remote \
  --select "${SELECTION}" \
  --num-workers "${NUM_WORKERS}" \
  --max-iterations "${MAX_ITERATIONS}" \
  --conversation-timeout "${CONVERSATION_TIMEOUT}" \
  --setup-timeout "${SETUP_TIMEOUT}" \
  --eval-timeout "${EVAL_TIMEOUT}" \
  --sandbox-lifetime "${SANDBOX_LIFETIME}" \
  --max-attempts 1 \
  --max-retries 1 \
  --output-dir "${RUN_ROOT}/results" \
  --note "${DATA_NAME}" \
  >"${RUN_ROOT}/logs/smoke8.log" 2>&1
infer_status=$?
set -e

mapfile -t outputs < <(find "${RUN_ROOT}/results" -type f -name output.jsonl | sort)
failed="${infer_status}"
if [[ "${#outputs[@]}" -ne 1 ]]; then
  echo "ERROR: expected one output file, found ${#outputs[@]}" >&2
  failed=1
else
  output_rows="$(wc -l < "${outputs[0]}")"
  [[ "${output_rows}" -eq 8 ]] || {
    echo "ERROR: expected 8 output rows, found ${output_rows}" >&2
    failed=1
  }
  "${UV}" run swegym-score "${outputs[0]}" | tee "${RUN_ROOT}/score.txt" || failed=1
  abnormal_count="$(jq -s '[.[] | select(
    .error or
    (.test_result == null) or
    (.test_result.grader != "scale_swe") or
    ((.test_result.resolved != true) and (.test_result.grader_completed != true))
  )] | length' "${outputs[0]}")"
  echo "abnormal evaluator outcomes: ${abnormal_count}" | tee "${RUN_ROOT}/infra_audit.txt"
  [[ "${abnormal_count}" -eq 0 ]] || failed=1
  fake_response_limit_count="$({
    grep -R -l 'Reached maximum fake responses' "${RUN_ROOT}/results"/ 2>/dev/null || true
  } | wc -l)"
  echo "fake-response-limit outcomes: ${fake_response_limit_count}" | tee -a "${RUN_ROOT}/infra_audit.txt"
  action_instance_count="$(jq -Rr '
    fromjson?
    | select(.event_type == "ActionEvent")
    | .instance_id
  ' "${RUN_ROOT}/logs/smoke8.log" | sort -u | wc -l)"
  echo "instances with structured actions: ${action_instance_count}" | tee -a "${RUN_ROOT}/infra_audit.txt"
  [[ "${action_instance_count}" -gt 0 ]] || failed=1
  [[ "${fake_response_limit_count}" -lt "${output_rows}" ]] || failed=1
fi

echo "${failed}" > "${RUN_ROOT}/finished.status"
echo "Smoke output: ${RUN_ROOT}"
exit "${failed}"
