#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${BENCHMARKS_ROOT}/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-qwen3-Coder-30B}"
DATA_NAME="${DATA_NAME:-swe-gym}"
NUM_SHARDS=8
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-7200}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-10800}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
MAX_RETRIES="${MAX_RETRIES:-2}"
DATASET="${DATASET:-${WORKSPACE_ROOT}/data/SWE-Gym/processed/swe_gym_verified_osb.jsonl}"
SELECTION="${SELECTION:-}"
SGLANG_DIR="${SGLANG_DIR:?Set SGLANG_DIR to a healthy eight-replica cluster}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_eval_8shards_w${NUM_WORKERS}}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARKS_ROOT}/eval_outputs/${MODEL_NAME}/${DATA_NAME}/${RUN_ID}}"

if [[ -n "${UV_BIN:-}" ]]; then
  UV="${UV_BIN}"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  UV="${WORKSPACE_ROOT}/.tools/bin/uv"
fi

: "${SANDBOX_API_KEY:?Set SANDBOX_API_KEY before running this script}"
export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-10.124.120.15:30003}"
export SANDBOX_IMAGE_PREFIX="${SANDBOX_IMAGE_PREFIX:-10.124.120.15:30001}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-${SANDBOX_IMAGE_PREFIX}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.uv-cache}"

[[ -x "${UV}" ]] || { echo "ERROR: uv is not executable: ${UV}" >&2; exit 2; }
[[ -s "${DATASET}" ]] || { echo "ERROR: dataset is missing or empty: ${DATASET}" >&2; exit 2; }
[[ -s "${SGLANG_DIR}/ready.json" ]] || { echo "ERROR: SGLang ready file is missing: ${SGLANG_DIR}/ready.json" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/shards"
if [[ -f "${RUN_ROOT}/finished.status" ]]; then
  mv "${RUN_ROOT}/finished.status" \
    "${RUN_ROOT}/finished.status.before_resume_$(date +%Y%m%d_%H%M%S)"
fi
dataset_sha256="$(sha256sum "${DATASET}" | awk '{print $1}')"
dataset_lock="${RUN_ROOT}/dataset.sha256"
if [[ -s "${dataset_lock}" ]]; then
  locked_sha256="$(awk 'NR == 1 {print $1}' "${dataset_lock}")"
  [[ "${locked_sha256}" == "${dataset_sha256}" ]] || {
    echo "ERROR: dataset changed since this run was created: locked=${locked_sha256}, current=${dataset_sha256}" >&2
    exit 2
  }
else
  echo "${dataset_sha256}  ${DATASET}" > "${dataset_lock}"
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  : > "${RUN_ROOT}/shards/instances-$(printf '%02d' "${shard}")"
done

if [[ -n "${SELECTION}" ]]; then
  [[ -s "${SELECTION}" ]] || { echo "ERROR: selection is missing or empty: ${SELECTION}" >&2; exit 2; }
  duplicate_count="$(sort "${SELECTION}" | uniq -d | wc -l)"
  unknown_count="$(comm -23 <(sort "${SELECTION}") <(jq -r '.metadata.instance_id // .label' "${DATASET}" | sort) | wc -l)"
  [[ "${duplicate_count}" -eq 0 ]] || { echo "ERROR: selection contains duplicate instance IDs" >&2; exit 2; }
  [[ "${unknown_count}" -eq 0 ]] || { echo "ERROR: selection contains ${unknown_count} IDs absent from dataset" >&2; exit 2; }
  instance_source="${SELECTION}"
  dataset_rows="$(wc -l < "${SELECTION}")"
else
  instance_source="${DATASET}"
  dataset_rows="$(wc -l < "${DATASET}")"
fi

if [[ -n "${SELECTION}" ]]; then
  cat "${instance_source}"
else
  jq -r '.metadata.instance_id // .label' "${instance_source}"
fi |
  awk -v root="${RUN_ROOT}/shards" -v count="${NUM_SHARDS}" \
    '{ path=sprintf("%s/instances-%02d", root, (NR-1)%count); print > path }'

shard_rows="$(wc -l "${RUN_ROOT}"/shards/instances-* | tail -1 | awk '{print $1}')"
[[ "${shard_rows}" -eq "${dataset_rows}" ]] || {
  echo "ERROR: sharding lost rows: dataset=${dataset_rows}, shards=${shard_rows}" >&2
  exit 2
}

pids=()
cleanup_children() {
  local pid
  for pid in "${pids[@]}"; do
    kill -- "-${pid}" 2>/dev/null || true
  done
}
handle_signal() {
  cleanup_children
  exit 143
}
trap cleanup_children EXIT
trap handle_signal INT TERM

cd "${BENCHMARKS_ROOT}"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  index="$(printf '%02d' "${shard}")"
  config="${SGLANG_DIR}/configs/worker-${shard}.json"
  [[ -s "${config}" ]] || { echo "ERROR: worker config is missing: ${config}" >&2; exit 2; }
  setsid "${UV}" run swegym-infer "${config}" \
    --dataset "${DATASET}" \
    --split train \
    --workspace remote \
    --select "${RUN_ROOT}/shards/instances-${index}" \
    --num-workers "${NUM_WORKERS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --conversation-timeout "${CONVERSATION_TIMEOUT}" \
    --setup-timeout "${SETUP_TIMEOUT}" \
    --eval-timeout "${EVAL_TIMEOUT}" \
    --sandbox-lifetime "${SANDBOX_LIFETIME}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --max-retries "${MAX_RETRIES}" \
    --output-dir "${RUN_ROOT}/results" \
    --note "${DATA_NAME}_shard_${index}" \
    >>"${RUN_ROOT}/logs/shard-${index}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
pids=()

mapfile -t outputs < <(find "${RUN_ROOT}/results" -type f -name output.jsonl | sort)
if [[ "${#outputs[@]}" != "${NUM_SHARDS}" ]]; then
  echo "ERROR: expected ${NUM_SHARDS} output files, found ${#outputs[@]}." >&2
  failed=1
else
  output_rows="$(wc -l "${outputs[@]}" | tail -1 | awk '{print $1}')"
  echo "Dataset rows: ${dataset_rows}; output rows: ${output_rows}" | tee "${RUN_ROOT}/row_count.txt"
  if [[ "${output_rows}" != "${dataset_rows}" ]]; then
    echo "ERROR: expected ${dataset_rows} output rows, found ${output_rows}." >&2
    failed=1
  fi
  "${UV}" run swegym-score "${outputs[@]}" | tee "${RUN_ROOT}/score.txt" || failed=1
fi

echo "${failed}" > "${RUN_ROOT}/finished.status"
exit "${failed}"
