#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${BENCHMARKS_ROOT}/../.." && pwd)"
NANHU_SLIME_ROOT="${WORKSPACE_ROOT}/nanhu-slime"

MODEL_PROFILE="${MODEL_PROFILE:-021-32b}"
case "${MODEL_PROFILE}" in
  021-32b)
    default_model_path="/mnt/zj-data/models/public/models/021-32B-Agentic/hf"
    default_model_name="021-32B-Agentic"
    default_tool_call_parser="glm47"
    ;;
  qwen3-coder)
    default_model_path="/mnt/zj-data/models/001688/rl_models/Qwen3-Coder-30B-A3B-Instruct"
    default_model_name="Qwen3-Coder-30B"
    default_tool_call_parser="qwen3_coder"
    ;;
  *)
    echo "ERROR: MODEL_PROFILE must be 021-32b or qwen3-coder: ${MODEL_PROFILE}" >&2
    exit 2
    ;;
esac

DATASET_TYPE="${DATASET_TYPE:-scale-swe}"
case "${DATASET_TYPE}" in
  scale-swe)
    default_dataset="${WORKSPACE_ROOT}/data/Scale-SWE/scale-swe-random1000-final-slime.jsonl"
    default_data_name="scale-swe-random1000-final"
    default_dataset_rows=1000
    ;;
  swe-gym)
    default_dataset="${WORKSPACE_ROOT}/data/SWE-Gym/processed/swe_gym_verified_osb.jsonl"
    default_data_name="swe-gym-verified"
    default_dataset_rows=1819
    ;;
  *)
    echo "ERROR: DATASET_TYPE must be scale-swe or swe-gym: ${DATASET_TYPE}" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${default_model_path}}"
MODEL_NAME="${MODEL_NAME:-${default_model_name}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${default_tool_call_parser}}"
SGLANG_PYTHON_ROOT="${SGLANG_PYTHON_ROOT:-${WORKSPACE_ROOT}/nanhu-sglang}"
DATASET="${DATASET:-${default_dataset}}"
DATA_NAME="${DATA_NAME:-${default_data_name}}"
EXPECTED_DATASET_ROWS="${EXPECTED_DATASET_ROWS:-${default_dataset_rows}}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-4}"
NODE_COUNT="${WORLD_SIZE:-8}"

for value_name in GPUS_PER_NODE TP_SIZE NODE_COUNT; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${value_name} must be a positive integer: ${value}" >&2
    exit 2
  }
done
(( GPUS_PER_NODE % TP_SIZE == 0 )) || {
  echo "ERROR: GPUS_PER_NODE must be divisible by TP_SIZE" >&2
  exit 2
}

REPLICAS_PER_NODE="${REPLICAS_PER_NODE:-$((GPUS_PER_NODE / TP_SIZE))}"
REPLICAS="${REPLICAS:-$((NODE_COUNT * REPLICAS_PER_NODE))}"
NUM_SHARDS="${NUM_SHARDS:-${REPLICAS}}"
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-${NUM_WORKERS:-4}}"
SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SGLANG_DIR="${SGLANG_DIR:-${BENCHMARKS_ROOT}/eval_outputs/${MODEL_NAME}/${DATA_NAME}/sglang_tp${TP_SIZE}_r${REPLICAS}_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${BENCHMARKS_ROOT}/eval_outputs/${MODEL_NAME}/${DATA_NAME}/${RUN_STAMP}_full_tp${TP_SIZE}_r${REPLICAS}}"
SUBMISSION_ID="${SUBMISSION_ID:-${DATASET_TYPE}-${MODEL_PROFILE}-sglang-tp${TP_SIZE}-r${REPLICAS}-${RUN_STAMP}}"
export SANDBOX_CLEANUP_OWNER="${SUBMISSION_ID}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://${HEAD_IP:-${MASTER_ADDR:-127.0.0.1}}:8265}"

for value_name in REPLICAS_PER_NODE REPLICAS NUM_SHARDS EXPECTED_DATASET_ROWS SGLANG_SERVER_CONCURRENCY SWE_BOOT_CONCURRENCY OVER_SAMPLING_BATCH_SIZE; do
  value="${!value_name}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: ${value_name} must be a positive integer: ${value}" >&2
    exit 2
  }
done
[[ "${OVER_SAMPLING_BATCH_SIZE}" -eq 1 ]] || {
  echo "ERROR: direct evaluation requires OVER_SAMPLING_BATCH_SIZE=1" >&2
  exit 2
}
max_replicas_per_node=$((GPUS_PER_NODE / TP_SIZE))
(( REPLICAS_PER_NODE <= max_replicas_per_node )) || {
  echo "ERROR: REPLICAS_PER_NODE=${REPLICAS_PER_NODE} exceeds capacity ${max_replicas_per_node}" >&2
  exit 2
}
(( REPLICAS <= NODE_COUNT * REPLICAS_PER_NODE )) || {
  echo "ERROR: REPLICAS=${REPLICAS} exceeds cluster capacity $((NODE_COUNT * REPLICAS_PER_NODE))" >&2
  exit 2
}
[[ "${NUM_SHARDS}" -eq "${REPLICAS}" ]] || {
  echo "ERROR: NUM_SHARDS must match REPLICAS" >&2
  exit 2
}
[[ -s "${MODEL_PATH}/model.safetensors.index.json" ]] || {
  echo "ERROR: model is incomplete: ${MODEL_PATH}" >&2
  exit 2
}
[[ -d "${SGLANG_PYTHON_ROOT}/python/sglang" ]] || {
  echo "ERROR: invalid SGLANG_PYTHON_ROOT: ${SGLANG_PYTHON_ROOT}" >&2
  exit 2
}
[[ -s "${DATASET}" ]] || { echo "ERROR: dataset is missing: ${DATASET}" >&2; exit 2; }
dataset_rows="$(wc -l < "${DATASET}")"
[[ "${dataset_rows}" -eq "${EXPECTED_DATASET_ROWS}" ]] || {
  echo "ERROR: expected ${EXPECTED_DATASET_ROWS} dataset rows, found ${dataset_rows}" >&2
  exit 2
}
instance_rows="$(jq -r '.metadata.instance_id // .label // empty' "${DATASET}" | wc -l)"
unique_instances="$(jq -r '.metadata.instance_id // .label // empty' "${DATASET}" | sort -u | wc -l)"
[[ "${instance_rows}" -eq "${dataset_rows}" && "${unique_instances}" -eq "${dataset_rows}" ]] || {
  echo "ERROR: expected ${dataset_rows} non-empty unique instance IDs" >&2
  exit 2
}

echo "model=${MODEL_PROFILE} dataset=${DATASET_TYPE} tp=${TP_SIZE} replicas=${REPLICAS} shards=${NUM_SHARDS} server_concurrency=${SGLANG_SERVER_CONCURRENCY} boot_concurrency=${SWE_BOOT_CONCURRENCY} over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE} submission_id=${SUBMISSION_ID}"

if [[ -z "${SANDBOX_API_KEY:-}" ]]; then
  SANDBOX_ENV_FILE="${SANDBOX_ENV_FILE:-${NANHU_SLIME_ROOT}/musa_run/runtime/sandbox_connection_opensandbox.env}"
  [[ -s "${SANDBOX_ENV_FILE}" ]] || {
    echo "ERROR: sandbox environment file is missing: ${SANDBOX_ENV_FILE}" >&2
    exit 2
  }
  source "${SANDBOX_ENV_FILE}"
  export SANDBOX_API_KEY="${SLIME_AGENT_OPENSANDBOX_API_KEY}"
  export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-${SLIME_AGENT_OPENSANDBOX_DOMAIN}}"
fi
export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-10.124.120.15:30003}"
export SANDBOX_IMAGE_PREFIX="${SANDBOX_IMAGE_PREFIX:-10.124.120.15:30001}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-${SANDBOX_IMAGE_PREFIX}}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.uv-cache}"

cleanup_sglang() {
  ray job stop --address "${RAY_DASHBOARD_ADDRESS}" "${SUBMISSION_ID}" >/dev/null 2>&1 || true
}
trap cleanup_sglang EXIT INT TERM

cd "${BENCHMARKS_ROOT}"
REPLICAS="${REPLICAS}" \
TP_SIZE="${TP_SIZE}" \
MODEL_PATH="${MODEL_PATH}" \
SGLANG_DIR="${SGLANG_DIR}" \
SUBMISSION_ID="${SUBMISSION_ID}" \
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS}" \
SGLANG_PYTHON_ROOT="${SGLANG_PYTHON_ROOT}" \
TOOL_CALL_PARSER="${TOOL_CALL_PARSER}" \
WAIT_FOR_READY=1 \
bash benchmarks/swegym/run_sglang_multi_replicas.sh

MODEL_NAME="${MODEL_NAME}" \
DATA_NAME="${DATA_NAME}" \
NUM_SHARDS="${NUM_SHARDS}" \
NUM_WORKERS="${SGLANG_SERVER_CONCURRENCY}" \
SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY}" \
MAX_ITERATIONS="${MAX_ITERATIONS:-100}" \
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-7200}" \
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1200}" \
EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}" \
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-10800}" \
MAX_RETRIES="${MAX_RETRIES:-5}" \
DATASET="${DATASET}" \
SGLANG_DIR="${SGLANG_DIR}" \
RUN_ROOT="${RUN_ROOT}" \
bash benchmarks/swegym/run_eval_8_shards.sh

echo "${DATASET_TYPE} evaluation output: ${RUN_ROOT}"
