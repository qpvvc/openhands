#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${BENCHMARKS_ROOT}/../.." && pwd)"
NANHU_SLIME_ROOT="${WORKSPACE_ROOT}/nanhu-slime"
EVAL_OUTPUTS_ROOT="${SCRIPT_DIR}/eval_outputs"

MODEL_PROFILE="${MODEL_PROFILE:-021-32b-agentic}"
case "${MODEL_PROFILE}" in
  021-32b|021-32b-agentic)
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
    echo "ERROR: MODEL_PROFILE must be 021-32b-agentic, 021-32b, or qwen3-coder: ${MODEL_PROFILE}" >&2
    exit 2
    ;;
esac

MODEL_PATH="${MODEL_PATH:-${default_model_path}}"
MODEL_NAME="${MODEL_NAME:-${default_model_name}}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${default_tool_call_parser}}"
SGLANG_PYTHON_ROOT="${SGLANG_PYTHON_ROOT:-${WORKSPACE_ROOT}/nanhu-sglang}"

DATASET_SOURCE="${DATASET_SOURCE:-${WORKSPACE_ROOT}/data/SWE-bench_Verified}"
DATA_NAME="${DATA_NAME:-swe-bench-verified}"
EXPECTED_DATASET_ROWS="${EXPECTED_DATASET_ROWS:-500}"
HARBOR_IMAGE_PREFIX="${HARBOR_IMAGE_PREFIX:-10.124.120.15:30001/zj021/swebench}"
HARBOR_REGISTRY="${HARBOR_IMAGE_PREFIX%%/*}"
VERIFY_HARBOR_IMAGES="${VERIFY_HARBOR_IMAGES:-1}"
HARBOR_USERNAME="${HARBOR_USERNAME:-zjdev}"
HARBOR_PASSWORD="${HARBOR_PASSWORD:-}"
HARBOR_DOCKER_CONFIG="${HARBOR_DOCKER_CONFIG:-}"
HARBOR_API_URL="${HARBOR_API_URL:-http://10.124.120.15:30001}"
PREPARED_DATASET="${PREPARED_DATASET:-${EVAL_OUTPUTS_ROOT}/data/swe_bench_verified_osb.jsonl}"

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
SGLANG_DIR="${SGLANG_DIR:-${EVAL_OUTPUTS_ROOT}/${MODEL_NAME}/${DATA_NAME}/sglang_tp${TP_SIZE}_r${REPLICAS}_${RUN_STAMP}}"
RUN_ROOT="${RUN_ROOT:-${EVAL_OUTPUTS_ROOT}/${MODEL_NAME}/${DATA_NAME}/${RUN_STAMP}_tp${TP_SIZE}_r${REPLICAS}}"
SUBMISSION_ID="${SUBMISSION_ID:-swebench-verified-${MODEL_PROFILE}-tp${TP_SIZE}-r${REPLICAS}-${RUN_STAMP}}"
export SANDBOX_CLEANUP_OWNER="${SUBMISSION_ID}"
export MASTER_ADDR=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
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
[[ -d "${DATASET_SOURCE}" || -f "${DATASET_SOURCE}" ]] || {
  echo "ERROR: SWE-bench Verified source is missing: ${DATASET_SOURCE}" >&2
  exit 2
}

if [[ -n "${UV_BIN:-}" ]]; then
  UV="${UV_BIN}"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  UV="${WORKSPACE_ROOT}/.tools/bin/uv"
fi
[[ -x "${UV}" ]] || { echo "ERROR: uv is not executable: ${UV}" >&2; exit 2; }
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORKSPACE_ROOT}/.uv-cache}"
export HARBOR_REGISTRY HARBOR_USERNAME HARBOR_PASSWORD HARBOR_DOCKER_CONFIG HARBOR_API_URL

prepare_args=(
  --source "${DATASET_SOURCE}"
  --output "${PREPARED_DATASET}"
  --image-prefix "${HARBOR_IMAGE_PREFIX}"
)
if [[ "${VERIFY_HARBOR_IMAGES}" == 1 ]]; then
  prepare_args+=(--verify-images)
fi

echo "model=${MODEL_PROFILE} dataset=${DATA_NAME} tp=${TP_SIZE} replicas=${REPLICAS} shards=${NUM_SHARDS} server_concurrency=${SGLANG_SERVER_CONCURRENCY} boot_concurrency=${SWE_BOOT_CONCURRENCY} harbor=${HARBOR_IMAGE_PREFIX}"
cd "${BENCHMARKS_ROOT}"
"${UV}" run python -m benchmarks.swegym.prepare_swe_bench_verified "${prepare_args[@]}"

[[ -s "${PREPARED_DATASET}" ]] || {
  echo "ERROR: prepared dataset is missing: ${PREPARED_DATASET}" >&2
  exit 2
}
dataset_rows="$(wc -l < "${PREPARED_DATASET}")"
[[ "${dataset_rows}" -eq "${EXPECTED_DATASET_ROWS}" ]] || {
  echo "ERROR: expected ${EXPECTED_DATASET_ROWS} prepared rows, found ${dataset_rows}" >&2
  exit 2
}
instance_rows="$(jq -r '.metadata.instance_id // .label // empty' "${PREPARED_DATASET}" | wc -l)"
unique_instances="$(jq -r '.metadata.instance_id // .label // empty' "${PREPARED_DATASET}" | sort -u | wc -l)"
[[ "${instance_rows}" -eq "${dataset_rows}" && "${unique_instances}" -eq "${dataset_rows}" ]] || {
  echo "ERROR: prepared dataset instance IDs are not non-empty and unique" >&2
  exit 2
}

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
export SANDBOX_IMAGE_PREFIX="${SANDBOX_IMAGE_PREFIX:-${HARBOR_REGISTRY}}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-${SANDBOX_IMAGE_PREFIX}}"

cleanup_sglang() {
  ray job stop --address "${RAY_DASHBOARD_ADDRESS}" "${SUBMISSION_ID}" >/dev/null 2>&1 || true
}
trap cleanup_sglang EXIT INT TERM

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
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}" \
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-10800}" \
MAX_RETRIES="${MAX_RETRIES:-5}" \
DATASET="${PREPARED_DATASET}" \
SGLANG_DIR="${SGLANG_DIR}" \
RUN_ROOT="${RUN_ROOT}" \
bash benchmarks/swegym/run_eval_8_shards.sh

echo "SWE-bench Verified evaluation output: ${RUN_ROOT}"
