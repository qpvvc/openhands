#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$BENCHMARKS_ROOT/../.." && pwd)"

if [[ -n "${UV_BIN:-}" ]]; then
  UV="$UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "$WORKSPACE_ROOT/.tools/bin/uv" ]]; then
  UV="$WORKSPACE_ROOT/.tools/bin/uv"
else
  echo "uv was not found. Follow the Environment setup section in $SCRIPT_DIR/README.md." >&2
  exit 127
fi

: "${SANDBOX_API_KEY:?Set SANDBOX_API_KEY before running this script}"
export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-10.124.120.15:30003}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_ROOT/.uv-cache}"

DATASET="${DATASET:-$WORKSPACE_ROOT/data/SWE-Gym/processed/swe_gym_8unique_easy_osb.jsonl}"
LLM_CONFIG="${LLM_CONFIG:-$SCRIPT_DIR/configs/sglang-slime-eval.json}"
LLM_CONFIGS="${LLM_CONFIGS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$BENCHMARKS_ROOT/eval_outputs/swegym}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PARALLEL_ROUNDS="${PARALLEL_ROUNDS:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-3600}"
MAX_RETRIES="${MAX_RETRIES:-2}"
RUN_NOTE="${RUN_NOTE:-slime_aligned_$(date +%Y%m%d_%H%M%S)}"

cd "$BENCHMARKS_ROOT"
if [[ -n "$LLM_CONFIGS" ]]; then
  IFS=',' read -r -a configured_llm_configs <<< "$LLM_CONFIGS"
  llm_configs=()
  for config in "${configured_llm_configs[@]}"; do
    [[ -n "$config" ]] && llm_configs+=("$config")
  done
else
  llm_configs=("$LLM_CONFIG")
fi

run_rollout() {
  local rollout="$1"
  local llm_config="${llm_configs[$((rollout % ${#llm_configs[@]}))]}"
  "$UV" run swegym-infer "$llm_config" \
    --dataset "$DATASET" \
    --split train \
    --workspace remote \
    --num-workers "$NUM_WORKERS" \
    --max-iterations "$MAX_ITERATIONS" \
    --conversation-timeout "$CONVERSATION_TIMEOUT" \
    --setup-timeout 1200 \
    --eval-timeout 900 \
    --sandbox-lifetime 4500 \
    --max-attempts 1 \
    --max-retries "$MAX_RETRIES" \
    --output-dir "$OUTPUT_DIR" \
    --note "${RUN_NOTE}_r${rollout}"
}

for ((batch = 0; batch < N_ROLLOUTS; batch += PARALLEL_ROUNDS)); do
  pids=()
  for ((rollout = batch; rollout < N_ROLLOUTS && rollout < batch + PARALLEL_ROUNDS; rollout++)); do
    run_rollout "$rollout" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one evaluation round failed." >&2
    exit 1
  fi
done

mapfile -t outputs < <(
  find "$OUTPUT_DIR" -type f -path "*_N_${RUN_NOTE}_r*/output.jsonl" | sort
)
if [[ "${#outputs[@]}" -ne "$N_ROLLOUTS" ]]; then
  echo "Expected $N_ROLLOUTS output files, found ${#outputs[@]} under $OUTPUT_DIR." >&2
  exit 1
fi
"$UV" run swegym-score "${outputs[@]}"
