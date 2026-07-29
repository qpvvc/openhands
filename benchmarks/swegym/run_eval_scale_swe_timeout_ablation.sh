#!/usr/bin/env bash
# Compare a longer conversation budget on a small, fixed Scale-SWE sample.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARKS_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$BENCHMARKS_ROOT/../.." && pwd)"

MODEL_NAME="${MODEL_NAME:-qwen3-Coder-30B}"
DATA_NAME="scale-swe"
TASKS_PER_REPLICA="${TASKS_PER_REPLICA:-1}" # 1 => 8 tasks; set to 2 => 16 tasks.
MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-3600}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-5400}"

if [[ "$TASKS_PER_REPLICA" != "1" && "$TASKS_PER_REPLICA" != "2" ]]; then
  echo "TASKS_PER_REPLICA must be 1 (8 tasks) or 2 (16 tasks)." >&2
  exit 2
fi
if (( SANDBOX_LIFETIME < CONVERSATION_TIMEOUT + SETUP_TIMEOUT )); then
  echo "SANDBOX_LIFETIME must cover SETUP_TIMEOUT + CONVERSATION_TIMEOUT." >&2
  exit 2
fi

SGLANG_DIR="${SGLANG_DIR:-$SCRIPT_DIR/eval_outputs/$MODEL_NAME/$DATA_NAME/sglang_replicas8_20260729_121531}"
CONFIG_ROOT="$SGLANG_DIR/configs"
SAMPLE_SHARD_ROOT="${SAMPLE_SHARD_ROOT:-$SCRIPT_DIR/eval_outputs/$MODEL_NAME/$DATA_NAME/shards}"
DATASET="${DATASET:-$WORKSPACE_ROOT/data/Scale-SWE/scale-swe-samples100-slime.jsonl}"
UV="${UV_BIN:-$WORKSPACE_ROOT/.tools/bin/uv}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_timeout_${CONVERSATION_TIMEOUT}s_${TASKS_PER_REPLICA}per_replica}"
ROOT="$SCRIPT_DIR/eval_outputs/$MODEL_NAME/$DATA_NAME/$RUN_ID"

export SANDBOX_API_KEY=88761cba-8d25-4858-98db-9f4d107609f3
export SANDBOX_DOMAIN="${SANDBOX_DOMAIN:-10.124.120.15:30003}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-10.124.120.15:30001}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$WORKSPACE_ROOT/.uv-cache}"

for shard in {0..7}; do
  [[ -f "$CONFIG_ROOT/worker-$shard.json" ]] || {
    echo "Missing SGLang config: $CONFIG_ROOT/worker-$shard.json" >&2
    exit 2
  }
  [[ -f "$SAMPLE_SHARD_ROOT/instances-$(printf '%02d' "$shard")" ]] || {
    echo "Missing sample shard: $SAMPLE_SHARD_ROOT/instances-$(printf '%02d' "$shard")" >&2
    exit 2
  }
done

mkdir -p "$ROOT/shards" "$ROOT/logs"
for shard in {0..7}; do
  index="$(printf '%02d' "$shard")"
  sed -n "1,${TASKS_PER_REPLICA}p" "$SAMPLE_SHARD_ROOT/instances-$index" \
    > "$ROOT/shards/instances-$index"
done

selected_count="$(cat "$ROOT"/shards/instances-* | sort -u | wc -l)"
expected_count="$((8 * TASKS_PER_REPLICA))"
if [[ "$selected_count" -ne "$expected_count" ]]; then
  echo "Expected $expected_count unique selected instances, found $selected_count." >&2
  exit 2
fi

cd "$SCRIPT_DIR"
pids=()
for shard in {0..7}; do
  index="$(printf '%02d' "$shard")"
  "$UV" run swegym-infer "$CONFIG_ROOT/worker-$shard.json" \
    --dataset "$DATASET" \
    --split train \
    --workspace remote \
    --select "$ROOT/shards/instances-$index" \
    --num-workers 1 \
    --max-iterations "$MAX_ITERATIONS" \
    --conversation-timeout "$CONVERSATION_TIMEOUT" \
    --setup-timeout "$SETUP_TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --sandbox-lifetime "$SANDBOX_LIFETIME" \
    --max-attempts 1 \
    --max-retries 0 \
    --output-dir "$ROOT/results" \
    --note "scale_swe_timeout_ablation_${CONVERSATION_TIMEOUT}s_shard_$index" \
    > "$ROOT/logs/shard-$index.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

mapfile -t outputs < <(find "$ROOT/results" -type f -name output.jsonl | sort)
if [[ "${#outputs[@]}" -ne 8 ]]; then
  echo "Expected 8 output files, found ${#outputs[@]}." >&2
  failed=1
else
  "$UV" run swegym-score "${outputs[@]}" | tee "$ROOT/score.txt" || failed=1
fi
echo "$failed" > "$ROOT/finished.status"
exit "$failed"
