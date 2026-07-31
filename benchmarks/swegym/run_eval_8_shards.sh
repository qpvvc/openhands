#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME="qwen3-Coder-30B"
DATA_NAME="scale-swe"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-100}"
CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-7200}"
SETUP_TIMEOUT="${SETUP_TIMEOUT:-1200}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
SANDBOX_LIFETIME="${SANDBOX_LIFETIME:-10800}"

cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks/benchmarks/swegym
SGLANG_DIR="${SGLANG_DIR:?Set SGLANG_DIR to a healthy eight-replica cluster}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_eval_8shards_w${NUM_WORKERS}}"
ROOT="eval_outputs/${MODEL_NAME}/${DATA_NAME}/${RUN_ID}"
mkdir -p "$ROOT/logs"

CONFIG_ROOT=${SGLANG_DIR}/configs
DATASET=/mnt/zj-data/data/dongjie/data/Scale-SWE/scale-swe-samples100-slime.jsonl
UV=/mnt/zj-data/data/dongjie/.tools/bin/uv

export SANDBOX_API_KEY=88761cba-8d25-4858-98db-9f4d107609f3
export SANDBOX_DOMAIN=10.124.120.15:30003
export SCALE_SWE_IMAGE_PREFIX=10.124.120.15:30001
export UV_CACHE_DIR=/mnt/zj-data/data/dongjie/.uv-cache

pids=()
for shard in {0..7}; do
  index="$(printf '%02d' "$shard")"
  "$UV" run swegym-infer "$CONFIG_ROOT/worker-$shard.json" \
    --dataset "$DATASET" \
    --split train \
    --workspace remote \
    --select "$ROOT/../shards/instances-$index" \
    --num-workers "$NUM_WORKERS" \
    --max-iterations "$MAX_ITERATIONS" \
    --conversation-timeout "$CONVERSATION_TIMEOUT" \
    --setup-timeout "$SETUP_TIMEOUT" \
    --eval-timeout "$EVAL_TIMEOUT" \
    --sandbox-lifetime "$SANDBOX_LIFETIME" \
    --max-attempts 1 \
    --max-retries 0 \
    --output-dir "$ROOT/results" \
    --note "scale_swe_100_shard_$index" \
    >"$ROOT/logs/shard-$index.log" 2>&1 &
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
