#!/usr/bin/env bash
set -euo pipefail
MODEL_NAME="qwen3-Coder-30B"
DATA_NAME="scale-swe"

ROOT=eval_outputs/${MODEL_NAME}/${DATA_NAME}/$(date +%Y%m%d_%H%M%S)_eval_8shards
mkdir -p "$ROOT/logs"

CONFIG_ROOT=${SGLANG_DIR}/configs
DATASET=/mnt/zj-data/data/dongjie/data/Scale-SWE/scale-swe-samples100-slime.jsonl
UV=/mnt/zj-data/data/dongjie/.tools/bin/uv

export SANDBOX_API_KEY=88761cba-8d25-4858-98db-9f4d107609f3
export SANDBOX_DOMAIN=10.124.120.15:30003
export SCALE_SWE_IMAGE_PREFIX=10.124.120.15:30001
export UV_CACHE_DIR=/mnt/zj-data/data/dongjie/.uv-cache

cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks/benchmarks/swegym
pids=()
for shard in {0..7}; do
  index="$(printf '%02d' "$shard")"
  "$UV" run swegym-infer "$CONFIG_ROOT/worker-$shard.json" \
    --dataset "$DATASET" \
    --split train \
    --workspace remote \
    --select "$ROOT/../shards/instances-$index" \
    --num-workers 8 \
    --max-iterations 40 \
    --conversation-timeout 1800 \
    --setup-timeout 1200 \
    --eval-timeout 900 \
    --sandbox-lifetime 4500 \
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
echo "$failed" > "$ROOT/finished.status"
exit "$failed"
