#!/usr/bin/env bash
# Small, independent Scale-SWE capability baseline using the OpenHands runner.
# Start the TP8/cluster SGLang replicas first (start_cluster.sh is compatible);
# this script only changes dataset and diagnostic-scale defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"

export DATASET="${DATASET:-$WORKSPACE_ROOT/data/Scale-SWE/scale-swe-samples100-slime.jsonl}"
export SCALE_SWE_IMAGE_PREFIX="${SCALE_SWE_IMAGE_PREFIX:-10.124.120.15:30001}"
export N_ROLLOUTS="${N_ROLLOUTS:-1}"
export PARALLEL_ROUNDS="${PARALLEL_ROUNDS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-1}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-40}"
export CONVERSATION_TIMEOUT="${CONVERSATION_TIMEOUT:-600}"
export RUN_NOTE="${RUN_NOTE:-$(date +%Y%m%d_%H%M%S)_scale_swe_baseline}"

exec "$SCRIPT_DIR/start_eval.sh"
