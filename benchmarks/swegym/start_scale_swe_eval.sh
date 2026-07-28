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

export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$WORKSPACE_ROOT/OpenHands/benchmarks/eval_outputs/scale-swe}"
export OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_OUTPUT_ROOT}"
export EVAL_LOG="${EVAL_LOG:-$EVAL_OUTPUT_ROOT/eval_$(date +%Y%m%d_%H%M%S)_scale_swe.log}"
mkdir -p "$EVAL_OUTPUT_ROOT"

if [[ -z "${LLM_CONFIGS:-}" ]]; then
  : "${CLUSTER_READY_FILE:?Set CLUSTER_READY_FILE or LLM_CONFIGS explicitly}"
  cluster_configs_dir="$(dirname "$CLUSTER_READY_FILE")/configs"
  export LLM_CONFIGS="$({ find "$cluster_configs_dir" -maxdepth 1 -name 'worker-*.json' | sort | paste -sd,; })"
fi
if [[ -z "$LLM_CONFIGS" ]]; then
  echo "No worker configs found for the requested SGLang cluster." >&2
  exit 1
fi
exec > >(tee "$EVAL_LOG") 2>&1
exec "$SCRIPT_DIR/start_eval.sh"
