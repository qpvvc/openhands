cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks/benchmarks/swegym
export REPLICAS=8
export SGLANG_DIR="eval_outputs/qwen3-Coder-30B/scale-swe/sglang_replicas${REPLICAS}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SGLANG_DIR"

nohup bash -c 'MATE_FORCE_JIT=1 python3 launch_sglang_cluster.py \
  --model-path /mnt/zj-data/models/001688/rl_models/Qwen3-Coder-30B-A3B-Instruct \
  --replicas "$REPLICAS" \
  --worker-port 18080 \
  --log-dir "'"$SGLANG_DIR"'" \
  --ready-file "'"$SGLANG_DIR"'/ready.json" \
  --llm-config-template $PWD/configs/sglang-cluster-eval.json
' > "$SGLANG_DIR/controller.log" 2>&1 &