cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks

export CLUSTER_DIR="/mnt/zj-data/data/dongjie/OpenHands/benchmarks/eval_outputs/swegym/20260724_134803_sglang_cluster_qwen3-Coder"
export SANDBOX_DOMAIN="10.124.120.15:30003"
export SANDBOX_API_KEY="88761cba-8d25-4858-98db-9f4d107609f3"
export UV_BIN="/mnt/zj-data/data/dongjie/.tools/bin/uv"

export LLM_CONFIGS="$(
find "$CLUSTER_DIR/configs" -maxdepth 1 -name 'worker-*.json' \
  | sort \
  | paste -sd,
)"

export N_ROLLOUTS=8
export PARALLEL_ROUNDS=8
export NUM_WORKERS=8
export CONVERSATION_TIMEOUT=3600
export RUN_NOTE="$(date +%Y%m%d_%H%M%S)_qwen3_coder_30b_cluster8_tp8_w${NUM_WORKERS}"
export EVAL_LOG="/mnt/zj-data/data/dongjie/OpenHands/benchmarks/eval_outputs/swegym/eval_${RUN_NOTE}.log"

tmux set-environment -g SANDBOX_DOMAIN "$SANDBOX_DOMAIN"
tmux set-environment -g SANDBOX_API_KEY "$SANDBOX_API_KEY"
tmux set-environment -g UV_BIN "$UV_BIN"
tmux set-environment -g LLM_CONFIGS "$LLM_CONFIGS"
tmux set-environment -g N_ROLLOUTS "$N_ROLLOUTS"
tmux set-environment -g PARALLEL_ROUNDS "$PARALLEL_ROUNDS"
tmux set-environment -g NUM_WORKERS "$NUM_WORKERS"
tmux set-environment -g CONVERSATION_TIMEOUT "$CONVERSATION_TIMEOUT"
tmux set-environment -g RUN_NOTE "$RUN_NOTE"

tmux new-session -d -s swegym-eval-8node \
"cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks && \
 ./benchmarks/swegym/start_eval.sh > '$EVAL_LOG' 2>&1"