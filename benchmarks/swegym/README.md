# SWE-Gym Benchmark

This directory contains image-building and inference support for SWE-Gym. The
inference runner uses a local OpenHands agent and executes repository commands
inside an OpenSandbox instance.

## Dataset

- **Source**: [Paper](https://arxiv.org/abs/2412.21139)
- **Dataset**: 
  - `SWE-Gym/SWE-Gym` - Full dataset
- **Splits**: `train`

## Usage

### Environment setup

This evaluation requires Python 3.12 or newer. In the current workspace, the
following local tools and environments were created outside the repository:

- `uv 0.8.13`: `/mnt/zj-data/data/dongjie/.tools/bin/uv`
- uv-managed CPython 3.12.11: `/mnt/zj-data/data/dongjie/.uv-python`
- uv cache: `/mnt/zj-data/data/dongjie/.uv-cache`
- project environment: `/mnt/zj-data/data/dongjie/OpenHands/benchmarks/.venv`

The project dependency set includes `opensandbox 0.1.14`. The pinned
`vendor/software-agent-sdk` submodule must also be populated before syncing the
environment. Reproduce the setup with:

```bash
export WORKSPACE_ROOT=/mnt/zj-data/data/dongjie
export UV_INSTALL_DIR="$WORKSPACE_ROOT/.tools/bin"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE_ROOT/.uv-python"
export UV_CACHE_DIR="$WORKSPACE_ROOT/.uv-cache"

mkdir -p "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"
curl -LsSf https://astral.sh/uv/0.8.13/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

cd "$WORKSPACE_ROOT/OpenHands/benchmarks"
git submodule update --init vendor/software-agent-sdk
uv python install 3.12
uv sync --python 3.12
```

This workspace used the submodule commit
`93f082c033c9b1d63bde0ab3040df94ccd5d07ff`. If GitHub is unavailable, populate
the submodule from a local checkout at that exact commit before running
`uv sync`.

### Build Docker Images

You need to build Docker images for the SWE-Gym instances. Each instance requires a specific environment setup based on the repository and issue. **Note that this will consume atleast 5-6TB of disk space. Considering setting `--n-limit` to a smaller value if required**

```bash
uv run python -m benchmarks.swegym.build_images \
  --dataset SWE-Gym/SWE-Gym \
  --split train \
  --image ghcr.io/openhands/eval-agent-server \
  --target source-minimal
```

### Run inference with OpenSandbox

Use a prepared `*_osb.jsonl` dataset. Raw SWE-Gym Parquet rows do not contain
the image, setup commands, and evaluation command required by this runner.

```bash
export SANDBOX_DOMAIN="10.124.120.15:30003"
export SANDBOX_API_KEY="<your-api-key>"

uv run swegym-infer .llm_config/sglang-local.json \
  --dataset /path/to/swe_gym_8unique_easy_osb.jsonl \
  --split train \
  --workspace remote \
  --num-workers 1 \
  --n-limit 1 \
  --max-iterations 30
```

For a local SGLang OpenAI-compatible endpoint, the LLM configuration should
enable native tool calling:

```json
{
  "model": "openai/base-model",
  "base_url": "http://127.0.0.1:30000/v1",
  "api_key": "EMPTY",
  "native_tool_calling": true
}
```

The SGLang tool parser must match the model's chat template. For
`Qwen3-30B-A3B-Instruct-2507`, start SGLang with
`--tool-call-parser qwen25` and do not configure a reasoning parser. On MUSA,
also export `MATE_FORCE_JIT=1`; otherwise CUDA Graph startup can fail when a
MATE AOT shared library is not prebuilt. OpenHands uses automatic tool
selection. Do not force `tool_choice=required` in a MUSA smoke test because the
xgrammar logits-mask backend does not support MUSA.

#### Single-node TP8

On one eight-GPU node, start one TP8 SGLang replica in a persistent tmux
session:

```bash
cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks
mkdir -p eval_outputs/swegym/sglang_single

tmux new-session -d -s swegym-sglang-single \
  "cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks && \
   MATE_FORCE_JIT=1 python3 -m sglang.launch_server \
     --model-path /mnt/zj-data/models/001688/rl_models/Qwen3-30B-A3B-Instruct-2507 \
     --served-model-name base-model \
     --host 127.0.0.1 \
     --port 30000 \
     --tp-size 8 \
     --mem-fraction-static 0.7 \
     --attention-backend fa3 \
     --chunked-prefill-size -1 \
     --cuda-graph-max-bs 8 \
     --disable-overlap-schedule \
     --tool-call-parser qwen25 \
     > eval_outputs/swegym/sglang_single/server.log 2>&1"
```

Wait for CUDA Graph capture to finish, then verify the endpoint:

```bash
tail -f eval_outputs/swegym/sglang_single/server.log
curl --noproxy '*' http://127.0.0.1:30000/v1/models
```

Port `30000` is safe here because the client uses loopback. The Kubernetes
NodePort conflict only applies when a multi-node client connects through a
Ray node IP.

Start the 8-prompt by 8-rollout evaluation with eight concurrent workers. The
included script automatically finds `uv` in `PATH` or in the workspace-local
`.tools/bin` directory:

```bash
export SANDBOX_DOMAIN="10.124.120.15:30003"
export SANDBOX_API_KEY="<your-api-key>"
export LLM_CONFIG="benchmarks/swegym/configs/sglang-slime-eval.json"

NUM_WORKERS=8 \
PARALLEL_ROUNDS=1 \
RUN_NOTE="single_tp8_w8_$(date +%Y%m%d_%H%M%S)" \
./benchmarks/swegym/start_eval.sh
```

Set `UV_BIN=/absolute/path/to/uv` if `uv` is installed elsewhere.

The launcher is aligned with the sampling and budget settings in
`MT-Slime-625/examples/coding_agent_rl/start_tmux.sh`: eight prompts, eight
independent rollouts per prompt, temperature 1.0, 32K input tokens, and 2048
output tokens. It runs eight full-dataset evaluation rounds and reports the
aggregate reward as `resolved / 64`. The standalone default uses eight
concurrent workers and a 3600-second agent timeout to measure capability on one
TP8 server; set `NUM_WORKERS=4 CONVERSATION_TIMEOUT=1500` to reproduce the slime
client limits more closely. `N_ROLLOUTS`, `MAX_ITERATIONS`, and `RUN_NOTE` are
also configurable through environment variables. With multiple SGLang
replicas behind one router, set `PARALLEL_ROUNDS` to the replica count (up to
8) so independent rollout rounds run concurrently.

For an eight-node Ray cluster, `launch_sglang_cluster.py` reserves all eight
GPUs on each node and starts one independent TP8 replica per node. It writes one
LLM config per worker so the eight rollout rounds can connect directly to the
eight replicas without a router version dependency. Four workers per round
give each replica four concurrent requests while all 64 samples run together.

If a Ray cluster is not already running, set `WORKER_ADDR` to the comma-separated
worker hosts, verify passwordless root SSH, then create `hostfile` and initialize
the cluster from the benchmark root:

```bash
cd /mnt/zj-data/data/dongjie/OpenHands/benchmarks
bash benchmarks/swegym/generate_hostfile.sh
bash benchmarks/swegym/setup_env.sh
ray status
```

`setup_env.sh` stops existing Ray, SGLang, Slime, and Megatron processes on the
listed nodes. Do not run it while another job is using the cluster.

The following is the validated persistent launch for eight Qwen3-Coder TP8
replicas. Absolute paths keep the controller log, generated configs, and
`ready.json` in the same directory even though tmux starts a new shell:

```bash
BENCHMARKS_ROOT=/mnt/zj-data/data/dongjie/OpenHands/benchmarks
CLUSTER_DIR="${BENCHMARKS_ROOT}/eval_outputs/swegym/$(date +%Y%m%d_%H%M%S)_sglang_cluster_qwen3-Coder"
mkdir -p "$CLUSTER_DIR"

tmux new-session -d -s swegym-sglang-8node \
  "cd '${BENCHMARKS_ROOT}' && \
   MATE_FORCE_JIT=1 python3 benchmarks/swegym/launch_sglang_cluster.py \
     --model-path /mnt/zj-data/models/001688/rl_models/Qwen3-Coder-30B-A3B-Instruct \
     --replicas 8 \
     --worker-port 18080 \
     --log-dir '${CLUSTER_DIR}' \
     --ready-file '${CLUSTER_DIR}/ready.json' \
     --llm-config-template benchmarks/swegym/configs/sglang-cluster-eval.json \
     > '${CLUSTER_DIR}/controller.log' 2>&1"

tail -f "$CLUSTER_DIR/controller.log"
```

After the controller reports `SGLang replicas ready`, start all eight rollout
rounds from a persistent shell or tmux pane:

```bash
export SANDBOX_DOMAIN="10.124.120.15:30003"
export SANDBOX_API_KEY="<your-api-key>"
export UV_BIN="/mnt/zj-data/data/dongjie/.tools/bin/uv"
export LLM_CONFIGS="$(
  find "$CLUSTER_DIR/configs" -maxdepth 1 -name 'worker-*.json' \
    | sort \
    | paste -sd,
)"

N_ROLLOUTS=8 \
PARALLEL_ROUNDS=8 \
NUM_WORKERS=4 \
CONVERSATION_TIMEOUT=3600 \
RUN_NOTE="$(date +%Y%m%d_%H%M%S)_qwen3_coder_30b_cluster8_tp8_w4" \
./benchmarks/swegym/start_eval.sh
```

When evaluation is complete, stop only the SGLang controller session to release
the 64 GPUs:

```bash
tmux kill-session -t swegym-sglang-8node
```

The current cluster launcher uses `--tool-call-parser qwen3_coder`, matching the
Qwen3-Coder model above. Use the parser that matches the selected model's chat
template when changing model families.

On mthreads-gmi, avoid ports in Kubernetes' NodePort range (normally
`30000-32767`) for addresses advertised through the Ray node IP. A server can
be healthy on `127.0.0.1:30100` while `NODE_IP:30100` is intercepted by
kube-proxy and returns an unrelated HTTP response. The cluster launcher
therefore defaults to port `18080`. It also propagates the driver's MUSA
library environment to each Ray actor; launch it from the same prepared shell
used for a working local SGLang server.

Keep the tmux controller session alive for the duration of the evaluation. The
Ray cluster itself can remain running after that session is stopped.

### Score results

```bash
uv run swegym-score path/to/output.jsonl
```

The score is the number and percentage of instances whose prepared evaluation
command exits successfully.
