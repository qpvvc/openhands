#!/bin/bash
# Nanhu platform launcher for distributed MUSA jobs.
#
# Mirrors verl's run_cmd/run_nanhu_training_job_verl.sh: the platform starts this
# script on every allocated node and injects RANK / MASTER_ADDR / WORLD_SIZE.
#   - RANK 0  -> start the Ray head, wait for workers, then run the entry
#                script, which `ray job submit`s to the head and BLOCKS until the
#                job finishes (so the platform keeps the allocation).
#   - RANK>0  -> start a Ray worker that joins the head and stays alive.
#
# Select the entry script with TRAINING_SCRIPT (a bare filename in this dir, or
# an absolute path). Evaluation profiles are selected through environment variables:
#
#   MODEL_PROFILE=qwen3-coder DATASET_TYPE=swe-gym TP_SIZE=4 \
#       bash run_nanhu_training_job_musa.sh

echo "RANK=${RANK}, MASTER_ADDR=${MASTER_ADDR}, WORLD_SIZE=${WORLD_SIZE}"
date '+%Y-%m-%d %H:%M:%S'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "${SCRIPT_DIR}"

# Some cluster nodes lack jq; install it on every node before launching.
# Prefer the shared-workspace static binary, fall back to downloading it.
command -v jq >/dev/null 2>&1 || (install -m 755 "${SCRIPT_DIR}/.local/bin/jq" /usr/local/bin/jq 2>/dev/null || (curl -fsSL -o /usr/local/bin/jq https://github.com/jqlang/jq/releases/download/jq-1.7.1/jq-linux-amd64 && chmod +x /usr/local/bin/jq))

# MUSA runtime libs; keep loopback/head off any proxy; pin the collective NIC.
export LD_LIBRARY_PATH="/usr/local/musa/lib:${LD_LIBRARY_PATH}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"

RAY_PORT="${RAY_PORT:-6379}"

if [ "${RANK:-0}" -eq 0 ]; then
    # === HEAD NODE (RANK=0) ===
    echo "$(date '+%F %T'), starting Ray head (port ${RAY_PORT}, dashboard :8265)..."
    ray start --head --port="${RAY_PORT}" --dashboard-host=0.0.0.0 --disable-usage-stats

    if [ "${WORLD_SIZE:-1}" -ge 2 ]; then
        echo "$(date '+%F %T'), waiting 60s for $(( WORLD_SIZE - 1 )) worker(s) to join..."
        sleep 60
    else
        sleep 5
    fi
    ray status

    # Pick the entry script (bare filename in this dir, or an absolute path).
    TRAINING_SCRIPT="${TRAINING_SCRIPT:-run_021_32b_agentic_scale_swe_eval.sh}"
    if [[ "${TRAINING_SCRIPT}" = /* ]]; then
        SCRIPT_PATH="${TRAINING_SCRIPT}"
    else
        SCRIPT_PATH="${SCRIPT_DIR}/${TRAINING_SCRIPT}"
    fi
    if [ ! -f "${SCRIPT_PATH}" ]; then
        echo "ERROR: training script not found: ${SCRIPT_PATH}"
        echo "       Set TRAINING_SCRIPT to a filename in ${SCRIPT_DIR}/ or an absolute path."
        ls "${SCRIPT_DIR}"/*.sh 2>/dev/null | sed 's/^/  - /'
        exit 1
    fi

    # The training script submits to http://${HEAD_IP}:8265 and must block so the
    # platform allocation is held until training completes.
    export HEAD_IP="${MASTER_ADDR:-127.0.0.1}"
    export RAY_JOB_WAIT=1

    echo "$(date '+%F %T'), launching job: ${SCRIPT_PATH}"
    bash "${SCRIPT_PATH}"
    code=$?
    echo "$(date '+%F %T'), job finished (exit ${code})."
    exit "${code}"
else
    # === WORKER NODE (RANK>0) ===
    echo "$(date '+%F %T'), worker RANK=${RANK}, waiting 10s for head..."
    sleep 10
    echo "$(date '+%F %T'), joining Ray head at ${MASTER_ADDR}:${RAY_PORT}..."
    ray start --address="${MASTER_ADDR}:${RAY_PORT}"
    ray status
    # keep the worker alive while training runs on the head
    sleep 360000
fi
