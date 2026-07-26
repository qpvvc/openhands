#!/bin/bash

# ============================================================
# Script: generate_hostfile.sh
# Description: Automatically generates a hostfile for MPI/cluster
#              environments by detecting master and worker node IPs.
#
# Usage:
#   ./generate_hostfile.sh [HOSTFILE_PATH]  # Default: ./hostfile
#
# Requirements:
#   - $WORKER_ADDR must be set (comma-separated worker hostnames/IPs)
#   - Passwordless SSH access to workers
#
# How it works:
#   1. MASTER NODE:
#      - Gets its own IP using `ip a` (first non-local IPv4)
#      - Writes "MASTER_IP slots=8" to hostfile
#
#   2. WORKER NODES:
#      - Reads $WORKER_ADDR (format: "worker1,worker2,worker3")
#      - For each worker:
#        a) SSH into the node to run `ip a`
#        b) Extracts its IP (same method as master)
#        c) Writes "WORKER_IP slots=8" to hostfile
#
#   3. ERROR HANDLING:
#      - Failed SSH attempts logged to ssh_errors.log
#      - Includes ping diagnostics for troubleshooting
#
# Output:
#   - HOSTFILE: List of nodes with IPs and GPU slots
#   - ssh_errors.log: Detailed connection issues
# ============================================================


# Default hostfile path (current directory)
HOSTFILE="${1:-./hostfile}"  # Use command-line argument or default to ./hostfile
ERROR_LOG="ssh_errors.log"

# Initialize files
> "$HOSTFILE"
> "$ERROR_LOG"

# Get master node IP
MASTER_IP=$(ip a | awk '/inet / && !/127.0.0.1/ {print $2}' | cut -d/ -f1 | head -n 1)
echo "$MASTER_IP slots=8" >> "$HOSTFILE"
echo "Master node $MASTER_IP added" | tee -a "$ERROR_LOG"

# Process worker nodes
WORKERS=$(echo "$WORKER_ADDR" | tr ',' ' ')
echo "Worker nodes: $WORKERS" | tee -a "$ERROR_LOG"

# Get worker node IPs
for WORKER in $WORKERS; do
    echo "Connecting to $WORKER..." | tee -a "$ERROR_LOG"
    
    if WORKER_IP=$(ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$WORKER" \
         "ip a | awk '/inet / && !/127.0.0.1/ {print \$2}' | cut -d/ -f1 | head -n 1" 2>/dev/null)
    then
        echo "$WORKER_IP slots=8" >> "$HOSTFILE"
        echo "Worker $WORKER added: $WORKER_IP" | tee -a "$ERROR_LOG"
    else
        # Log connection error details
        {
            echo "---- Connection failed for $WORKER ----"
            echo "Time: $(date)"
            echo "Node: $WORKER"
            echo "Error type: ${ERROR_TYPE:-Unknown error}"
            echo "Diagnostics:"
            ping -c 2 "$WORKER" | grep 'packet loss' || true
            echo "-----------------------------"
        } >> "$ERROR_LOG"
    fi
done

echo "Hostfile generated at: $HOSTFILE"
