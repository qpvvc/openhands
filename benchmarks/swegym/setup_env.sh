#!/bin/bash

# 定义hostfile路径（如果不在当前目录，请修改路径）
HOSTFILE="hostfile"

# 读取第一台机器的IP（用于其他机器连接）
HEAD_IP=$(head -n 1 "$HOSTFILE" | awk '{print $1}')

# 计数器，用于判断是否是第一台机器
COUNT=0
hostlist=$(grep -v '^#\|^$' $HOSTFILE | awk '{print $1}' | xargs)
# 读取hostfile中的每一行
for host in ${hostlist[@]}; do
    # 提取IP地址
    IP=$host
    
    echo "正在连接 $IP..."
    
    # 使用SSH执行命令（使用 nohup 让 Ray 在后台持续运行）
    if ! ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "root@$IP" << 'EOF'
        # 安装依赖
        pip install math_verify latex2sympy2_extended

        cp /mnt/seed17/001688/zhenxue.xu/torchada/src/torchada/triton/autotune/fused_moe/../../configs/triton_3_2_0/E=80,N=384,device_name=MTT_S5000.json /home/sglang/python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_3_2_0/
        
        # 先停止已有的 Ray 进程
        ray stop --force
        
	        # 清理可能残留的进程
	        pkill -f "ray::" || true
	        pkill -f "SGLangEngine|sglang|sgl-router|sglang_router" || true
	        pkill -f "python.*train.py|torchrun|megatron|slime" || true

        # 等待旧 SGLang 释放端口；超时后只强制终止该端口的监听进程。
        sglang_port=18080
        listener_pids() {
            ss -H -ltnp "sport = :$sglang_port" 2>/dev/null \
                | grep -o 'pid=[0-9]\+' \
                | cut -d= -f2 \
                | sort -u
        }

        for _ in $(seq 1 30); do
            [ -z "$(listener_pids)" ] && break
            sleep 1
        done

        port_pids=$(listener_pids)
        if [ -n "$port_pids" ]; then
            echo "端口 $sglang_port 仍被 PID $port_pids 占用，强制终止"
            kill -KILL $port_pids || true
            for _ in $(seq 1 10); do
                [ -z "$(listener_pids)" ] && break
                sleep 1
            done
        fi

        if [ -n "$(listener_pids)" ]; then
            echo "错误: 无法释放端口 $sglang_port" >&2
            ss -H -ltnp "sport = :$sglang_port" >&2 || true
            exit 1
        fi
        echo "端口 $sglang_port 已释放"
EOF
    then
        echo "错误: $IP 清理失败，停止初始化" >&2
        exit 1
    fi
    
    # 根据是否是第一台机器执行不同的启动命令（使用 nohup 和后台运行）
    if [ $COUNT -eq 0 ]; then
        echo "启动 Head 节点: $IP"
        ssh -o StrictHostKeyChecking=no "root@$IP" "nohup ray start --head --dashboard-host=0.0.0.0 --port=6379 --num-cpus=800 > /tmp/ray-head.log 2>&1 &"
        sleep 10  # 等待 head 节点完全启动
    else
        echo "启动 Worker 节点: $IP, 连接到 $HEAD_IP"
        ssh -o StrictHostKeyChecking=no "root@$IP" "nohup ray start --address=${HEAD_IP}:6379 --num-cpus=800 > /tmp/ray-worker.log 2>&1 &"
    fi
    
    COUNT=$((COUNT + 1))
    echo "已完成 $IP 的配置"
    echo "----------------------------------------"
done

# 等待所有节点注册
echo "等待所有节点注册到 Ray 集群..."
sleep 20

# 在 Head 节点上检查集群状态
echo "检查集群状态:"
ssh -o StrictHostKeyChecking=no "root@$HEAD_IP" "ray status"

echo "所有机器配置完成!"
