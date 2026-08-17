#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
"""Launch tensor-parallel SGLang replicas on a Ray cluster."""

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import ray


def wait_until_ready(url: str, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Process exited with status {process.returncode}: {url}"
            )
        try:
            with opener.open(url, timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


@ray.remote
class SGLangReplica:
    def __init__(
        self,
        args: argparse.Namespace,
        log_dir: str,
        process_env: dict[str, str],
        replica_index: int,
    ) -> None:
        self.args = args
        self.replica_index = replica_index
        self.gpu_ids = sorted(int(gpu_id) for gpu_id in ray.get_gpu_ids())
        if len(self.gpu_ids) != args.tp_size:
            raise RuntimeError(
                f"Replica {replica_index} requested {args.tp_size} GPUs, "
                f"but Ray assigned {self.gpu_ids}"
            )
        self.ip = ray.util.get_node_ip_address()
        self.port = args.worker_port + replica_index
        self.log_path = (
            Path(log_dir) / f"sglang-{self.ip}-replica-{replica_index:02d}.log"
        )
        self.process_env = process_env
        self.process: subprocess.Popen[str] | None = None
        self.log_file = None

    def start(self) -> str:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("w")
        process_env = os.environ.copy()
        process_env.update(self.process_env)
        # vllm-musa deliberately disables Ray's automatic MUSA visibility mask.
        # Set it explicitly so multiple TP replicas on one node cannot share cards.
        process_env["MUSA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
        self.log_file.write(
            f"Ray GPU assignment: MUSA_VISIBLE_DEVICES="
            f"{process_env['MUSA_VISIBLE_DEVICES']}\n"
        )
        self.log_file.flush()
        if self.args.sglang_python_root:
            python_root = str(Path(self.args.sglang_python_root).resolve() / "python")
            existing = process_env.get("PYTHONPATH")
            process_env["PYTHONPATH"] = (
                f"{python_root}:{existing}" if existing else python_root
            )
        process_env.setdefault("MATE_FORCE_JIT", "1")
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.args.model_path,
            "--served-model-name",
            self.args.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--tp-size",
            str(self.args.tp_size),
            "--mem-fraction-static",
            str(self.args.mem_fraction_static),
            "--attention-backend",
            "fa3",
            "--chunked-prefill-size",
            "-1",
            "--cuda-graph-max-bs",
            "8",
            "--disable-overlap-schedule",
            "--tool-call-parser",
            self.args.tool_call_parser,
        ]
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=process_env,
        )
        wait_until_ready(
            f"http://127.0.0.1:{self.port}/v1/models",
            self.process,
            self.args.startup_timeout,
        )
        return f"http://{self.ip}:{self.port}"

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
        if self.log_file is not None:
            self.log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="base-model")
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--worker-port", type=int, default=18080)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--tool-call-parser", default="qwen3_coder")
    parser.add_argument("--sglang-python-root")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--llm-config-template", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicas <= 0:
        raise ValueError(f"replicas must be positive: {args.replicas}")
    if args.tp_size <= 0:
        raise ValueError(f"tp-size must be positive: {args.tp_size}")
    if args.worker_port + args.replicas - 1 > 65535:
        raise ValueError("worker port range exceeds 65535")
    args.log_dir = str(Path(args.log_dir).resolve())
    args.ready_file = str(Path(args.ready_file).resolve())
    args.llm_config_template = str(Path(args.llm_config_template).resolve())
    ray.init(address="auto")
    propagated_env = {
        name: os.environ[name]
        for name in (
            "LD_LIBRARY_PATH",
            "MUSA_HOME",
            "MATE_FORCE_JIT",
            "SGLANG_USE_MUSA_FUSED_KERNEL",
            "VLLM_PATCH_MUSA_CUSTOM_OPS",
        )
        if name in os.environ
    }
    actors = [
        SGLangReplica.options(num_gpus=args.tp_size, num_cpus=args.tp_size).remote(
            args, args.log_dir, propagated_env, replica_index
        )
        for replica_index in range(args.replicas)
    ]
    try:
        worker_urls = ray.get([actor.start.remote() for actor in actors])
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        config_template = json.loads(Path(args.llm_config_template).read_text())
        config_dir = log_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        config_paths = []
        for index, worker_url in enumerate(worker_urls):
            config = {**config_template, "base_url": f"{worker_url}/v1"}
            config_path = config_dir / f"worker-{index}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            config_paths.append(str(config_path))
        ready_file = Path(args.ready_file)
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text(
            json.dumps(
                {"worker_urls": worker_urls, "llm_config_paths": config_paths},
                indent=2,
            )
            + "\n"
        )
        print(f"SGLang replicas ready: {worker_urls}", flush=True)
        while True:
            time.sleep(3600)
    finally:
        ray.get([actor.stop.remote() for actor in actors], timeout=60)


if __name__ == "__main__":
    main()
