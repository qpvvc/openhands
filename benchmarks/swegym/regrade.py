#!/usr/bin/env python3
"""Re-run SWE-Gym grading for saved candidate patches in fresh sandboxes."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from benchmarks.swegym.run_infer import SWEGymEvaluation, normalize_row
from benchmarks.utils.models import EvalInstance


_NUMERIC_THREAD_ENV = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class _PatchEvaluator:
    _evaluate_swe_gym_patch = SWEGymEvaluation._evaluate_swe_gym_patch

    def __init__(
        self,
        setup_timeout: int,
        eval_timeout: int,
        sandbox_lifetime: int,
        resource_factor: int,
    ) -> None:
        self.setup_timeout = setup_timeout
        self.eval_timeout = eval_timeout
        self.sandbox_lifetime = sandbox_lifetime
        self.resource_factor = resource_factor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def _index_unique(
    records: list[dict[str, Any]], key: str, source: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{source}: record has no non-empty {key}")
        if value in indexed:
            raise ValueError(f"{source}: duplicate {key} {value}")
        indexed[value] = record
    return indexed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _candidate_patch(original: dict[str, Any], mode: str) -> str:
    if mode == "empty":
        return ""
    return str(original.get("test_result", {}).get("git_patch") or "")


def _with_eval_thread_limit(
    instance_data: dict[str, Any], thread_limit: int | None
) -> dict[str, Any]:
    if thread_limit is None:
        return instance_data
    assignments = " ".join(f"{name}={thread_limit}" for name in _NUMERIC_THREAD_ENV)
    return {
        **instance_data,
        "eval_cmd": f"export {assignments}; {instance_data['eval_cmd']}",
    }


def _with_parallel_test_batches(
    instance_data: dict[str, Any], parallel_batches: int | None
) -> dict[str, Any]:
    if parallel_batches is None:
        return instance_data
    total = len(instance_data.get("fail_to_pass") or []) + len(
        instance_data.get("pass_to_pass") or []
    )
    if total < 1:
        raise ValueError("parallel test batches require at least one expected test")
    batch_size = (total + parallel_batches - 1) // parallel_batches
    launches = ["pids=()"]
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        launches.extend(
            [
                f"python .swe_eval/run_tests.py {start} {stop} &",
                'pids+=("$!")',
            ]
        )
    launches.extend(
        [
            "status=0",
            'for pid in "${pids[@]}"; do',
            '  wait "$pid" || status=$?',
            "done",
            'exit "$status"',
        ]
    )
    marker = "python .swe_eval/run_tests.py\nOPENHANDS_SWE_GYM_EVAL_SCRIPT"
    commands = str(instance_data["evaluator_pre_commands"])
    if commands.count(marker) != 1:
        raise ValueError("could not locate the SWE-Gym runner invocation")
    replacement = "\n".join(launches) + "\nOPENHANDS_SWE_GYM_EVAL_SCRIPT"
    return {
        **instance_data,
        "evaluator_pre_commands": commands.replace(marker, replacement),
    }


def _read_candidate_patch_dirs(paths: list[Path]) -> dict[str, str]:
    patches: dict[str, str] = {}
    suffix = ".candidate.patch"
    for directory in paths:
        for path in sorted(directory.glob(f"*{suffix}")):
            instance_id = path.name[: -len(suffix)]
            if instance_id in patches:
                raise ValueError(f"duplicate persisted candidate patch {instance_id}")
            patches[instance_id] = path.read_text(encoding="utf-8")
    return patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument(
        "--candidate-patch-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing <instance_id>.candidate.patch files.",
    )
    parser.add_argument("--select", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setup-timeout", type=int, default=1200)
    parser.add_argument("--eval-timeout", type=int, default=900)
    parser.add_argument("--override-eval-timeout", action="store_true")
    parser.add_argument("--sandbox-lifetime", type=int, default=3600)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--resource-factor",
        type=int,
        default=1,
        help="Scale evaluator sandbox CPU and memory for infrastructure controls.",
    )
    parser.add_argument(
        "--eval-thread-limit",
        type=int,
        help="Override numeric-library thread limits only for the evaluator command.",
    )
    parser.add_argument(
        "--parallel-test-batches",
        type=int,
        help="Run disjoint exact-test ranges concurrently for a high-resource control.",
    )
    parser.add_argument(
        "--candidate-patch-mode",
        choices=("saved", "empty"),
        default="saved",
        help="Use the saved model patch, or an empty patch for a baseline control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 1:
        raise ValueError("num-workers must be at least 1")
    if args.resource_factor < 1:
        raise ValueError("resource-factor must be at least 1")
    if args.eval_thread_limit is not None and args.eval_thread_limit < 1:
        raise ValueError("eval-thread-limit must be at least 1")
    if args.parallel_test_batches is not None and args.parallel_test_batches < 1:
        raise ValueError("parallel-test-batches must be at least 1")
    dataset_rows = _read_jsonl(args.dataset)
    dataset = _index_unique(
        [normalize_row(row) for row in dataset_rows], "instance_id", "dataset"
    )

    saved_rows: list[dict[str, Any]] = []
    for path in args.result:
        saved_rows.extend(_read_jsonl(path))
    saved = _index_unique(saved_rows, "instance_id", "saved results")
    persisted_patches = _read_candidate_patch_dirs(args.candidate_patch_dir)

    selected = [
        line.split()[0] for line in args.select.read_text().splitlines() if line.strip()
    ]
    if len(selected) != len(set(selected)):
        raise ValueError("selection contains duplicate instance IDs")
    missing_dataset = sorted(set(selected) - set(dataset))
    missing_results = (
        sorted(set(selected) - set(saved) - set(persisted_patches))
        if args.candidate_patch_mode == "saved"
        else []
    )
    if missing_dataset or missing_results:
        raise ValueError(
            f"missing dataset IDs={missing_dataset}; missing result IDs={missing_results}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = _PatchEvaluator(
        setup_timeout=args.setup_timeout,
        eval_timeout=args.eval_timeout,
        sandbox_lifetime=args.sandbox_lifetime,
        resource_factor=args.resource_factor,
    )

    def grade(selected_item: tuple[int, str]) -> None:
        position, instance_id = selected_item
        output_path = args.output_dir / f"{instance_id}.json"
        if output_path.is_file():
            print(f"[{position}/{len(selected)}] skip {instance_id}", flush=True)
            return

        print(f"[{position}/{len(selected)}] grade {instance_id}", flush=True)
        original = saved.get(instance_id, {})
        git_patch = (
            ""
            if args.candidate_patch_mode == "empty"
            else persisted_patches.get(instance_id, _candidate_patch(original, "saved"))
        )
        instance_data = _with_eval_thread_limit(
            dict(dataset[instance_id]), args.eval_thread_limit
        )
        instance_data = _with_parallel_test_batches(
            instance_data, args.parallel_test_batches
        )
        if args.override_eval_timeout:
            instance_data["eval_timeout_sec"] = args.eval_timeout
        try:
            result = evaluator._evaluate_swe_gym_patch(
                EvalInstance(id=instance_id, data=instance_data),
                git_patch,
                resource_factor=evaluator.resource_factor,
            )
            payload = {
                "instance_id": instance_id,
                "candidate_patch_mode": args.candidate_patch_mode,
                "resource_factor": args.resource_factor,
                "eval_thread_limit": args.eval_thread_limit,
                "parallel_test_batches": args.parallel_test_batches,
                "original_reason": original.get("test_result", {}).get("reason"),
                "corrected_reason": result.get("reason") or "resolved",
                **result,
            }
        except Exception as exc:
            payload = {
                "instance_id": instance_id,
                "candidate_patch_mode": args.candidate_patch_mode,
                "resource_factor": args.resource_factor,
                "eval_thread_limit": args.eval_thread_limit,
                "parallel_test_batches": args.parallel_test_batches,
                "original_reason": original.get("test_result", {}).get("reason"),
                "corrected_reason": "regrade_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        _write_json(output_path, payload)
        print(
            f"[{position}/{len(selected)}] {instance_id}: "
            f"{payload['corrected_reason']}",
            flush=True,
        )

    selected_items = list(enumerate(selected, 1))
    if args.num_workers == 1:
        for item in selected_items:
            grade(item)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(executor.map(grade, selected_items))


if __name__ == "__main__":
    main()
