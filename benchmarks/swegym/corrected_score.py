#!/usr/bin/env python3
"""Audit and score SWE-Gym outputs after infrastructure corrections."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


_TEST_PATCH_SENTINEL = "OPENHANDS_SWE_GYM_TEST_PATCH_APPLIED=1"
_ABNORMAL_EVAL_EXIT_CODES = {75, 137, 139, 247}


def _read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
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
    records: Iterable[dict[str, Any]], key: str, source: str
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


def _result_reason(record: dict[str, Any]) -> str:
    result = record.get("test_result")
    if not isinstance(result, dict):
        return "missing_test_result"
    if result.get("resolved") is True:
        return "resolved"
    correction = record.get("correction")
    if (
        isinstance(correction, dict)
        and correction.get("kind") == "evaluator_regrade"
        and correction.get("corrected_reason") == "tests_failed"
    ):
        return "tests_failed"
    reason = result.get("reason")
    exit_code = result.get("eval_exit_code")
    stdout = str(result.get("eval_stdout") or "")
    pytest_output = stdout.rsplit(_TEST_PATCH_SENTINEL, 1)[-1]
    if (
        exit_code in _ABNORMAL_EVAL_EXIT_CODES
        and reason == "tests_failed"
        and "F" not in pytest_output
    ):
        return "eval_process_killed"
    return str(reason) if reason else "unknown"


def _load_regrades(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    layered: dict[str, dict[str, Any]] = {}
    for directory in paths:
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"{path}: expected a JSON object")
            records.append(payload)
        layer = _index_unique(records, "instance_id", f"regrades in {directory}")
        layered.update(layer)
    return layered


def _apply_regrade(record: dict[str, Any], artifact: dict[str, Any]) -> dict[str, Any]:
    corrected_reason = artifact.get("corrected_reason")
    if not isinstance(corrected_reason, str) or not corrected_reason:
        raise ValueError(
            f"regrade {artifact.get('instance_id')}: missing corrected_reason"
        )
    result = dict(record.get("test_result") or {})
    result["resolved"] = corrected_reason == "resolved"
    result["reason"] = None if corrected_reason == "resolved" else corrected_reason
    for key in (
        "candidate_patch_applied",
        "test_patch_applied",
        "eval_exit_code",
        "eval_stdout",
        "eval_stderr",
        "timed_out",
    ):
        if key in artifact:
            result[key] = artifact[key]
    return {
        **record,
        "test_result": result,
        "correction": {
            "kind": "evaluator_regrade",
            "original_reason": artifact.get("original_reason"),
            "corrected_reason": corrected_reason,
        },
    }


def corrected_records(
    main: dict[str, dict[str, Any]],
    regrades: dict[str, dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], int, int]:
    unknown_replacements = sorted(set(replacements) - set(main))
    if unknown_replacements:
        raise ValueError(
            "replacement corrections reference absent main results: "
            f"{unknown_replacements}"
        )

    corrected = {
        instance_id: _apply_regrade(record, regrades[instance_id])
        if instance_id in regrades
        else record
        for instance_id, record in main.items()
    }
    for instance_id in sorted(set(regrades) - set(main)):
        corrected[instance_id] = _apply_regrade(
            {"instance_id": instance_id, "test_result": {}},
            regrades[instance_id],
        )
    replacement_count = 0
    for instance_id, replacement in replacements.items():
        corrected[instance_id] = {
            **replacement,
            "correction": {
                "kind": "agent_environment_rerun",
                "replaced_reason": _result_reason(main[instance_id]),
            },
        }
        replacement_count += 1
    return corrected, len(regrades), replacement_count


def build_summary(
    dataset_rows: list[dict[str, Any]],
    main: dict[str, dict[str, Any]],
    corrected: dict[str, dict[str, Any]],
    regrade_count: int,
    replacement_count: int,
) -> dict[str, Any]:
    dataset_ids = [str(row["label"]) for row in dataset_rows]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("dataset contains duplicate labels")
    unknown_results = sorted(set(main) - set(dataset_ids))
    unknown_corrected = sorted(set(corrected) - set(dataset_ids))
    if unknown_results or unknown_corrected:
        raise ValueError(
            "results contain unknown IDs: "
            f"main={unknown_results}, corrected={unknown_corrected}"
        )

    raw_reasons = Counter(_result_reason(record) for record in main.values())
    corrected_reasons = Counter(_result_reason(record) for record in corrected.values())
    missing = set(dataset_ids) - set(corrected)
    by_repo: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "evaluated": 0, "resolved": 0}
    )
    for row in dataset_rows:
        instance_id = str(row["label"])
        repo = str(row["metadata"]["repo"])
        by_repo[repo]["total"] += 1
        if instance_id in corrected:
            by_repo[repo]["evaluated"] += 1
            if _result_reason(corrected[instance_id]) == "resolved":
                by_repo[repo]["resolved"] += 1

    resolved = corrected_reasons["resolved"]
    evaluated = len(corrected)
    total = len(dataset_ids)
    unhandled = {
        reason: count
        for reason, count in corrected_reasons.items()
        if reason not in {"resolved", "tests_failed"}
    }
    unhandled_instance_ids: dict[str, list[str]] = defaultdict(list)
    for instance_id, record in corrected.items():
        reason = _result_reason(record)
        if reason not in {"resolved", "tests_failed"}:
            unhandled_instance_ids[reason].append(instance_id)
    return {
        "dataset_rows": total,
        "evaluated_rows": evaluated,
        "missing_rows": len(missing),
        "complete": not missing,
        "regrades_applied": regrade_count,
        "regrade_only_rows": len(set(corrected) - set(main)),
        "replacement_reruns_applied": replacement_count,
        "raw_by_reason": dict(sorted(raw_reasons.items())),
        "corrected_by_reason": dict(sorted(corrected_reasons.items())),
        "unhandled_by_reason": dict(sorted(unhandled.items())),
        "unhandled_instance_ids": {
            reason: sorted(instance_ids)
            for reason, instance_ids in sorted(unhandled_instance_ids.items())
        },
        "resolved_rate_evaluated": resolved / evaluated if evaluated else None,
        "resolved_rate_dataset": resolved / total,
        "by_repo": {
            repo: {
                **counts,
                "resolved_rate_evaluated": (
                    counts["resolved"] / counts["evaluated"]
                    if counts["evaluated"]
                    else None
                ),
                "resolved_rate_dataset": counts["resolved"] / counts["total"],
            }
            for repo, counts in sorted(by_repo.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument(
        "--regrade-dir",
        type=Path,
        action="append",
        default=[],
        help="Regrade directory; later directories override earlier ones by instance ID.",
    )
    parser.add_argument("--replacement-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_rows = _read_jsonl([args.dataset])
    main_records = _index_unique(
        _read_jsonl(args.result), "instance_id", "main results"
    )
    replacement_records = _index_unique(
        _read_jsonl(args.replacement_result), "instance_id", "replacement results"
    )
    regrades = _load_regrades(args.regrade_dir)
    corrected, regrade_count, replacement_count = corrected_records(
        main_records, regrades, replacement_records
    )
    summary = build_summary(
        dataset_rows,
        main_records,
        corrected,
        regrade_count,
        replacement_count,
    )
    if args.require_complete and not summary["complete"]:
        raise RuntimeError(
            f"evaluation is incomplete: {summary['missing_rows']} missing"
        )
    if args.require_clean and summary["unhandled_by_reason"]:
        raise RuntimeError(
            f"evaluation has unhandled outcomes: {summary['unhandled_by_reason']}"
        )

    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
