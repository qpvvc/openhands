from pathlib import Path

import pytest

from benchmarks.swegym.regrade import (
    _candidate_patch,
    _PatchEvaluator,
    _read_candidate_patch_dirs,
    _with_eval_thread_limit,
    _with_parallel_test_batches,
)


def test_patch_evaluator_keeps_resource_factor_for_fresh_sandbox() -> None:
    evaluator = _PatchEvaluator(1200, 900, 3600, resource_factor=4)

    assert evaluator.resource_factor == 4


def test_eval_thread_limit_overrides_single_thread_sandbox_prefix() -> None:
    data = _with_eval_thread_limit({"eval_cmd": "./.swe_eval/eval.sh"}, 8)

    assert data["eval_cmd"].startswith(
        "export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
        "NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8; "
    )
    assert data["eval_cmd"].endswith("./.swe_eval/eval.sh")


def test_parallel_test_batches_partition_exact_tests_once() -> None:
    marker = "python .swe_eval/run_tests.py\nOPENHANDS_SWE_GYM_EVAL_SCRIPT"
    data = _with_parallel_test_batches(
        {
            "fail_to_pass": ["f2p"],
            "pass_to_pass": [f"p2p-{index}" for index in range(9)],
            "evaluator_pre_commands": marker,
        },
        4,
    )

    commands = data["evaluator_pre_commands"]
    assert "run_tests.py 0 3 &" in commands
    assert "run_tests.py 3 6 &" in commands
    assert "run_tests.py 6 9 &" in commands
    assert "run_tests.py 9 10 &" in commands
    assert commands.count('pids+=("$!")') == 4
    assert "python .swe_eval/run_tests.py\n" not in commands


def test_candidate_patch_uses_saved_grader_patch() -> None:
    original = {"test_result": {"git_patch": "diff --git a/a b/a\n"}}

    assert _candidate_patch(original, "saved") == "diff --git a/a b/a\n"


def test_candidate_patch_can_run_empty_baseline_control() -> None:
    original = {"test_result": {"git_patch": "model patch"}}

    assert _candidate_patch(original, "empty") == ""


def test_read_persisted_candidate_patches(tmp_path: Path) -> None:
    (tmp_path / "repo__issue-1.candidate.patch").write_text("model patch")

    assert _read_candidate_patch_dirs([tmp_path]) == {"repo__issue-1": "model patch"}


def test_persisted_candidate_patch_ids_must_be_unique(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "repo__issue-1.candidate.patch").write_text("first")
    (second / "repo__issue-1.candidate.patch").write_text("second")

    with pytest.raises(ValueError, match="duplicate persisted candidate patch"):
        _read_candidate_patch_dirs([first, second])
