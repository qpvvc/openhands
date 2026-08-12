from __future__ import annotations

import json

import pytest

from benchmarks.swegym.corrected_score import (
    _index_unique,
    _load_regrades,
    _result_reason,
    build_summary,
    corrected_records,
)


def _result(instance_id: str, resolved: bool, reason: str | None) -> dict:
    return {
        "instance_id": instance_id,
        "test_result": {"resolved": resolved, "reason": reason},
    }


def test_replacement_rerun_takes_precedence_over_regrade() -> None:
    main = {
        "a": _result("a", False, "eval_timeout"),
        "b": _result("b", True, None),
    }
    regrades = {
        "a": {
            "instance_id": "a",
            "original_reason": "eval_timeout",
            "corrected_reason": "resolved",
        }
    }
    replacements = {"a": _result("a", False, "tests_failed")}

    corrected, regrade_count, replacement_count = corrected_records(
        main, regrades, replacements
    )
    summary = build_summary(
        [
            {"label": "a", "metadata": {"repo": "owner/repo"}},
            {"label": "b", "metadata": {"repo": "owner/repo"}},
        ],
        main,
        corrected,
        regrade_count,
        replacement_count,
    )

    assert corrected["a"]["test_result"]["reason"] == "tests_failed"
    assert corrected["a"]["correction"]["kind"] == "agent_environment_rerun"
    assert summary["regrades_applied"] == 1
    assert summary["replacement_reruns_applied"] == 1
    assert summary["corrected_by_reason"] == {"resolved": 1, "tests_failed": 1}
    assert summary["unhandled_by_reason"] == {}
    assert summary["unhandled_instance_ids"] == {}


def test_index_unique_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate instance_id a"):
        _index_unique(
            [{"instance_id": "a"}, {"instance_id": "a"}],
            "instance_id",
            "results",
        )


def test_later_regrade_directory_overrides_an_earlier_result(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.json").write_text(
        json.dumps({"instance_id": "a", "corrected_reason": "eval_timeout"})
    )
    (second / "a.json").write_text(
        json.dumps({"instance_id": "a", "corrected_reason": "resolved"})
    )

    assert _load_regrades([first, second])["a"]["corrected_reason"] == "resolved"


def test_abnormal_eval_exit_requires_failure_evidence() -> None:
    killed = {
        "test_result": {
            "resolved": False,
            "reason": "tests_failed",
            "eval_exit_code": 137,
            "eval_stdout": "OPENHANDS_SWE_GYM_TEST_PATCH_APPLIED=1\n........",
        }
    }
    failed_before_kill = {
        "test_result": {
            "resolved": False,
            "reason": "tests_failed",
            "eval_exit_code": 137,
            "eval_stdout": "OPENHANDS_SWE_GYM_TEST_PATCH_APPLIED=1\n..F.....",
        }
    }

    assert _result_reason(killed) == "eval_process_killed"
    assert _result_reason(failed_before_kill) == "tests_failed"


def test_evaluator_regrade_can_confirm_candidate_induced_process_kill() -> None:
    record = {
        "test_result": {
            "resolved": False,
            "reason": "tests_failed",
            "eval_exit_code": 75,
            "eval_stdout": "OPENHANDS_SWE_GYM_TEST_PATCH_APPLIED=1\n",
        },
        "correction": {
            "kind": "evaluator_regrade",
            "corrected_reason": "tests_failed",
        },
    }

    assert _result_reason(record) == "tests_failed"


def test_regrade_can_supply_result_when_candidate_was_persisted_before_eval() -> None:
    main = {"a": _result("a", False, "tests_failed")}
    regrades = {
        "b": {
            "instance_id": "b",
            "original_reason": "missing_result_after_candidate_persisted",
            "corrected_reason": "resolved",
            "candidate_patch_applied": True,
            "test_patch_applied": True,
            "eval_exit_code": 0,
        }
    }

    corrected, regrade_count, replacement_count = corrected_records(main, regrades, {})
    summary = build_summary(
        [
            {"label": "a", "metadata": {"repo": "owner/repo"}},
            {"label": "b", "metadata": {"repo": "owner/repo"}},
        ],
        main,
        corrected,
        regrade_count,
        replacement_count,
    )

    assert corrected["b"]["test_result"]["resolved"] is True
    assert summary["complete"] is True
    assert summary["regrade_only_rows"] == 1
    assert summary["corrected_by_reason"] == {"resolved": 1, "tests_failed": 1}


def test_regrade_only_result_must_belong_to_dataset() -> None:
    corrected, regrade_count, replacement_count = corrected_records(
        {},
        {"unknown": {"instance_id": "unknown", "corrected_reason": "resolved"}},
        {},
    )

    with pytest.raises(ValueError, match="corrected=\\['unknown'\\]"):
        build_summary(
            [{"label": "a", "metadata": {"repo": "owner/repo"}}],
            {},
            corrected,
            regrade_count,
            replacement_count,
        )
