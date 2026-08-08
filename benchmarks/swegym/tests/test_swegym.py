import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.swegym.opensandbox_workspace import OpenSandboxWorkspace
from benchmarks.swegym.run_infer import (
    ConversationWallClockTimeoutError,
    _normalize_unified_diff_hunk_counts,
    _pre_commands_for,
    _sanitize_pre_commands,
    _scale_swe_evaluator_files,
    _scale_swe_grader_completed,
    _shell_write_text_commands,
    _validate_f2p_preflight_result,
    _warn_max_input_tokens,
    conversation_wall_clock_timeout,
    normalize_row,
)
from benchmarks.swegym.score import score_file, score_files


def test_normalize_prepared_row() -> None:
    row = {
        "prompt": "fix it",
        "label": "org__repo-1",
        "metadata": {
            "image": "registry/image:latest",
            "workdir": "/testbed",
            "base_commit": "abc",
            "pre_commands": "true",
            "eval_cmd": "./eval.sh",
        },
    }
    data = normalize_row(row)
    assert data["instance_id"] == "org__repo-1"
    assert data["problem_statement"] == "fix it"


def test_normalize_prepared_row_applies_sandbox_image_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_IMAGE_PREFIX", "registry.test")
    row = {
        "prompt": "fix it",
        "label": "org__repo-1",
        "metadata": {
            "image": "xingyaoww/sweb.eval.x86_64.org_s_repo-1:latest",
            "workdir": "/testbed",
            "base_commit": "abc",
            "pre_commands": "true",
            "eval_cmd": "./eval.sh",
        },
    }

    data = normalize_row(row)

    assert data["image"] == (
        "registry.test/xingyaoww/sweb.eval.x86_64.org_s_repo-1:latest"
    )


def test_normalize_rejects_raw_row() -> None:
    with pytest.raises(ValueError, match="prepared .*_osb.jsonl"):
        normalize_row({"instance_id": "org__repo-1", "problem_statement": "x"})


def test_normalize_scale_swe_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCALE_SWE_IMAGE_PREFIX", "registry.test")
    data = normalize_row(
        {
            "prompt": "fix it",
            "label": "owner_repo_pr1",
            "metadata": {
                "_slime_dataset": "scale_swe",
                "image_url": "aweaiteam/scaleswe:owner_repo_pr1",
                "workdir": "/workspace/repo",
                "f2p_patch": "diff --git a/test.py b/test.py\n",
                "FAIL_TO_PASS": '["test.py::test_fix"]',
                "PASS_TO_PASS": [],
            },
        }
    )
    assert data["dataset_kind"] == "scale_swe"
    assert data["image"] == "registry.test/aweaiteam/scaleswe:owner_repo_pr1"


def test_normalize_scale_swe_does_not_treat_gold_patch_as_tests() -> None:
    with pytest.raises(ValueError, match="f2p_patch/f2p_script"):
        normalize_row(
            {
                "_slime_dataset": "scale_swe",
                "instance_id": "owner_repo_pr1",
                "image_url": "registry/image",
                "workdir": "/workspace/repo",
                "problem_statement": "fix it",
                "patch": "gold answer must not become evaluator tests",
                "FAIL_TO_PASS": ["test.py::test_fix"],
            }
        )


def test_normalize_scale_swe_accepts_script_only_evaluator() -> None:
    data = normalize_row(
        {
            "_slime_dataset": "scale_swe",
            "instance_id": "owner_repo_pr1",
            "image_url": "registry/image",
            "workdir": "/workspace/repo",
            "problem_statement": "fix it",
            "f2p_patch": "",
            "f2p_script": "def test_fix(): assert True\n",
            "FAIL_TO_PASS": ["test_fail_to_pass.py::test_fix"],
        }
    )

    assert data["f2p_script"].startswith("def test_fix")


def test_scale_swe_evaluator_materializes_f2p_script() -> None:
    data = {
        "workdir": "/workspace/repo",
        "f2p_patch": "test patch",
        "f2p_script": "def test_fix(): assert True\n",
        "FAIL_TO_PASS": ["test_fail_to_pass.py::test_fix"],
    }

    files = dict(_scale_swe_evaluator_files(data, "candidate patch"))

    assert files["/workspace/repo/test_fail_to_pass.py"] == data["f2p_script"]
    assert files["/tmp/openhands_f2p.patch"] == "test patch\n"
    assert files["/tmp/openhands_candidate.patch"] == "candidate patch\n"


def test_normalize_unified_diff_hunk_counts() -> None:
    malformed = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -10,7 +10,9 @@
 context
-old
+new one
+new two
 context
"""

    normalized = _normalize_unified_diff_hunk_counts(malformed)

    assert "@@ -10,3 +10,4 @@" in normalized


def test_shell_write_text_commands_do_not_exceed_argv_limit(tmp_path: Path) -> None:
    destination = tmp_path / "large patch.txt"
    content = "large patch payload\n" * 20_000

    commands = _shell_write_text_commands(str(destination), content)

    assert max(map(len, commands)) < 64 * 1024
    for command in commands:
        subprocess.run(["bash", "-lc", command], check=True)
    assert destination.read_text() == content


def test_max_input_tokens_overage_is_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = SimpleNamespace(
        token_usages=[
            SimpleNamespace(prompt_tokens=32_768),
            SimpleNamespace(prompt_tokens=32_769),
        ]
    )

    warnings: list[tuple[str, tuple[int, int]]] = []
    monkeypatch.setattr(
        "benchmarks.swegym.run_infer.logger.warning",
        lambda message, *args: warnings.append((message, args)),
    )

    _warn_max_input_tokens(metrics, 32_768)

    assert warnings == [
        (
            "LLM request exceeded configured max_input_tokens: %d > %d",
            (32_769, 32_768),
        )
    ]


def test_invalid_f2p_patch_preflight_is_an_error() -> None:
    result = SimpleNamespace(
        exit_code=128,
        stdout="",
        stderr="error: corrupt patch at line 15",
    )

    with pytest.raises(RuntimeError, match="corrupt patch"):
        _validate_f2p_preflight_result(result)


def test_scale_swe_grader_requires_completion_evidence() -> None:
    result = SimpleNamespace(exit_code=0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="without pytest completion evidence"):
        _scale_swe_grader_completed(result)


def test_scale_swe_grader_accepts_matching_completion_evidence() -> None:
    result = SimpleNamespace(
        exit_code=0,
        stdout="1 passed\nOPENHANDS_SCALE_SWE_PYTEST_EXIT_CODE=0\n",
        stderr="",
    )

    assert _scale_swe_grader_completed(result)


def test_opensandbox_missing_exit_code_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="without an exit code"):
        OpenSandboxWorkspace._exit_code(SimpleNamespace())


def test_pre_commands_sanitization_is_scale_swe_only() -> None:
    commands = "git gc --prune=now --aggressive\\n"

    assert _pre_commands_for({"pre_commands": commands}) == commands
    assert (
        _pre_commands_for({"dataset_kind": "scale_swe", "pre_commands": commands})
        == "git gc --prune=now\n"
    )


def test_sanitize_pre_commands_keeps_trailing_newline_after_git_gc() -> None:
    commands = "git checkout -b scaleswe && git gc --prune=now --aggressive\\n"

    assert _sanitize_pre_commands(commands) == (
        "git checkout -b scaleswe && git gc --prune=now\n"
    )


def test_score_file(tmp_path: Path) -> None:
    output = tmp_path / "output.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps({"instance_id": "a", "test_result": {"resolved": True}}),
                json.dumps({"instance_id": "b", "test_result": {"resolved": False}}),
            ]
        )
    )
    assert score_file(output) == (1, 2, ["a"])


def test_score_files(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"instance_id": "a", "test_result": {"resolved": True}})
    )
    second.write_text(
        json.dumps({"instance_id": "b", "test_result": {"resolved": False}})
    )
    assert score_files([first, second]) == (1, 2, ["a"])


def test_conversation_wall_clock_timeout() -> None:
    with pytest.raises(ConversationWallClockTimeoutError):
        with conversation_wall_clock_timeout(1):
            time.sleep(2)
