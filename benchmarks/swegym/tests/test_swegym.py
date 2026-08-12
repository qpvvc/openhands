import asyncio
import json
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from benchmarks.swegym.opensandbox_workspace import (
    OpenSandboxWorkspace,
    _command_prefix_for_image,
    _resource_for_image,
)
from benchmarks.swegym.run_infer import (
    ConversationWallClockTimeoutError,
    SWEGymEvaluation,
    _candidate_apply_command,
    _candidate_deleted_missing_path,
    _candidate_diff_command,
    _evaluator_pre_commands_for,
    _extract_plaintext_eval_script,
    _normalize_unified_diff_hunk_counts,
    _pre_commands_for,
    _sanitize_pre_commands,
    _scale_swe_evaluator_files,
    _scale_swe_grader_completed,
    _shell_write_text_commands,
    _terminate_patch,
    _to_builtin_types,
    _validate_f2p_preflight_result,
    _warn_max_input_tokens,
    _workspace_baseline_tree_command,
    conversation_wall_clock_timeout,
    normalize_row,
)
from benchmarks.swegym.score import (
    classify_row,
    failure_counts,
    score_file,
    score_files,
)


def test_normalize_prepared_row() -> None:
    row = {
        "prompt": "fix it",
        "label": "org__repo-1",
        "metadata": {
            "image": "registry/image:latest",
            "workdir": "/testbed",
            "base_commit": "abc",
            "evaluator_pre_commands": "cat > .swe_eval/eval.sh <<'EOF'\nexit 1\nEOF\n",
            "eval_cmd": "./eval.sh",
            "test_patch": "diff --git a/test.py b/test.py\n",
        },
    }
    data = normalize_row(row)
    assert data["instance_id"] == "org__repo-1"
    assert data["problem_statement"] == "fix it"
    assert data["dataset_kind"] == "swe_gym"


def test_to_builtin_types_makes_numpy_values_json_serializable() -> None:
    data = _to_builtin_types({"tests": np.array(["a", "b"]), "count": np.int64(2)})

    assert data == {"tests": ["a", "b"], "count": 2}
    assert json.loads(json.dumps(data)) == data


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
            "evaluator_pre_commands": "cat > .swe_eval/eval.sh <<'EOF'\nexit 1\nEOF\n",
            "eval_cmd": "./eval.sh",
            "test_patch": "diff --git a/test.py b/test.py\n",
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


def test_candidate_patch_is_terminated_for_git_apply() -> None:
    assert _terminate_patch("diff body") == "diff body\n"
    assert _terminate_patch("diff body\n") == "diff body\n"
    assert _terminate_patch("") == ""


def test_candidate_patch_uses_plain_apply_before_three_way_fallback() -> None:
    command = _candidate_apply_command("/tmp/candidate patch")

    assert command.startswith("git apply --whitespace=nowarn")
    assert " || git apply --3way --whitespace=nowarn " in command


def test_candidate_deleted_missing_path_attributes_model_caused_setup_failure() -> None:
    patch = """diff --git a/test-requirements.txt b/test-requirements.txt
deleted file mode 100644
index 1234567..0000000
--- a/test-requirements.txt
+++ /dev/null
@@ -1 +0,0 @@
-pytest
"""

    assert (
        _candidate_deleted_missing_path(
            patch,
            "Could not open requirements file: [Errno 2] No such file or directory: "
            "'/testbed/test-requirements.txt'",
            "/testbed",
        )
        == "test-requirements.txt"
    )
    assert (
        _candidate_deleted_missing_path(
            patch,
            "No such file or directory: 'different-file.txt'",
            "/testbed",
        )
        is None
    )


def test_workspace_baseline_tree_excludes_image_dirt_from_agent_patch(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("initial\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)

    tracked.write_text("image preparation\n")
    baseline = subprocess.run(
        ["bash", "-lc", _workspace_baseline_tree_command()],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "agent.py").write_text("agent change\n")
    patch = subprocess.run(
        ["bash", "-lc", _candidate_diff_command(baseline)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "agent.py" in patch
    assert "tracked.py" not in patch


def test_candidate_patch_embeds_new_binary_files(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.py").write_text("initial\n")
    subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)

    (tmp_path / "agent.so").write_bytes(bytes(range(256)) * 4)
    patch = subprocess.run(
        ["bash", "-lc", _candidate_diff_command("HEAD")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    subprocess.run(["git", "reset", "--hard", "-q", "HEAD"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=tmp_path,
        check=True,
        input=patch,
        text=True,
    )

    assert "GIT binary patch" in patch
    assert (tmp_path / "agent.so").read_bytes() == bytes(range(256)) * 4


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


def test_opensandbox_recovers_missing_stream_exit_code_from_file() -> None:
    calls: list[str] = []

    async def run(command: str, **kwargs: object) -> object:
        del kwargs
        calls.append(command)
        if len(calls) == 1:
            return SimpleNamespace(
                id="already-removed",
                exit_code=None,
                logs=SimpleNamespace(stdout=[], stderr=[]),
            )
        return SimpleNamespace(
            id="recovery-command",
            exit_code=None,
            logs=SimpleNamespace(
                stdout=[SimpleNamespace(text="7\n")],
                stderr=[],
            ),
        )

    async def unexpected_status_call(execution_id: str) -> object:
        raise AssertionError(f"status should not be queried: {execution_id}")

    commands = SimpleNamespace(
        run=run,
        get_command_status=unexpected_status_call,
    )
    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.working_dir = "/testbed"
    workspace.__dict__["_sandbox"] = SimpleNamespace(commands=commands)
    workspace._loop = asyncio.new_event_loop()

    try:
        result = workspace.execute_command("pytest -q", timeout=20)
    finally:
        workspace._loop.close()

    assert result.exit_code == 7
    assert len(calls) == 2
    assert calls[1].startswith("cat /tmp/__OPENHANDS_EXIT_CODE_")


def test_opensandbox_unrecoverable_exit_code_is_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run(command: str, **kwargs: object) -> object:
        del command, kwargs
        return SimpleNamespace(
            id="already-removed",
            exit_code=None,
            logs=SimpleNamespace(stdout=[], stderr=[]),
        )

    async def missing_exit_file(*args: object, **kwargs: object) -> int:
        raise TimeoutError

    async def unexpected_status_call(execution_id: str) -> object:
        raise AssertionError(f"status should not be queried: {execution_id}")

    monkeypatch.setattr(
        OpenSandboxWorkspace,
        "_recover_exit_code_file",
        missing_exit_file,
    )
    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.working_dir = "/testbed"
    workspace.__dict__["_sandbox"] = SimpleNamespace(
        commands=SimpleNamespace(
            run=run,
            get_command_status=unexpected_status_call,
        )
    )
    workspace._loop = asyncio.new_event_loop()

    try:
        result = workspace.execute_command("pytest -q", timeout=20)
    finally:
        workspace._loop.close()

    assert result.exit_code == -1
    assert result.timeout_occurred
    assert "without a recoverable exit code" in result.stderr


def test_opensandbox_recovers_missing_stream_exit_code_from_shell_marker() -> None:
    seen: dict[str, object] = {}

    async def run(command: str, **kwargs: object) -> object:
        seen["command"] = command
        seen.update(kwargs)
        marker = command.split("printf '\\n", 1)[1].split("=%s", 1)[0]
        return SimpleNamespace(
            id="already-removed",
            exit_code=None,
            logs=SimpleNamespace(
                stdout=[SimpleNamespace(text="test output")],
                stderr=[SimpleNamespace(text=f"test error\n{marker}=7\n")],
            ),
        )

    async def unexpected_status_call(execution_id: str) -> object:
        raise AssertionError(f"status should not be queried: {execution_id}")

    commands = SimpleNamespace(
        run=run,
        get_command_status=unexpected_status_call,
    )
    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.working_dir = "/testbed"
    workspace.__dict__["_sandbox"] = SimpleNamespace(commands=commands)
    workspace._loop = asyncio.new_event_loop()

    try:
        result = workspace.execute_command("pytest -q", timeout=20)
    finally:
        workspace._loop.close()

    assert result.exit_code == 7
    assert result.stdout == "test output"
    assert result.stderr == "test error"
    assert seen["opts"].timeout == timedelta(seconds=20)  # type: ignore[union-attr]
    assert "sleep 0.1" in str(seen["command"])


def test_modin_sandbox_commands_limit_visible_cpus() -> None:
    base = (
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NINJAFLAGS=-j2 "
        "MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2"
    )
    assert (
        _command_prefix_for_image(
            "registry/xingyaoww/sweb.eval.x86_64.modin-project_s_modin-1:latest"
        )
        == f"{base} MODIN_CPUS=2 && "
    )
    assert _command_prefix_for_image("registry/other:latest") == f"{base} && "
    assert (
        _command_prefix_for_image(
            "registry/xingyaoww/sweb.eval.x86_64.dask_s_dask-6660:latest"
        )
        == f"{base} DASK_NUM_WORKERS=2 && "
    )


def test_sandbox_resources_scale_for_memory_heavy_images() -> None:
    assert _resource_for_image("registry/project-monai_s_monai-5254:latest") == {
        "cpu": "2",
        "memory": "8Gi",
    }
    assert _resource_for_image("registry/dask_s_dask-6660:latest", 2) == {
        "cpu": "4",
        "memory": "8Gi",
    }
    assert _resource_for_image("registry/pandas-dev_s_pandas-56341:latest") == {
        "cpu": "2",
        "memory": "4Gi",
    }
    assert _resource_for_image("registry/other:latest", 2) == {
        "cpu": "2",
        "memory": "4Gi",
    }


def test_opensandbox_command_has_local_hard_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float] = {}

    async def command_operation() -> None:
        await asyncio.sleep(60)

    async def force_timeout(operation: object, timeout: float) -> None:
        seen["timeout"] = timeout
        operation.close()  # type: ignore[attr-defined]
        raise TimeoutError

    async def missing_exit_file(*args: object, **kwargs: object) -> int:
        raise TimeoutError

    commands = SimpleNamespace(run=lambda *args, **kwargs: command_operation())
    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.working_dir = "/testbed"
    workspace.__dict__["_sandbox"] = SimpleNamespace(commands=commands)
    workspace._loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "wait_for", force_timeout)
    monkeypatch.setattr(
        OpenSandboxWorkspace,
        "_recover_exit_code_file",
        missing_exit_file,
    )

    try:
        result = workspace.execute_command("echo ok", timeout=20)
    finally:
        workspace._loop.close()

    assert seen["timeout"] == 30
    assert result.exit_code == -1
    assert result.timeout_occurred
    assert "local hard timeout" in result.stderr


def test_opensandbox_hard_timeout_recovers_persisted_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, float] = {}

    async def command_operation() -> None:
        await asyncio.sleep(60)

    async def force_timeout(operation: object, timeout: float) -> None:
        seen["hard_timeout"] = timeout
        operation.close()  # type: ignore[attr-defined]
        raise TimeoutError

    async def recover_exit_file(
        cls: type[OpenSandboxWorkspace],
        commands: object,
        path: str,
        timeout: float,
    ) -> int:
        del cls, commands, path
        seen["recovery_timeout"] = timeout
        return 0

    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.working_dir = "/testbed"
    workspace.__dict__["_sandbox"] = SimpleNamespace(
        commands=SimpleNamespace(run=lambda *args, **kwargs: command_operation())
    )
    workspace._loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "wait_for", force_timeout)
    monkeypatch.setattr(
        OpenSandboxWorkspace,
        "_recover_exit_code_file",
        classmethod(recover_exit_file),
    )

    try:
        result = workspace.execute_command("echo ok", timeout=20)
    finally:
        workspace._loop.close()

    assert seen == {"hard_timeout": 30, "recovery_timeout": 10}
    assert result.exit_code == 0
    assert not result.timeout_occurred


def test_opensandbox_write_file_uses_filesystem_api() -> None:
    seen: dict[str, object] = {}

    async def write_file(path: str, content: str, *, mode: int) -> None:
        seen.update(path=path, content=content, mode=mode)

    workspace = OpenSandboxWorkspace.__new__(OpenSandboxWorkspace)
    workspace.__dict__["_sandbox"] = SimpleNamespace(
        files=SimpleNamespace(write_file=write_file)
    )
    workspace._loop = asyncio.new_event_loop()

    try:
        workspace.write_file("/testbed/.swe_eval/eval.sh", "echo ok\n", mode=755)
    finally:
        workspace._loop.close()

    assert seen == {
        "path": "/testbed/.swe_eval/eval.sh",
        "content": "echo ok\n",
        "mode": 755,
    }


def test_swe_gym_evaluator_reuses_and_resets_agent_workspace() -> None:
    commands: list[tuple[str, float]] = []

    def execute_command(command: str, timeout: float) -> object:
        commands.append((command, timeout))
        return SimpleNamespace(
            exit_code=1,
            stdout="reset stdout",
            stderr="reset stderr",
            timeout_occurred=False,
        )

    workspace = cast(Any, SimpleNamespace(execute_command=execute_command))
    evaluator = cast(
        Any,
        SimpleNamespace(
            setup_timeout=1200,
            eval_timeout=900,
            sandbox_lifetime=3600,
        ),
    )
    instance = cast(
        Any,
        SimpleNamespace(
            data={
                "workdir": "/testbed",
                "image": "registry/image:latest",
                "base_commit": "abc123",
            }
        ),
    )

    result = SWEGymEvaluation._evaluate_swe_gym_patch(
        evaluator, instance, "diff --git a/a b/a\n", workspace
    )

    assert result["reason"] == "evaluator_setup_failed"
    assert commands == [("git reset --hard abc123 && git clean -fd", 60)]


def test_pre_commands_sanitization_is_scale_swe_only() -> None:
    commands = "git gc --prune=now --aggressive\\n"

    assert _pre_commands_for({"pre_commands": commands}) == commands
    assert (
        _pre_commands_for({"dataset_kind": "scale_swe", "pre_commands": commands})
        == "git gc --prune=now\n"
    )


def test_pandas_evaluator_pre_commands_limit_meson_compile_jobs() -> None:
    commands = "python -m pip install -ve . -Ceditable-verbose=true"

    assert _evaluator_pre_commands_for(
        {"repo": "pandas-dev/pandas", "evaluator_pre_commands": commands}
    ) == (commands + " -Ccompile-args=-j2")
    assert (
        _evaluator_pre_commands_for(
            {"repo": "other/repo", "evaluator_pre_commands": commands}
        )
        == commands
    )
    assert _evaluator_pre_commands_for({}) == "true"


def test_extract_plaintext_eval_script_from_prepared_commands() -> None:
    script = "#!/bin/bash\necho ok\n"
    commands = (
        "mkdir -p .swe_eval\n"
        "cat > .swe_eval/eval.sh <<'OPENHANDS_SWE_GYM_EVAL_SCRIPT'\n"
        f"{script.rstrip()}\n"
        "OPENHANDS_SWE_GYM_EVAL_SCRIPT\n"
        "chmod 755 .swe_eval/eval.sh\n"
    )

    assert _extract_plaintext_eval_script(commands) == script
    assert _extract_plaintext_eval_script("echo custom setup") is None


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


def test_score_classifies_evaluator_and_infrastructure_failures(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.jsonl"
    rows = [
        {"instance_id": "a", "test_result": {"resolved": True}, "error": None},
        {
            "instance_id": "b",
            "test_result": {"resolved": False, "reason": "eval_timeout"},
            "error": None,
        },
        {"instance_id": "c", "test_result": None, "error": "sandbox failed"},
    ]
    output.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    assert classify_row(rows[1]) == "eval_timeout"
    assert failure_counts([output]) == {  # type: ignore[comparison-overlap]
        "resolved": 1,
        "eval_timeout": 1,
        "infrastructure_error": 1,
    }


def test_conversation_wall_clock_timeout() -> None:
    with pytest.raises(ConversationWallClockTimeoutError):
        with conversation_wall_clock_timeout(1):
            time.sleep(2)
