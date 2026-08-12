from __future__ import annotations

import base64
import json
import os
import re
import shlex
import signal
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, List

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from omegaconf import OmegaConf
from pydantic import Field

from benchmarks.swegym.opensandbox_workspace import (
    OpenSandboxTerminalTool,
    OpenSandboxWorkspace,
)
from benchmarks.utils.args_parser import get_parser
from benchmarks.utils.conversation import build_event_persistence_callback
from benchmarks.utils.critics import create_critic
from benchmarks.utils.dataset import get_dataset, prepare_dataset
from benchmarks.utils.evaluation import Evaluation
from benchmarks.utils.evaluation_utils import (
    construct_eval_output_dir,
    get_default_on_result_writer,
)
from benchmarks.utils.fake_user_response import run_conversation_with_fake_user_response
from benchmarks.utils.models import EvalInstance, EvalMetadata, EvalOutput
from openhands.sdk import LLM, Agent, Conversation, Tool, get_logger
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.llm_convertible.system import SystemPromptEvent
from openhands.sdk.tool import ToolDefinition, register_tool
from openhands.sdk.workspace import LocalWorkspace, RemoteWorkspace


logger = get_logger(__name__)

DEFAULT_DATASET = (
    Path(__file__).resolve().parents[4]
    / "data/SWE-Gym/processed/swe_gym_8unique_easy_osb.jsonl"
)
_SCALE_SWE_RUNNER = r"""import json
import sys

import pytest

with open(sys.argv[1], encoding="utf-8") as stream:
    test_ids = json.load(stream)["test_ids"]
sys.path.insert(0, ".")
exit_code = int(pytest.main(["-q", "--tb=short", "--no-header", "-o", "addopts=", "--rootdir=.", *test_ids]))
print(f"OPENHANDS_SCALE_SWE_PYTEST_EXIT_CODE={exit_code}", flush=True)
raise SystemExit(exit_code)
"""
_SCALE_SWE_PYTEST_SENTINEL = "OPENHANDS_SCALE_SWE_PYTEST_EXIT_CODE="
_SWE_GYM_TEST_PATCH_MARKER = ".swe_eval/test_patch_applied"
_SWE_GYM_ISOLATED_TEST_KILLED = "OPENHANDS_SWE_GYM_ISOLATED_TEST_KILLED="
_SWE_GYM_EVAL_HEREDOC = "OPENHANDS_SWE_GYM_EVAL_SCRIPT"
_SHELL_WRITE_CHUNK_BYTES = 24 * 1024
_DIFF_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*?)(\r?\n)?$")
_GIT_TREE_HASH = re.compile(r"[0-9a-f]{40,64}")
_MISSING_FILE = re.compile(r"No such file or directory: ['\"]([^'\"]+)['\"]")


def _shell_write_text_commands(path: str, content: str) -> list[str]:
    """Build bounded shell commands that materialize UTF-8 text."""
    quoted_path = shlex.quote(path)
    commands = [f": > {quoted_path}"]
    payload = content.encode()
    for offset in range(0, len(payload), _SHELL_WRITE_CHUNK_BYTES):
        encoded = base64.b64encode(
            payload[offset : offset + _SHELL_WRITE_CHUNK_BYTES]
        ).decode()
        commands.append(
            f"printf %s {shlex.quote(encoded)} | base64 -d >> {quoted_path}"
        )
    return commands


def _materialize_workspace_files(
    workspace: OpenSandboxWorkspace, files: list[tuple[str, str]]
) -> None:
    for path, content in files:
        try:
            workspace.write_file(path, content)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to materialize evaluator file {path}: {exc}"
            ) from exc


def _extract_plaintext_eval_script(commands: str) -> str | None:
    prefix = (
        f"mkdir -p .swe_eval\ncat > .swe_eval/eval.sh <<'{_SWE_GYM_EVAL_HEREDOC}'\n"
    )
    suffix = f"\n{_SWE_GYM_EVAL_HEREDOC}\nchmod 755 .swe_eval/eval.sh\n"
    if not commands.startswith(prefix) or not commands.endswith(suffix):
        return None
    return commands[len(prefix) : -len(suffix)] + "\n"


def _normalize_unified_diff_hunk_counts(patch: str) -> str:
    lines = patch.splitlines(keepends=True)
    normalized: list[str] = []
    index = 0
    while index < len(lines):
        match = _DIFF_HUNK_HEADER.match(lines[index])
        if match is None:
            normalized.append(lines[index])
            index += 1
            continue

        body_end = index + 1
        while body_end < len(lines) and not (
            lines[body_end].startswith("@@ ")
            or lines[body_end].startswith("diff --git ")
        ):
            body_end += 1
        body = lines[index + 1 : body_end]
        old_count = sum(line.startswith((" ", "-")) for line in body)
        new_count = sum(line.startswith((" ", "+")) for line in body)
        newline = match.group(4) or ""
        normalized.append(
            f"@@ -{match.group(1)},{old_count} +{match.group(2)},{new_count} "
            f"@@{match.group(3)}{newline}"
        )
        normalized.extend(body)
        index = body_end
    return "".join(normalized)


def _terminate_patch(patch: str) -> str:
    return patch + "\n" if patch and not patch.endswith("\n") else patch


def _candidate_apply_command(path: str) -> str:
    quoted_path = shlex.quote(path)
    return (
        f"git apply --whitespace=nowarn {quoted_path} || "
        f"git apply --3way --whitespace=nowarn {quoted_path}"
    )


def _candidate_deleted_missing_path(
    patch: str, stderr: str, workdir: str
) -> str | None:
    deleted_paths: set[str] = set()
    current_path: str | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git a/") and " b/" in line:
            current_path = line.split(" b/", 1)[1]
        elif line.startswith("deleted file mode ") and current_path:
            deleted_paths.add(current_path)

    for match in _MISSING_FILE.finditer(stderr):
        missing = match.group(1)
        if missing.startswith(f"{workdir.rstrip('/')}/"):
            missing = missing[len(workdir.rstrip("/")) + 1 :]
        if missing.startswith("./"):
            missing = missing[2:]
        if missing in deleted_paths:
            return missing
    return None


def _workspace_baseline_tree_command() -> str:
    """Write the sandbox's initial worktree to a temporary Git index."""
    return (
        'baseline_index="$(mktemp)" && '
        "trap 'rm -f \"$baseline_index\"' EXIT && "
        'GIT_INDEX_FILE="$baseline_index" git read-tree HEAD && '
        'GIT_INDEX_FILE="$baseline_index" git add -A -- . '
        "':(exclude).swe_eval/**' ':(exclude)PROBLEM_STATEMENT.md' && "
        'GIT_INDEX_FILE="$baseline_index" git write-tree'
    )


def _candidate_diff_command(baseline_tree: str) -> str:
    return (
        "git config core.fileMode false && "
        "git add -A -- . ':(exclude).swe_eval/**' ':(exclude)PROBLEM_STATEMENT.md' && "
        f"git --no-pager diff --cached --binary --no-color {shlex.quote(baseline_tree)} -- . "
        "':(exclude).swe_eval/**' ':(exclude)PROBLEM_STATEMENT.md'"
    )


def _warn_max_input_tokens(metrics: Any, max_input_tokens: int | None) -> None:
    if max_input_tokens is None:
        return
    for usage in metrics.token_usages:
        if usage.prompt_tokens > max_input_tokens:
            logger.warning(
                "LLM request exceeded configured max_input_tokens: %d > %d",
                usage.prompt_tokens,
                max_input_tokens,
            )


def _validate_f2p_preflight_result(result: Any) -> None:
    if result.exit_code != 0:
        raise RuntimeError(
            f"Scale-SWE F2P patch failed preflight: {result.stderr or result.stdout}"
        )


def _scale_swe_grader_completed(result: Any) -> bool:
    sentinel_match = re.search(rf"{_SCALE_SWE_PYTEST_SENTINEL}(\d+)", result.stdout)
    if result.exit_code == 0 and sentinel_match is None:
        raise RuntimeError(
            "Scale-SWE grader returned exit code 0 without pytest completion evidence"
        )
    if sentinel_match is not None and int(sentinel_match.group(1)) != result.exit_code:
        raise RuntimeError(
            "Scale-SWE grader exit code disagrees with pytest completion evidence"
        )
    return sentinel_match is not None


def _scale_swe_test_ids(data: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = data.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = [value]
        if isinstance(value, list):
            result.extend(str(item) for item in value if str(item).strip())
    return result


class ConversationWallClockTimeoutError(BaseException):
    pass


@contextmanager
def conversation_wall_clock_timeout(seconds: int | None) -> Iterator[None]:
    if seconds is None:
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(signum: int, frame: Any) -> None:
        del signum, frame
        raise ConversationWallClockTimeoutError

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Flatten the prompt/label/metadata format used by prepared SWE-Gym data."""
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        data = _to_builtin_types(dict(metadata))
        data.setdefault("instance_id", row.get("label"))
        data.setdefault("problem_statement", row.get("prompt"))
    else:
        data = _to_builtin_types(dict(row))

    if data.get("_slime_dataset") == "scale_swe":
        image = str(data.get("image") or data.get("image_url") or "")
        prefix = (
            os.getenv("SCALE_SWE_IMAGE_PREFIX") or os.getenv("SANDBOX_IMAGE_PREFIX", "")
        ).rstrip("/")
        if prefix and image and not image.startswith(f"{prefix}/"):
            image = f"{prefix}/{image.lstrip('/')}"
        data["image"] = image
        data.setdefault("problem_statement", row.get("prompt"))
        data.setdefault("instance_id", row.get("label"))

        required = {"instance_id", "image", "workdir", "problem_statement"}
        missing = sorted(key for key in required if not data.get(key))
        if not (data.get("f2p_patch") or data.get("f2p_script")):
            missing.append("f2p_patch/f2p_script")
        if not _scale_swe_test_ids(data):
            missing.append("FAIL_TO_PASS/PASS_TO_PASS")
        if missing:
            raise ValueError(f"Scale-SWE row is missing executable metadata {missing}")
        data["dataset_kind"] = "scale_swe"
        return data

    image = str(data.get("image") or "")
    image_prefix = os.getenv("SANDBOX_IMAGE_PREFIX", "").rstrip("/")
    if image and image_prefix and not image.startswith(f"{image_prefix}/"):
        image = f"{image_prefix}/{image.lstrip('/')}"
    data["image"] = image

    required = {
        "instance_id",
        "image",
        "workdir",
        "problem_statement",
        "base_commit",
        "evaluator_pre_commands",
        "eval_cmd",
        "test_patch",
    }
    missing = sorted(key for key in required if not data.get(key))
    if missing:
        raise ValueError(
            "SWE-Gym row is missing executable metadata "
            f"{missing}. Use a prepared *_osb.jsonl dataset, not the raw Parquet."
        )
    data["dataset_kind"] = "swe_gym"
    return data


def _to_builtin_types(value: Any) -> Any:
    """Convert pandas/NumPy container values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: _to_builtin_types(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin_types(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin_types(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_instruction(
    instance: dict[str, Any], metadata: EvalMetadata, workspace_path: str
) -> str:
    assert metadata.prompt_path is not None
    prompts_dir = os.path.dirname(metadata.prompt_path)
    template_name = os.path.basename(metadata.prompt_path)
    template = Environment(loader=FileSystemLoader(prompts_dir)).get_template(
        template_name
    )
    context = {
        "instance": instance,
        "workspace_dir_name": instance["repo"].split("/")[-1],
        "actual_workspace_path": workspace_path,
        "metadata": metadata,
        "test_instructions": "",
    }
    return template.render(context)


def _sanitize_pre_commands(pre_commands: str | None) -> str:
    if not pre_commands:
        return "true"

    sanitized = pre_commands
    sanitized = sanitized.replace("\\n", "\n")
    sanitized = sanitized.replace("\\r", "\r")
    return re.sub(r"(git\s+gc\b[^;\n&|]*?)\s+--aggressive\b", r"\1", sanitized)


def _pre_commands_for(data: dict[str, Any]) -> str:
    commands = str(data.get("pre_commands") or "true")
    if data.get("dataset_kind") == "scale_swe":
        return _sanitize_pre_commands(commands)
    return commands


def _evaluator_pre_commands_for(data: dict[str, Any]) -> str:
    """Apply evaluator resource guards to prepared and legacy datasets."""
    commands = str(data.get("evaluator_pre_commands") or "true")
    if data.get("repo") == "pandas-dev/pandas":
        editable_setting = "-Ceditable-verbose=true"
        compile_limit = "-Ccompile-args=-j2"
        if editable_setting in commands and compile_limit not in commands:
            commands = commands.replace(
                editable_setting,
                f"{editable_setting} {compile_limit}",
            )
    return commands


def _scale_swe_evaluator_files(
    data: dict[str, Any], git_patch: str
) -> list[tuple[str, str]]:
    git_patch = _terminate_patch(git_patch)
    f2p_patch = _terminate_patch(str(data.get("f2p_patch") or ""))
    files = [
        ("/tmp/openhands_candidate.patch", git_patch),
        (
            "/tmp/openhands_f2p.patch",
            _normalize_unified_diff_hunk_counts(f2p_patch),
        ),
        (
            "/tmp/openhands_scale_swe_tests.json",
            json.dumps({"test_ids": _scale_swe_test_ids(data)}),
        ),
        ("/tmp/openhands_scale_swe_pytest.py", _SCALE_SWE_RUNNER),
    ]
    if data.get("f2p_script"):
        files.append(
            (f"{data['workdir']}/test_fail_to_pass.py", str(data["f2p_script"]))
        )
    return files


class SWEGymEvaluation(Evaluation):
    setup_timeout: int = Field(default=1200, ge=1)
    eval_timeout: int = Field(default=900, ge=1)
    sandbox_lifetime: int = Field(default=3600, ge=1)

    def prepare_instances(self) -> List[EvalInstance]:
        raw_df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=None,
            selected_instances_file=None,
        )
        normalized_df = pd.DataFrame(
            normalize_row(row.to_dict()) for _, row in raw_df.iterrows()
        )
        df = prepare_dataset(
            normalized_df,
            self.metadata.eval_limit,
            self.metadata.selected_instances_file,
        )
        instances: list[EvalInstance] = []
        for _, row in df.iterrows():
            data = _to_builtin_types(row.to_dict())
            instances.append(EvalInstance(id=str(data["instance_id"]), data=data))
        logger.info("Total SWE-Gym instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        del forward_env
        workspace = OpenSandboxWorkspace(
            image=str(instance.data["image"]),
            working_dir=str(instance.data["workdir"]),
            lifetime=self.sandbox_lifetime,
            resource_factor=resource_factor,
        )
        filemode = workspace.execute_command(
            "git config core.fileMode false", timeout=30
        )
        if filemode.exit_code != 0:
            workspace.__exit__(None, None, None)
            raise RuntimeError(
                "failed to disable Git file-mode tracking: "
                f"{filemode.stderr or filemode.stdout}"
            )
        if instance.data.get("dataset_kind") != "swe_gym":
            setup = workspace.execute_command(
                _pre_commands_for(instance.data), timeout=self.setup_timeout
            )
            if setup.exit_code != 0:
                workspace.__exit__(None, None, None)
                raise RuntimeError(
                    f"pre_commands failed with exit code {setup.exit_code}: "
                    f"{setup.stderr or setup.stdout}"
                )
            if instance.data.get("dataset_kind") == "scale_swe":
                self._prepare_scale_swe_workspace(workspace, instance)

        baseline = workspace.execute_command(
            _workspace_baseline_tree_command(), timeout=120
        )
        baseline_tree = (
            baseline.stdout.strip().splitlines()[-1] if baseline.stdout else ""
        )
        if baseline.exit_code != 0 or _GIT_TREE_HASH.fullmatch(baseline_tree) is None:
            workspace.__exit__(None, None, None)
            raise RuntimeError(
                "failed to capture initial Git worktree: "
                f"{baseline.stderr or baseline.stdout}"
            )
        instance.data["_workspace_baseline_tree"] = baseline_tree
        return workspace  # type: ignore[return-value]

    def _evaluate_swe_gym_patch(
        self,
        instance: EvalInstance,
        git_patch: str,
        agent_workspace: OpenSandboxWorkspace | None = None,
        resource_factor: int = 1,
    ) -> dict[str, Any]:
        """Grade after the agent is done, before hidden files become visible."""
        data = instance.data
        workdir = str(data["workdir"])
        evaluator_context = (
            nullcontext(agent_workspace)
            if agent_workspace is not None
            else OpenSandboxWorkspace(
                str(data["image"]),
                workdir,
                self.sandbox_lifetime,
                resource_factor=resource_factor,
            )
        )
        with evaluator_context as evaluator:
            assert evaluator is not None
            reset = evaluator.execute_command(
                "git reset --hard "
                f"{shlex.quote(str(data['base_commit']))} && git clean -fd",
                timeout=min(self.setup_timeout, 60),
            )
            if reset.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "evaluator_setup_failed",
                    "setup_exit_code": reset.exit_code,
                    "setup_stdout": reset.stdout[-4000:],
                    "setup_stderr": reset.stderr[-4000:],
                    "setup_timed_out": reset.timeout_occurred,
                }
            evaluator_commands = _evaluator_pre_commands_for(data)
            eval_script = _extract_plaintext_eval_script(evaluator_commands)
            if eval_script is None:
                setup = evaluator.execute_command(
                    evaluator_commands,
                    timeout=min(self.setup_timeout, 120),
                )
            else:
                setup = evaluator.execute_command(
                    "mkdir -p .swe_eval",
                    timeout=min(self.setup_timeout, 60),
                )
                if setup.exit_code == 0:
                    try:
                        _materialize_workspace_files(
                            evaluator,
                            [(f"{workdir}/.swe_eval/eval.sh", eval_script)],
                        )
                    except RuntimeError as exc:
                        return {
                            "resolved": False,
                            "reason": "evaluator_setup_failed",
                            "setup_exit_code": -1,
                            "setup_stdout": "",
                            "setup_stderr": str(exc)[-4000:],
                            "setup_timed_out": False,
                        }
                    setup = evaluator.execute_command(
                        f"chmod 755 {shlex.quote(workdir + '/.swe_eval/eval.sh')}",
                        timeout=min(self.setup_timeout, 60),
                    )
            if setup.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "evaluator_setup_failed",
                    "setup_exit_code": setup.exit_code,
                    "setup_stdout": setup.stdout[-4000:],
                    "setup_stderr": setup.stderr[-4000:],
                    "setup_timed_out": setup.timeout_occurred,
                }

            _materialize_workspace_files(
                evaluator,
                [
                    ("/tmp/openhands_candidate.patch", _terminate_patch(git_patch)),
                    (f"{workdir}/.swe_eval/test.patch", str(data["test_patch"])),
                ],
            )
            chmod = evaluator.execute_command(
                f"chmod 755 {shlex.quote(workdir + '/.swe_eval/eval.sh')}",
                timeout=30,
            )
            if chmod.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "evaluator_setup_failed",
                    "setup_exit_code": chmod.exit_code,
                    "setup_stdout": chmod.stdout[-4000:],
                    "setup_stderr": chmod.stderr[-4000:],
                    "setup_timed_out": chmod.timeout_occurred,
                }
            candidate_apply = evaluator.execute_command(
                "true"
                if not git_patch.strip()
                else _candidate_apply_command("/tmp/openhands_candidate.patch"),
                timeout=min(self.setup_timeout, 120),
            )
            if candidate_apply.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "candidate_patch_apply_failed",
                    "candidate_patch_applied": False,
                    "candidate_apply_exit_code": candidate_apply.exit_code,
                    "candidate_apply_stdout": candidate_apply.stdout[-4000:],
                    "candidate_apply_stderr": candidate_apply.stderr[-4000:],
                    "candidate_apply_timed_out": candidate_apply.timeout_occurred,
                }

            eval_timeout = int(data.get("eval_timeout_sec") or self.eval_timeout)
            result = evaluator.execute_command(
                str(data["eval_cmd"]), timeout=eval_timeout
            )
            marker = evaluator.execute_command(
                f"test -f {shlex.quote(f'{workdir}/{_SWE_GYM_TEST_PATCH_MARKER}')}",
                timeout=30,
            )
            test_patch_applied = marker.exit_code == 0
            resolved = result.exit_code == 0 and test_patch_applied
            reason = None
            candidate_broke_evaluator_input = None
            if result.timeout_occurred:
                reason = "eval_timeout"
            elif (
                result.exit_code == 75
                and _SWE_GYM_ISOLATED_TEST_KILLED in result.stderr
            ):
                reason = "eval_worker_killed"
            elif not test_patch_applied:
                candidate_broke_evaluator_input = _candidate_deleted_missing_path(
                    git_patch, result.stderr, workdir
                )
                reason = (
                    "tests_failed"
                    if candidate_broke_evaluator_input
                    else "test_patch_not_applied"
                )
            elif result.exit_code != 0:
                reason = "tests_failed"
            return {
                "resolved": resolved,
                "reason": reason,
                "grader": "swe_gym",
                "candidate_patch_applied": True,
                "test_patch_applied": test_patch_applied,
                "eval_exit_code": result.exit_code,
                "eval_stdout": result.stdout,
                "eval_stderr": result.stderr,
                "timed_out": result.timeout_occurred,
                "candidate_broke_evaluator_input": candidate_broke_evaluator_input,
            }

    def _prepare_scale_swe_workspace(
        self, workspace: OpenSandboxWorkspace, instance: EvalInstance
    ) -> None:
        problem = str(instance.data["problem_statement"])
        _materialize_workspace_files(
            workspace,
            [(f"{instance.data['workdir']}/PROBLEM_STATEMENT.md", problem)],
        )
        command = " && ".join(
            [
                "git config user.email 'openhands-eval@example.test'",
                "git config user.name 'OpenHands Eval'",
            ]
        )
        setup = workspace.execute_command(command, timeout=self.setup_timeout)
        if setup.exit_code != 0:
            raise RuntimeError(
                f"Scale-SWE setup failed: {setup.stderr or setup.stdout}"
            )

    def _evaluate_scale_swe_patch(
        self, instance: EvalInstance, git_patch: str
    ) -> dict[str, Any]:
        """Grade on a clean sandbox, matching Slime's Scale-SWE F2P/P2P contract."""
        data = instance.data
        workdir = str(data["workdir"])
        with OpenSandboxWorkspace(
            str(data["image"]), workdir, self.sandbox_lifetime
        ) as evaluator:
            pre = evaluator.execute_command(
                _pre_commands_for(data), timeout=self.setup_timeout
            )
            if pre.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "evaluator_setup_failed",
                    "stderr": pre.stderr[-4000:],
                }
            evaluator_pre = evaluator.execute_command(
                _evaluator_pre_commands_for(data), timeout=self.setup_timeout
            )
            if evaluator_pre.exit_code != 0:
                return {
                    "resolved": False,
                    "reason": "evaluator_setup_failed",
                    "stderr": evaluator_pre.stderr[-4000:],
                }
            files = _scale_swe_evaluator_files(data, git_patch)
            _materialize_workspace_files(evaluator, files)
            if data.get("f2p_patch"):
                preflight = evaluator.execute_command(
                    "git apply --check --whitespace=nowarn /tmp/openhands_f2p.patch",
                    timeout=self.setup_timeout,
                )
                _validate_f2p_preflight_result(preflight)
            candidate_apply = (
                "true"
                if not git_patch.strip()
                else _candidate_apply_command("/tmp/openhands_candidate.patch")
            )
            f2p_apply = (
                "true"
                if not data.get("f2p_patch")
                else "git apply --recount --check --whitespace=nowarn "
                "/tmp/openhands_f2p.patch "
                "&& git apply --recount --whitespace=nowarn "
                "/tmp/openhands_f2p.patch"
            )
            apply = (
                f"{candidate_apply} && {f2p_apply} "
                "&& python /tmp/openhands_scale_swe_pytest.py /tmp/openhands_scale_swe_tests.json"
            )
            result = evaluator.execute_command(apply, timeout=self.eval_timeout)
            grader_completed = _scale_swe_grader_completed(result)
            return {
                "resolved": result.exit_code == 0,
                "grader": "scale_swe",
                "grader_completed": grader_completed,
                "eval_exit_code": result.exit_code,
                "eval_stdout": result.stdout[-4000:],
                "eval_stderr": result.stderr[-4000:],
                "timed_out": result.timeout_occurred,
            }

    def _capture_conversation_archive(
        self, workspace: RemoteWorkspace, instance: EvalInstance
    ) -> None:
        # Local conversations are persisted by the controller callback.
        return None

    def evaluate_instance(
        self, instance: EvalInstance, workspace: RemoteWorkspace
    ) -> EvalOutput:
        assert isinstance(workspace, OpenSandboxWorkspace)
        tool = OpenSandboxTerminalTool.for_workspace(workspace)
        register_tool(OpenSandboxTerminalTool.name, tool)

        agent = Agent(
            llm=self.metadata.llm,
            tools=[Tool(name=OpenSandboxTerminalTool.name)],
            system_prompt_kwargs={"cli_mode": True},
        )

        controller_dir = (
            Path(self.metadata.eval_output_dir) / "controller_workspaces" / instance.id
        )
        controller_dir.mkdir(parents=True, exist_ok=True)
        local_workspace = LocalWorkspace(working_dir=controller_dir)
        persist_callback = build_event_persistence_callback(
            run_id=self.metadata.eval_output_dir,
            instance_id=instance.id,
            attempt=self.current_attempt,
        )
        conversation = Conversation(
            agent=agent,
            workspace=local_workspace,
            callbacks=[persist_callback],
            max_iteration_per_run=self.metadata.max_iterations,
        )

        instance.data["repo_path"] = str(instance.data["workdir"])
        instruction = get_instruction(
            instance.data, self.metadata, str(instance.data["workdir"])
        )
        conversation.send_message(instruction)
        try:
            with conversation_wall_clock_timeout(self.metadata.conversation_timeout):
                run_conversation_with_fake_user_response(conversation)
        except ConversationWallClockTimeoutError:
            logger.warning(
                "SWE-Gym conversation reached the %s-second wall-clock limit: %s",
                self.metadata.conversation_timeout,
                instance.id,
            )

        metrics = conversation.conversation_stats.get_combined_metrics()
        _warn_max_input_tokens(metrics, self.metadata.llm.max_input_tokens)

        baseline_tree = str(
            instance.data.get("_workspace_baseline_tree")
            or instance.data.get("base_commit")
            or "HEAD"
        )
        patch_result = workspace.execute_command(
            _candidate_diff_command(baseline_tree),
            timeout=120,
        )
        if patch_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {patch_result.stderr}")
        git_patch = patch_result.stdout
        candidate_patch_file = (
            Path(self.metadata.eval_output_dir) / f"{instance.id}.candidate.patch"
        )
        candidate_patch_file.write_text(git_patch)

        if instance.data.get("dataset_kind") == "scale_swe":
            test_result = self._evaluate_scale_swe_patch(instance, git_patch)
            resolved = bool(test_result["resolved"])
        else:
            test_result = self._evaluate_swe_gym_patch(instance, git_patch, workspace)
            resolved = bool(test_result["resolved"])
        logger.info(
            "SWE-Gym result %s: resolved=%s exit_code=%s",
            instance.id,
            resolved,
            test_result.get("eval_exit_code"),
        )

        messages = []
        tools_list = []
        convertible_events = [
            event
            for event in conversation.state.events
            if isinstance(event, LLMConvertibleEvent)
        ]
        for message in LLMConvertibleEvent.events_to_messages(convertible_events):
            messages.append(
                message.model_copy(
                    update={"send_reasoning_content": True}
                ).to_chat_dict()
            )
        for event in conversation.state.events:
            if isinstance(event, SystemPromptEvent):
                for event_tool in event.tools:
                    if isinstance(event_tool, ToolDefinition):
                        tools_list.append(event_tool.to_openai_tool())

        history_file = (
            Path(self.metadata.eval_output_dir) / f"{instance.id}.history.json"
        )
        history_file.write_text(
            json.dumps(
                {
                    "instance_id": instance.id,
                    "messages": messages,
                    "model": self.metadata.llm.model,
                    "tools": tools_list,
                    "test_result": {"git_patch": git_patch, **test_result},
                },
                indent=2,
            )
        )

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={"git_patch": git_patch, **test_result},
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=metrics,
        )


def main() -> None:
    prompt_dir = (Path(__file__).parent / "prompts").resolve()
    default_prompt_path = prompt_dir / "default.j2"
    parser = get_parser()
    parser.set_defaults(dataset=str(DEFAULT_DATASET), split="train", workspace="remote")
    parser.add_argument("--prompt-path", default=str(default_prompt_path))
    parser.add_argument("--setup-timeout", type=int, default=1200)
    parser.add_argument("--eval-timeout", type=int, default=900)
    parser.add_argument("--sandbox-lifetime", type=int, default=3600)
    args = parser.parse_args()

    if args.workspace != "remote":
        parser.error("SWE-Gym currently uses OpenSandbox; pass --workspace remote")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    llm_config = json.dumps(
        OmegaConf.to_container(OmegaConf.load(args.llm_config_path), resolve=True)
    )
    llm = LLM.model_validate_json(llm_config)
    structured_output_dir = construct_eval_output_dir(
        base_dir=args.output_dir,
        dataset_name=(
            args.dataset.replace("/", "__") + "-" + args.split.replace("/", "__")
        ),
        model_name=llm.model,
        max_iterations=args.max_iterations,
        eval_note=args.note,
    )
    metadata = EvalMetadata(
        llm=llm,
        dataset=args.dataset,
        dataset_split=args.split,
        max_iterations=args.max_iterations,
        eval_output_dir=structured_output_dir,
        details={"benchmark": "SWE-Gym", "execution_backend": "OpenSandbox"},
        prompt_path=args.prompt_path,
        eval_limit=args.n_limit,
        env_setup_commands=[],
        max_attempts=args.max_attempts,
        critic=create_critic(args),
        selected_instances_file=args.select,
        max_retries=args.max_retries,
        workspace_type=args.workspace,
        conversation_timeout=args.conversation_timeout,
    )
    evaluator = SWEGymEvaluation(
        metadata=metadata,
        num_workers=args.num_workers,
        setup_timeout=args.setup_timeout,
        eval_timeout=args.eval_timeout,
        sandbox_lifetime=args.sandbox_lifetime,
    )
    evaluator.run(on_result=get_default_on_result_writer(evaluator.output_path))


if __name__ == "__main__":
    main()
