from __future__ import annotations

import asyncio
import os
import shlex
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence, cast
from uuid import uuid4

from openhands.sdk.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.sdk.workspace.models import CommandResult
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation
from openhands.tools.terminal.metadata import CmdOutputMetadata


_LOCAL_TIMEOUT_GRACE_SECONDS = 10.0
_THREAD_LIMIT_ENV = (
    "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
    "NINJAFLAGS=-j2 MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2 MAX_JOBS=2"
)


def _command_prefix_for_image(image: str) -> str:
    assignments = _THREAD_LIMIT_ENV
    normalized_image = image.lower()
    if "dask_s_dask-" in normalized_image:
        assignments += " DASK_NUM_WORKERS=2"
    if "modin-project_s_modin-" in normalized_image:
        assignments += " MODIN_CPUS=2"
    return f"export {assignments} && "


def _resource_for_image(image: str, resource_factor: int = 1) -> dict[str, str]:
    normalized_image = image.lower()
    factor = max(1, resource_factor)
    if "project-monai_s_monai-" in normalized_image:
        return {"cpu": str(2 * factor), "memory": f"{8 * factor}Gi"}
    if any(
        marker in normalized_image
        for marker in (
            "dask_s_dask-",
            "modin-project_s_modin-",
            "pandas-dev_s_pandas-",
        )
    ):
        return {"cpu": str(2 * factor), "memory": f"{4 * factor}Gi"}
    return {"cpu": str(factor), "memory": f"{2 * factor}Gi"}


class OpenSandboxWorkspace:
    """Synchronous OpenSandbox adapter used by the local OpenHands agent."""

    def __init__(
        self,
        image: str,
        working_dir: str,
        lifetime: int = 3600,
        resource_factor: int = 1,
    ) -> None:
        api_key = os.getenv("SANDBOX_API_KEY")
        if not api_key:
            raise ValueError("SANDBOX_API_KEY environment variable is not set")

        from opensandbox import Sandbox
        from opensandbox.config import ConnectionConfig

        domain = os.getenv("SANDBOX_DOMAIN", "10.124.120.15:30003")
        request_timeout = int(os.getenv("SANDBOX_REQUEST_TIMEOUT_SEC", "600"))
        config = ConnectionConfig(
            domain=domain,
            api_key=api_key,
            request_timeout=timedelta(seconds=request_timeout),
        )

        self.working_dir = working_dir
        self._command_prefix = _command_prefix_for_image(image)
        self._loop = asyncio.new_event_loop()
        self._sandbox = self._loop.run_until_complete(
            Sandbox.create(
                image,
                connection_config=config,
                timeout=timedelta(seconds=lifetime),
                resource=_resource_for_image(image, resource_factor),
            )
        )

    @staticmethod
    def _join_logs(items: Any) -> str:
        if not items:
            return ""
        return "\n".join(
            str(getattr(item, "text", item)).rstrip("\n") for item in items
        )

    @staticmethod
    def _exit_code(execution: Any) -> int:
        for name in ("exit_code", "returncode", "return_code", "code"):
            value = getattr(execution, name, None)
            if value is not None:
                return int(value)
        raise RuntimeError("OpenSandbox execution completed without an exit code")

    @staticmethod
    def _extract_exit_marker(text: str, marker: str) -> tuple[int | None, str]:
        prefix = f"{marker}="
        exit_code = None
        kept: list[str] = []
        for line in text.splitlines(keepends=True):
            value = line.rstrip("\r\n")
            if value.startswith(prefix):
                candidate = value[len(prefix) :]
                if candidate.lstrip("-").isdigit():
                    exit_code = int(candidate)
                    continue
            kept.append(line)
        return exit_code, "".join(kept).rstrip("\n")

    @staticmethod
    def _run_command_operation(commands: Any, command: str, timeout: float) -> Any:
        timeout_delta = timedelta(seconds=max(1, int(timeout)))
        try:
            from opensandbox.models.execd import RunCommandOpts

            return commands.run(
                command,
                opts=RunCommandOpts(timeout=timeout_delta),
            )
        except (ImportError, TypeError):
            try:
                return commands.run(command, timeout=timeout_delta)
            except TypeError:
                return commands.run(command)

    @classmethod
    async def _recover_exit_code_file(
        cls, commands: Any, path: str, timeout: float
    ) -> int:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        recovery_command = f"cat {shlex.quote(path)} 2>/dev/null; sleep 1"
        last_error: Exception | None = None
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    "OpenSandbox exit-code file recovery exceeded the command timeout"
                ) from last_error
            try:
                operation = cls._run_command_operation(
                    commands, recovery_command, min(5.0, remaining)
                )
                execution = await asyncio.wait_for(
                    operation, timeout=min(6.0, remaining)
                )
                logs = getattr(execution, "logs", None)
                stdout = cls._join_logs(getattr(logs, "stdout", None))
                if not stdout:
                    stdout = str(getattr(execution, "stdout", "") or "")
                for line in reversed(stdout.splitlines()):
                    value = line.strip()
                    if value.lstrip("-").isdigit():
                        return int(value)
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(min(0.25, max(0.0, deadline - loop.time())))

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 120.0,
    ) -> CommandResult:
        workdir = str(cwd or self.working_dir)
        prefix = getattr(self, "_command_prefix", "")
        inner = f"cd {shlex.quote(workdir)} && {prefix}{command}"
        exit_marker = f"__OPENHANDS_EXIT_CODE_{uuid4().hex}__"
        exit_code_path = f"/tmp/{exit_marker}.status"
        wrapped = (
            f"bash -lc {shlex.quote(inner)}; __openhands_status=$?; "
            f"printf '%s\\n' \"$__openhands_status\" > "
            f"{shlex.quote(exit_code_path)}; "
            f"printf '\\n{exit_marker}=%s\\n' \"$__openhands_status\" >&2; "
            "sleep 0.1; "
            'exit "$__openhands_status"'
        )
        commands = cast(Any, self._sandbox.commands)
        try:
            operation = self._run_command_operation(commands, wrapped, timeout)
            hard_timeout = max(1.0, float(timeout)) + _LOCAL_TIMEOUT_GRACE_SECONDS
            deadline = self._loop.time() + hard_timeout
            execution = self._loop.run_until_complete(
                asyncio.wait_for(operation, timeout=hard_timeout)
            )
            logs = getattr(execution, "logs", None)
            stdout = self._join_logs(getattr(logs, "stdout", None))
            stderr = self._join_logs(getattr(logs, "stderr", None))
            if not stdout:
                stdout = str(getattr(execution, "stdout", "") or "")
            if not stderr:
                stderr = str(getattr(execution, "stderr", "") or "")
            marker_exit_code, stderr = self._extract_exit_marker(stderr, exit_marker)
            try:
                exit_code = self._exit_code(execution)
            except RuntimeError:
                if marker_exit_code is not None:
                    exit_code = marker_exit_code
                else:
                    remaining = max(0.0, deadline - self._loop.time())
                    try:
                        exit_code = self._loop.run_until_complete(
                            self._recover_exit_code_file(
                                commands,
                                exit_code_path,
                                remaining,
                            )
                        )
                    except TimeoutError as exc:
                        raise TimeoutError(
                            "OpenSandbox command reached its deadline without a "
                            "recoverable exit code"
                        ) from exc
            return CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timeout_occurred=False,
            )
        except TimeoutError as exc:
            try:
                exit_code = self._loop.run_until_complete(
                    self._recover_exit_code_file(
                        commands,
                        exit_code_path,
                        10.0,
                    )
                )
            except TimeoutError:
                pass
            else:
                return CommandResult(
                    command=command,
                    exit_code=exit_code,
                    stdout="",
                    stderr="",
                    timeout_occurred=False,
                )
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=(
                    str(exc) or "OpenSandbox command exceeded its local hard timeout"
                ),
                timeout_occurred=True,
            )

    def write_file(self, path: str, content: str, mode: int = 755) -> None:
        files = cast(Any, self._sandbox.files)
        self._loop.run_until_complete(
            asyncio.wait_for(
                files.write_file(path, content, mode=mode),
                timeout=120.0,
            )
        )

    def __enter__(self) -> OpenSandboxWorkspace:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self._loop.run_until_complete(self._sandbox.kill())
        finally:
            self._loop.close()


class OpenSandboxTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    def __init__(self, workspace: OpenSandboxWorkspace) -> None:
        self.workspace = workspace

    def __call__(
        self, action: TerminalAction, conversation: Any = None
    ) -> TerminalObservation:
        if action.is_input:
            return TerminalObservation.from_text(
                "OpenSandbox commands are non-interactive; rerun the command with a "
                "sufficient timeout.",
                is_error=True,
                command=action.command,
                exit_code=2,
                metadata=CmdOutputMetadata(
                    exit_code=2, working_dir=self.workspace.working_dir
                ),
            )

        timeout = action.timeout or 120.0
        result = self.workspace.execute_command(action.command, timeout=timeout)
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        return TerminalObservation.from_text(
            output,
            is_error=result.exit_code != 0,
            command=action.command,
            exit_code=result.exit_code,
            timeout=result.timeout_occurred,
            metadata=CmdOutputMetadata(
                exit_code=result.exit_code,
                working_dir=self.workspace.working_dir,
            ),
        )


class OpenSandboxTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    name = "opensandbox_terminal"

    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence[OpenSandboxTerminalTool]:
        raise RuntimeError("OpenSandboxTerminalTool must be registered as an instance")

    @classmethod
    def for_workspace(cls, workspace: OpenSandboxWorkspace) -> OpenSandboxTerminalTool:
        return cls(
            action_type=TerminalAction,
            observation_type=TerminalObservation,
            description=(
                "Execute a non-interactive bash command inside the SWE-Gym sandbox. "
                f"Every command starts in {workspace.working_dir}; use one shell "
                "command for operations that require activation or directory state. "
                "Set timeout for tests or other long-running commands."
            ),
            annotations=ToolAnnotations(
                title="SWE-Gym terminal",
                readOnlyHint=False,
                destructiveHint=True,
                idempotentHint=False,
                openWorldHint=False,
            ),
            executor=OpenSandboxTerminalExecutor(workspace),
        )
