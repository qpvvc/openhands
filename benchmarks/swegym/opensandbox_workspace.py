from __future__ import annotations

import asyncio
import os
import shlex
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence, cast

from openhands.sdk.tool import ToolAnnotations, ToolDefinition, ToolExecutor
from openhands.sdk.workspace.models import CommandResult
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation
from openhands.tools.terminal.metadata import CmdOutputMetadata


class OpenSandboxWorkspace:
    """Synchronous OpenSandbox adapter used by the local OpenHands agent."""

    def __init__(self, image: str, working_dir: str, lifetime: int = 3600) -> None:
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
        self._loop = asyncio.new_event_loop()
        self._sandbox = self._loop.run_until_complete(
            Sandbox.create(
                image,
                connection_config=config,
                timeout=timedelta(seconds=lifetime),
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
        return 0

    def execute_command(
        self,
        command: str,
        cwd: str | Path | None = None,
        timeout: float = 120.0,
    ) -> CommandResult:
        workdir = str(cwd or self.working_dir)
        inner = f"cd {shlex.quote(workdir)} && {command}"
        wrapped = f"bash -lc {shlex.quote(inner)}"
        try:
            commands = cast(Any, self._sandbox.commands)
            try:
                operation = commands.run(
                    wrapped, timeout=timedelta(seconds=max(1, int(timeout)))
                )
            except TypeError:
                operation = commands.run(wrapped)
            execution = self._loop.run_until_complete(operation)
            logs = getattr(execution, "logs", None)
            stdout = self._join_logs(getattr(logs, "stdout", None))
            stderr = self._join_logs(getattr(logs, "stderr", None))
            if not stdout:
                stdout = str(getattr(execution, "stdout", "") or "")
            if not stderr:
                stderr = str(getattr(execution, "stderr", "") or "")
            exit_code = self._exit_code(execution)
            return CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timeout_occurred=False,
            )
        except TimeoutError as exc:
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=str(exc),
                timeout_occurred=True,
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
