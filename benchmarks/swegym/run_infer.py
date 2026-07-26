from __future__ import annotations

import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List

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
from benchmarks.utils.dataset import get_dataset
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
        data = dict(metadata)
        data.setdefault("instance_id", row.get("label"))
        data.setdefault("problem_statement", row.get("prompt"))
    else:
        data = dict(row)

    required = {
        "instance_id",
        "image",
        "workdir",
        "problem_statement",
        "base_commit",
        "pre_commands",
        "eval_cmd",
    }
    missing = sorted(key for key in required if not data.get(key))
    if missing:
        raise ValueError(
            "SWE-Gym row is missing executable metadata "
            f"{missing}. Use a prepared *_osb.jsonl dataset, not the raw Parquet."
        )
    return data


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


class SWEGymEvaluation(Evaluation):
    setup_timeout: int = Field(default=1200, ge=1)
    eval_timeout: int = Field(default=900, ge=1)
    sandbox_lifetime: int = Field(default=3600, ge=1)

    def prepare_instances(self) -> List[EvalInstance]:
        df = get_dataset(
            dataset_name=self.metadata.dataset,
            split=self.metadata.dataset_split,
            eval_limit=self.metadata.eval_limit,
            selected_instances_file=self.metadata.selected_instances_file,
        )
        instances = []
        for _, row in df.iterrows():
            data = normalize_row(row.to_dict())
            instances.append(EvalInstance(id=str(data["instance_id"]), data=data))
        logger.info("Total SWE-Gym instances to process: %d", len(instances))
        return instances

    def prepare_workspace(
        self,
        instance: EvalInstance,
        resource_factor: int = 1,
        forward_env: list[str] | None = None,
    ) -> RemoteWorkspace:
        del resource_factor, forward_env
        workspace = OpenSandboxWorkspace(
            image=str(instance.data["image"]),
            working_dir=str(instance.data["workdir"]),
            lifetime=self.sandbox_lifetime,
        )
        setup = workspace.execute_command(
            str(instance.data["pre_commands"]), timeout=self.setup_timeout
        )
        if setup.exit_code != 0:
            workspace.__exit__(None, None, None)
            raise RuntimeError(
                f"pre_commands failed with exit code {setup.exit_code}: "
                f"{setup.stderr or setup.stdout}"
            )
        return workspace  # type: ignore[return-value]

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
            with conversation_wall_clock_timeout(
                self.metadata.conversation_timeout
            ):
                run_conversation_with_fake_user_response(conversation)
        except ConversationWallClockTimeoutError:
            logger.warning(
                "SWE-Gym conversation reached the %s-second wall-clock limit: %s",
                self.metadata.conversation_timeout,
                instance.id,
            )

        base_commit = str(instance.data["base_commit"])
        patch_result = workspace.execute_command(
            "git add -A -- . ':(exclude).swe_eval/**' && "
            f"git --no-pager diff --cached --no-color {base_commit} -- . "
            "':(exclude).swe_eval/**'",
            timeout=120,
        )
        if patch_result.exit_code != 0:
            raise RuntimeError(f"git diff failed: {patch_result.stderr}")
        git_patch = patch_result.stdout

        eval_timeout = int(instance.data.get("eval_timeout_sec") or self.eval_timeout)
        eval_result = workspace.execute_command(
            str(instance.data["eval_cmd"]), timeout=eval_timeout
        )
        resolved = eval_result.exit_code == 0
        logger.info(
            "SWE-Gym result %s: resolved=%s exit_code=%s",
            instance.id,
            resolved,
            eval_result.exit_code,
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
                    "test_result": {
                        "git_patch": git_patch,
                        "resolved": resolved,
                        "eval_exit_code": eval_result.exit_code,
                        "eval_stdout": eval_result.stdout,
                        "eval_stderr": eval_result.stderr,
                    },
                },
                indent=2,
            )
        )

        return EvalOutput(
            instance_id=instance.id,
            attempt=self.current_attempt,
            test_result={
                "git_patch": git_patch,
                "resolved": resolved,
                "eval_exit_code": eval_result.exit_code,
                "eval_stdout": eval_result.stdout,
                "eval_stderr": eval_result.stderr,
            },
            instruction=instruction,
            error=None,
            history=list(conversation.state.events),
            metrics=conversation.conversation_stats.get_combined_metrics(),
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
