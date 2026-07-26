import json
import time
from pathlib import Path

import pytest

from benchmarks.swegym.run_infer import (
    ConversationWallClockTimeoutError,
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


def test_normalize_rejects_raw_row() -> None:
    with pytest.raises(ValueError, match="prepared .*_osb.jsonl"):
        normalize_row({"instance_id": "org__repo-1", "problem_statement": "x"})


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
