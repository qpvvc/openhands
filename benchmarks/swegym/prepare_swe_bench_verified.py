#!/usr/bin/env python3
"""Prepare local SWE-bench Verified data for the OpenSandbox evaluator."""

from __future__ import annotations

import argparse
import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import (
    MAP_REPO_TO_EXT,
    MAP_REPO_VERSION_TO_SPECS,
    TestSpec,
)


DEFAULT_IMAGE_PREFIX = "10.124.120.15:30001/zj021/swebench"
EXPECTED_ROWS = 500
REQUIRED_COLUMNS = {
    "repo",
    "instance_id",
    "base_commit",
    "test_patch",
    "problem_statement",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}


def _source_parquet(source: Path) -> Path:
    if source.is_file():
        return source
    candidates = sorted(source.glob("data/*.parquet"))
    if not candidates:
        candidates = sorted(source.glob("*.parquet"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one parquet file under {source}, found {candidates}"
        )
    return candidates[0]


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _harbor_auth() -> tuple[str, str] | None:
    username = os.getenv("HARBOR_USERNAME", "zjdev")
    password = os.getenv("HARBOR_PASSWORD")
    if password:
        return username, password

    config_path = os.getenv("HARBOR_DOCKER_CONFIG")
    if not config_path:
        return None
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    auths = config.get("auths", {})
    registry = os.getenv("HARBOR_REGISTRY", "10.124.120.15:30001")
    entry = auths.get(f"http://{registry}") or auths.get(registry)
    encoded = entry.get("auth") if isinstance(entry, dict) else None
    if not encoded:
        return None
    decoded = base64.b64decode(encoded).decode("utf-8")
    parts = decoded.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"invalid Docker auth entry in {config_path}")
    return parts[0], parts[1]


def verify_harbor_images(image_prefix: str, image_tags: set[str]) -> None:
    """Check the tag set and every manifest in the local Harbor registry."""
    registry, project, repository = image_prefix.split("/", 2)
    api_root = os.getenv("HARBOR_API_URL", f"http://{registry}").rstrip("/")
    auth = _harbor_auth()
    session = requests.Session()
    session.auth = auth
    headers = {
        "Accept": (
            "application/vnd.oci.image.manifest.v1+json, "
            "application/vnd.docker.distribution.manifest.v2+json"
        )
    }
    tags_url = f"{api_root}/v2/{project}/{repository}/tags/list"
    response = session.get(tags_url, headers=headers, timeout=20)
    if response.status_code in (401, 403):
        raise RuntimeError(
            f"Harbor returned HTTP {response.status_code}; set HARBOR_PASSWORD "
            "or HARBOR_DOCKER_CONFIG for the zjdev account"
        )
    response.raise_for_status()
    actual_tags = set(response.json().get("tags") or [])
    missing_tags = sorted(image_tags - actual_tags)
    extra_tags = sorted(actual_tags - image_tags)
    if missing_tags:
        raise RuntimeError(
            f"Harbor is missing {len(missing_tags)} SWE-bench tags; "
            f"first missing: {missing_tags[:5]}"
        )

    def check_manifest(tag: str) -> tuple[str, int]:
        url = f"{api_root}/v2/{project}/{repository}/manifests/{quote(tag, safe='')}"
        result = session.head(url, headers=headers, timeout=20)
        return tag, result.status_code

    with ThreadPoolExecutor(max_workers=32) as executor:
        statuses = list(executor.map(check_manifest, sorted(image_tags)))
    unavailable = [(tag, status) for tag, status in statuses if status != 200]
    if unavailable:
        raise RuntimeError(
            f"Harbor has {len(unavailable)} unavailable manifests; "
            f"first failures: {unavailable[:5]}"
        )
    print(
        f"Harbor verified: {len(image_tags)} expected tags, "
        f"{len(extra_tags)} extra tags, all manifests returned HTTP 200"
    )


def _test_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return [str(item) for item in (value or [])]


def _official_eval_script(metadata: dict[str, Any]) -> str:
    """Materialize the official SWE-bench TestSpec script for Slime's evaluator."""
    specs = MAP_REPO_VERSION_TO_SPECS[metadata["repo"]][metadata["version"]]
    eval_script_list = make_eval_script_list(
        metadata,
        specs,
        "testbed",
        metadata["workdir"],
        metadata["base_commit"],
        metadata["test_patch"],
    )
    test_spec = TestSpec(
        instance_id=metadata["instance_id"],
        repo=metadata["repo"],
        version=metadata["version"],
        repo_script_list=[],
        eval_script_list=eval_script_list,
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=_test_ids(metadata["FAIL_TO_PASS"]),
        PASS_TO_PASS=_test_ids(metadata["PASS_TO_PASS"]),
        language=MAP_REPO_TO_EXT[metadata["repo"]],
        docker_specs=specs.get("docker_specs", {}),
        namespace=None,
    )
    return test_spec.eval_script


def prepare_dataset(source: Path, output: Path, image_prefix: str) -> set[str]:
    frame = pd.read_parquet(_source_parquet(source))
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"dataset is missing columns: {missing_columns}")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(frame)}")

    ids = [str(value) for value in frame["instance_id"]]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("instance_id values must be non-empty and unique")

    image_tags: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for _, row in frame.iterrows():
            instance_id = str(row["instance_id"])
            tag = f"sweb.eval.x86_64.{instance_id.lower()}"
            image_tags.add(tag)
            metadata = {
                "_dataset_kind": "swe_bench_verified",
                "instance_id": instance_id,
                "repo": str(row["repo"]),
                "version": str(row["version"]),
                "base_commit": str(row["base_commit"]),
                "problem_statement": str(row["problem_statement"]),
                "hints_text": str(row.get("hints_text") or ""),
                "FAIL_TO_PASS": str(row["FAIL_TO_PASS"]),
                "PASS_TO_PASS": str(row["PASS_TO_PASS"]),
                "test_patch": str(row["test_patch"]),
                "image": f"{image_prefix}:{tag}",
                "workdir": "/testbed",
                "evaluator_pre_commands": "true",
                "eval_cmd": "bash /testbed/.swe_eval/eval.sh",
            }
            metadata["eval_script"] = _official_eval_script(metadata)
            output_row = {
                "prompt": metadata["problem_statement"],
                "label": instance_id,
                "metadata": {
                    key: _json_value(value) for key, value in metadata.items()
                },
            }
            stream.write(json.dumps(output_row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(image_tags)} SWE-bench Verified rows: {output}")
    return image_tags


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-prefix", default=DEFAULT_IMAGE_PREFIX)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()

    image_tags = prepare_dataset(
        args.source, args.output, args.image_prefix.rstrip("/")
    )
    if args.verify_images:
        verify_harbor_images(args.image_prefix.rstrip("/"), image_tags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
