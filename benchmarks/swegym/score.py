from __future__ import annotations

import argparse
import json
from pathlib import Path


def score_file(path: Path) -> tuple[int, int, list[str]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    resolved_ids = [
        str(row["instance_id"])
        for row in rows
        if bool((row.get("test_result") or {}).get("resolved"))
    ]
    return len(resolved_ids), len(rows), resolved_ids


def score_files(paths: list[Path]) -> tuple[int, int, list[str]]:
    results = [score_file(path) for path in paths]
    return (
        sum(result[0] for result in results),
        sum(result[1] for result in results),
        [instance_id for result in results for instance_id in result[2]],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SWE-Gym resolved score")
    parser.add_argument("output", type=Path, nargs="+")
    args = parser.parse_args()
    resolved, total, resolved_ids = score_files(args.output)
    rate = 100.0 * resolved / total if total else 0.0
    print(f"resolved: {resolved}/{total} ({rate:.2f}%)")
    if resolved_ids:
        print("resolved instances:")
        for instance_id in resolved_ids:
            print(f"  {instance_id}")


if __name__ == "__main__":
    main()
