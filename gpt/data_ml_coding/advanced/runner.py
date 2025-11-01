"""Entry point for evaluating agent interactions and making policy decisions."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

from evaluation import compute_metrics, load_interactions
from policy import Thresholds, decide_action


def load_records(path: pathlib.Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent evaluation and governance runner")
    parser.add_argument("data", type=pathlib.Path, help="Path to interactions JSONL file")
    parser.add_argument("--thresholds", type=pathlib.Path, default=None, help="Optional threshold override JSON")
    args = parser.parse_args()

    overrides = {}
    if args.thresholds:
        overrides = json.loads(args.thresholds.read_text())

    thresholds = Thresholds(**overrides)
    interactions = load_interactions(load_records(args.data))
    metrics = compute_metrics(interactions)
    action = decide_action(metrics, thresholds)

    report = {
        "metrics": metrics,
        "decision": action,
        "thresholds": thresholds.__dict__,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

