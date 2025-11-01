"""Critical path analyzer starter for the 8090.ai code challenge (medium variant)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Task:
    task_id: str
    dependencies: List[str] = field(default_factory=list)
    cost_us: int = 0


def parse_tasks(stream: Iterable[str]) -> Dict[str, Task]:
    tasks: Dict[str, Task] = {}
    for line_no, raw in enumerate(stream, start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON on line {line_no}: {exc}", file=sys.stderr)
            continue

        task_id = payload.get("task_id")
        if not task_id:
            print(f"Missing task_id on line {line_no}", file=sys.stderr)
            continue

        deps = payload.get("dependencies", [])
        cost = payload.get("cost_us", 0)
        tasks[task_id] = Task(task_id=task_id, dependencies=list(deps), cost_us=int(cost))

    return tasks


def detect_cycles(tasks: Dict[str, Task]) -> Optional[List[str]]:
    """Return a list representing one cycle, or None if acyclic."""

    visited: Dict[str, int] = defaultdict(int)  # 0=unseen,1=visiting,2=done
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        state = visited[node]
        if state == 1:
            # Found a back edge; slice stack to form the cycle.
            if node in stack:
                idx = stack.index(node)
                return stack[idx:] + [node]
            return [node, node]
        if state == 2:
            return None

        visited[node] = 1
        stack.append(node)
        for dep in tasks.get(node, Task(node)).dependencies:
            cycle = dfs(dep)
            if cycle:
                return cycle
        stack.pop()
        visited[node] = 2
        return None

    for task_id in tasks:
        if visited[task_id] == 0:
            cycle = dfs(task_id)
            if cycle:
                return cycle
    return None


def compute_critical_paths(tasks: Dict[str, Task]) -> Dict[str, Tuple[int, List[str]]]:
    """Placeholder implementation—fill in with memoized traversal."""

    results: Dict[str, Tuple[int, List[str]]] = {}
    # TODO: Implement dynamic programming to compute cumulative cost and critical path.
    for task_id, task in tasks.items():
        results[task_id] = (task.cost_us, [task_id])
    return results


def emit_results(results: Dict[str, Tuple[int, List[str]]]) -> None:
    for task_id, (total_cost, path) in results.items():
        payload = {
            "task_id": task_id,
            "total_cost_us": total_cost,
            "critical_path": path,
        }
        print(json.dumps(payload))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute critical paths for agent tasks")
    parser.add_argument("--profile", action="store_true", help="Run under cProfile and show top hotspots")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=0,
        help="Optional parallelism budget for extra analysis",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    tasks = parse_tasks(sys.stdin)

    cycle = detect_cycles(tasks)
    if cycle:
        print(f"Cycle detected: {' -> '.join(cycle)}", file=sys.stderr)
        # Downstream logic should decide how to skip or handle these nodes.

    results = compute_critical_paths(tasks)
    emit_results(results)

    if args.max_parallelism:
        print(
            "Parallelism estimates not yet implemented—fill this in as part of the stretch goal.",
            file=sys.stderr,
        )

    if args.profile:
        print("Tip: wrap compute_critical_paths with cProfile.runctx()", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

