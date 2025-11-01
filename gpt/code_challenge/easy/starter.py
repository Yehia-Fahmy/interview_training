"""Stream-processing starter for the 8090.ai code challenge (easy variant)."""

from __future__ import annotations

import argparse
import sys
from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Deque, Dict, Iterable, Optional, Tuple


@dataclass
class LogEntry:
    timestamp: str
    agent_id: str
    payload_bytes: int


def parse_line(line: str) -> Optional[LogEntry]:
    """Parse a raw log line into a LogEntry.

    Expected format: "<timestamp> <agent_id> <payload_bytes>".
    Returns None if the line is malformed.
    """

    parts = line.strip().split()
    if len(parts) != 3:
        return None

    ts, agent, payload_str = parts
    try:
        payload = int(payload_str)
    except ValueError:
        return None

    return LogEntry(timestamp=ts, agent_id=agent, payload_bytes=payload)


def update_stats(
    windows: Dict[str, Deque[int]], entry: LogEntry, window_size: int
) -> Tuple[float, float, int]:
    """Update the rolling window for an agent and return basic statistics."""

    window = windows.setdefault(entry.agent_id, deque(maxlen=window_size))
    window.append(entry.payload_bytes)

    values = list(window)
    avg = sum(values) / len(values)
    med = median(values)
    max_payload = max(values)

    return avg, med, max_payload


def emit_summary(entry: LogEntry, avg: float, med: float, max_payload: int) -> None:
    print(f"{entry.timestamp} {entry.agent_id} avg={avg:.2f} median={med:.2f} max={max_payload}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling payload stats per agent")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size per agent")
    parser.add_argument("--memory-profile", action="store_true", help="Print estimated memory usage and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.window <= 0:
        parser.error("--window must be positive")

    windows: Dict[str, Deque[int]] = {}

    for line_no, raw in enumerate(sys.stdin, start=1):
        entry = parse_line(raw)
        if entry is None:
            print(f"Skipping malformed line {line_no}: {raw.rstrip()}", file=sys.stderr)
            continue

        avg, med, max_payload = update_stats(windows, entry, args.window)
        emit_summary(entry, avg, med, max_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

