"""Stream-processing starter (easy, clarified).

Goal
-----
Read log lines from stdin and, for each agent, maintain a fixed-size rolling
window of the last k payload sizes. After each valid line, emit summary stats
for that agent: average, median, and max over the window.

What you must implement
-----------------------
1) parse_line_with_reason(line) -> tuple[Optional[LogEntry], Optional[str]]
   - Parse "<timestamp> <agent_id> <payload_bytes>".
   - Return (LogEntry, None) on success; (None, reason) on failure.

2) estimate_memory_bytes(windows, window_size) -> int
   - Given the in-memory structure (dict[str -> deque[int]]), return an
     estimated number of bytes used. A simple, consistent estimate is fine:
     e.g., container overhead + per-int cost × number of stored ints.

3) Integrate --memory-profile in main()
   - When the flag is passed, print the estimated memory in bytes for an
     example scenario or the current (empty) structure and exit(0).
   - Keep it simple and deterministic.

You can use update_stats and emit_summary as-is.
"""

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


def parse_line_with_reason(line: str) -> Tuple[Optional[LogEntry], Optional[str]]:
    """TODO: Implement parsing and return a reason on failure.

    Expected format: "<timestamp> <agent_id> <payload_bytes>".
    - On success: return (LogEntry, None)
    - On failure: return (None, "reason for failure")
    """

    # TODO: replace the placeholder implementation below
    timestamp, agent_id, payload_bytes = line.split(" ")
    print(timestamp)
    print(agent_id)
    print(payload_bytes)
    print("=======")
    parts = line.strip().split()
    if len(parts) != 3:
        return None, "expected 3 fields: <timestamp> <agent_id> <payload_bytes>"

    ts, agent, payload_str = parts
    try:
        payload = int(payload_str)
    except ValueError:
        return None, "payload_bytes is not an integer"

    return LogEntry(timestamp=ts, agent_id=agent, payload_bytes=payload), None


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


def estimate_memory_bytes(windows: Dict[str, Deque[int]], window_size: int) -> int:
    """TODO: Implement a simple, consistent memory estimate.

    Hints:
    - You may assume a small constant overhead per dict entry and per deque.
    - For ints, you can assume 28 bytes each (typical CPython small int size),
      or choose another constant as long as you are consistent.
    - Example approach (very rough upper bound):
        bytes = dict_overhead
              + sum(per_agent_overhead + len(deque) * bytes_per_int for each agent)
    """

    # TODO: replace the placeholder estimation below with your own formula
    bytes_per_int = 28
    dict_overhead = 1024
    per_agent_overhead = 128

    total = dict_overhead
    for _agent, dq in windows.items():
        total += per_agent_overhead + len(dq) * bytes_per_int
    return total


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rolling payload stats per agent (clarified)")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size per agent")
    parser.add_argument("--memory-profile", action="store_true", help="Print estimated memory usage and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.window <= 0:
        parser.error("--window must be positive")

    windows: Dict[str, Deque[int]] = {}

    # TODO: Implement --memory-profile behavior.
    # Suggested behavior: estimate the empty structure (or a small example)
    # and exit. Keep it deterministic.
    if args.memory_profile:
        estimated = estimate_memory_bytes(windows, args.window)
        print(estimated)
        return 0

    for line_no, raw in enumerate(sys.stdin, start=1):
        entry, reason = parse_line_with_reason(raw)
        if entry is None:
            print(f"Skipping malformed line {line_no} ({reason}): {raw.rstrip()}", file=sys.stderr)
            continue

        avg, med, max_payload = update_stats(windows, entry, args.window)
        emit_summary(entry, avg, med, max_payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


