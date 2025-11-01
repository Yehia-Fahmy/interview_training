"""Quantile estimator starter for the 8090.ai code challenge (hard variant)."""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class QuantileSummary:
    count: int
    p50: float
    p90: float
    p99: float
    memory_bytes: int


class Sketch:
    """Placeholder quantile sketch. Replace with t-digest, GK, or custom structure."""

    def __init__(self, target_error: float) -> None:
        self._target_error = target_error
        self._count = 0
        self._data: List[int] = []  # TODO: Replace with something memory efficient.

    def update(self, value: int) -> None:
        self._count += 1
        self._data.append(value)

    def summarize(self) -> QuantileSummary:
        if not self._data:
            return QuantileSummary(0, 0.0, 0.0, 0.0, self.memory_usage())

        sorted_data = sorted(self._data)
        def quantile(q: float) -> float:
            idx = int(q * (len(sorted_data) - 1))
            return float(sorted_data[idx])

        return QuantileSummary(
            count=self._count,
            p50=quantile(0.5),
            p90=quantile(0.9),
            p99=quantile(0.99),
            memory_bytes=self.memory_usage(),
        )

    def memory_usage(self) -> int:
        # Provide a rough estimate. Replace once _data changes.
        return len(self._data) * 8


def read_batches(buffer: io.BufferedReader, batch_size: int = 10_000) -> Iterable[List[int]]:
    chunk_size = batch_size * 8
    unpack = struct.Struct("<Q").unpack_from

    while True:
        raw = buffer.read(chunk_size)
        if not raw:
            break

        values: List[int] = []
        # Process full 8-byte chunks; ignore trailing bytes.
        for offset in range(0, len(raw) - len(raw) % 8, 8):
            (value,) = unpack(raw, offset)
            values.append(value)

        if values:
            yield values


def emit_summary(summary: QuantileSummary) -> None:
    payload = {
        "count": summary.count,
        "p50": summary.p50,
        "p90": summary.p90,
        "p99": summary.p99,
        "memory_bytes": summary.memory_bytes,
    }
    print(json.dumps(payload))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Streaming quantile estimator")
    parser.add_argument("--target-error", type=float, default=500.0, help="Maximum absolute error in microseconds")
    parser.add_argument("--profile", action="store_true", help="Enable tracemalloc-based allocation profiling")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional path to write periodic checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Restore estimator state from checkpoint")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.target_error <= 0:
        parser.error("--target-error must be positive")

    sketch = Sketch(target_error=args.target_error)

    batches_processed = 0
    for batch in read_batches(sys.stdin.buffer):
        for value in batch:
            sketch.update(value)

        batches_processed += 1
        summary = sketch.summarize()
        emit_summary(summary)

        # TODO: enforce memory ceiling and implement checkpointing/profiling hooks.

    if batches_processed == 0:
        emit_summary(sketch.summarize())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

