# Code Challenge – Hard

**Theme**: Cache-efficient streaming quantile estimator.

Software Factory needs to monitor tail latency of agent actions in real time. You will build a memory-bounded estimator that ingests millions of samples per minute while reporting the 50th, 90th, and 99th percentile latencies with tight error guarantees.

## Requirements
1. Consume unsigned 64-bit integers from stdin. Data arrives as binary little-endian values (no newlines). You must rely on `sys.stdin.buffer.read` and avoid reading the entire stream at once.
2. Maintain a data structure that approximates requested quantiles with absolute error ≤ 500 microseconds. Samples range from 0 to 10⁹.
3. After every 10⁴ samples, emit a JSON line containing:
   - `count`
   - `p50`
   - `p90`
   - `p99`
   - `memory_bytes`: estimated memory currently in use by your sketch (excluding Python interpreter overhead, but include buffers you allocate).
4. Guarantee total memory devoted to the estimator stays below 1.5 MB regardless of stream length.
5. Provide a CLI flag `--target-error` to override the 500 microsecond requirement.

## Stretch Goals
- Expose a `--profile` option that prints allocation hotspots using `tracemalloc`.
- Allow checkpointing and restore: `--checkpoint <path>` writes state every N batches; `--resume <path>` restores from disk.

## Constraints & Expectations
- Avoid naive storing of all samples; you must implement a sketch (P², t-digest, GK algorithm, or custom binning).
- Favor `array`, `bisect`, `heapq`, or memoryview-based buffers to minimize Python object overhead.
- Expect to justify error guarantees and memory accounting to an interviewer.

## Deliverables
- Implement logic in `starter.py` (structure provided) and add micro-benchmarks as needed.
- Record error bounds, profiling notes, and production considerations in `NOTES.md`.
- Prepare an explanation for how you would parallelize ingestion across cores while preserving accuracy.

