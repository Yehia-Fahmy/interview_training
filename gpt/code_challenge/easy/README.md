# Code Challenge – Easy

**Theme**: Memory-aware iteration & input sanitization.

You are handed logs emitted by an agent orchestration service. Each line encodes a timestamp, an agent identifier, and a payload size in bytes. The goal is to compute per-agent rolling statistics while streaming the file with sub-linear memory.

## Requirements
1. Read log lines from standard input (no loading the entire file).
2. For each agent, maintain a fixed-size rolling window of the last `k` payload sizes (configurable; default `k = 5`).
3. After processing each line, emit to stdout a summary line containing:
   - timestamp
   - agent id
   - rolling average payload size (rounded to 2 decimals)
   - rolling median payload size (if the window has fewer than `k` entries, the median of what exists)
   - rolling maximum payload size
4. If a line is malformed, skip it but emit a warning to stderr containing the line number and the reason.

## Stretch Goals
- Support a `--memory-profile` flag that prints total bytes allocated for core data structures.
- Allow dynamically changing `k` based on CLI argument while keeping the per-agent memory footprint bounded.

## Constraints & Expectations
- Aim to finish core requirements in 45–60 minutes.
- Write helper functions that can be unit-tested independently (window maintenance, stats calculation).
- Avoid third-party dependencies; use only the Python standard library.

## Deliverables
- Implement logic in `starter.py`.
- Provide a brief write-up (bullet list) summarizing complexity and trade-offs in a `NOTES.md` alongside your solution.
- Prepare to explain how you would extend this to include percentiles or histogram-based alerts without blowing memory.

