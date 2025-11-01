# Code Challenge – Easy (Clarified)

**Theme**: Memory-aware iteration & robust input handling.

You are handed logs emitted by an agent orchestration service. Each line encodes a timestamp, an agent identifier, and a payload size in bytes. Compute per-agent rolling statistics while streaming the file (no full-file loads).

You can choose between two starters:
- `starter.py`: almost complete reference-style solution.
- `starter_clear.py`: clarified scaffold with explicit TODOs (recommended for practice).

## What you need to build (using `starter_clear.py`)
Implement the following clearly marked TODOs:

1) `parse_line_with_reason(line)`
   - Parse lines of the form: `<timestamp> <agent_id> <payload_bytes>`.
   - On success: return `(LogEntry, None)`; on failure: `(None, "reason")`.
   - The main loop must print the reason to stderr for malformed lines.

2) `estimate_memory_bytes(windows, window_size)`
   - Given `windows: dict[str, deque[int]]`, return a simple, consistent estimate of bytes used.
   - A rough, deterministic formula is sufficient (container overhead + ints).

3) `--memory-profile` integration in `main()`
   - When provided, print the estimated memory usage (in bytes) and exit.
   - Keep behavior deterministic; estimating the current (empty) structure is acceptable.

The following helpers are provided and can be used as-is:
- `update_stats(windows, entry, window_size)`
- `emit_summary(entry, avg, med, max_payload)`

## I/O Behavior
1. Read from stdin, process line by line.
2. Maintain per-agent rolling window of size `k` (CLI: `--window`, default `5`).
3. After each valid line, print: `"<ts> <agent> avg=<..> median=<..> max=<..>"`.
4. On malformed line, print warning to stderr including line number and reason; continue.

## Run examples
```bash
# Use the clarified scaffold
python starter_clear.py < sample.log

# Change the window size
python starter_clear.py --window 10 < sample.log

# Memory profile only
python starter_clear.py --memory-profile
```

## Constraints & Expectations
- Aim to finish core requirements in 30–45 minutes.
- Standard library only (no third-party deps).
- Favor small, testable helpers.

## Deliverables
- Complete the TODOs in `starter_clear.py`.
- Add a brief `NOTES.md` summarizing complexity, trade-offs, and potential extensions.

## Extension Ideas (Optional)
- Percentiles or histogram alerts within bounded memory.
- Robust timestamp parsing and validation.