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

## Run with provided sample and expected output

From the `gpt/code_challenge/easy/` directory:

```bash
# 1) Run against the provided sample log (default window=3 here for grading)
python starter_clear.py --window 3 < sample.log

# Expected stdout:
```
```
2025-11-01T10:00:00Z a1 avg=100.00 median=100.00 max=100
2025-11-01T10:00:01Z a1 avg=150.00 median=150.00 max=200
2025-11-01T10:00:02Z a2 avg=50.00 median=50.00 max=50
2025-11-01T10:00:03Z a1 avg=116.67 median=100.00 max=200
2025-11-01T10:00:04Z a2 avg=62.50 median=62.50 max=75
2025-11-01T10:00:06Z a3 avg=120.00 median=120.00 max=120
```
```

Note: Malformed lines in `sample.log` are skipped with reasons emitted to stderr, so they do not appear in stdout.

## Grade your results automatically

We provide a small grading script that diffs your stdout against the expected output for `--window 3`:

```bash
./grade.sh
```

It prints `PASS` if your output matches `expected_output_window3.txt`, or `FAIL` with a unified diff otherwise.

Tip: Ensure you have not left any debug `print()` statements in `parse_line_with_reason`, as they will pollute stdout and cause grading to fail.

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