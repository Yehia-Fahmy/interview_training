# Code Challenge – Medium

**Theme**: Dependency-aware scheduling, caching, and profiling.

You receive telemetry describing agent actions within Software Factory. Each record captures a task identifier, its prerequisites, and execution cost in microseconds. You must compute critical-path metrics while handling up to 10⁵ tasks without exponential blowups.

## Requirements
1. Read JSON lines from stdin. Each line matches the schema:
   ```json
   {
     "task_id": "agent.step.123",
     "dependencies": ["agent.step.45", "agent.step.99"],
     "cost_us": 8712
   }
   ```
2. For each task, emit a JSON line with:
   - `task_id`
   - `total_cost_us`: cumulative cost of the task plus all transitive dependencies.
   - `critical_path`: list of task IDs forming one maximal-cost path ending at this task.
3. Detect cycles. If a cycle exists, emit a single warning to stderr naming the cycle and skip emitting metrics for affected nodes.
4. Ensure repeated dependencies are traversed only once—memoize intermediate results.

## Stretch Goals
- Add a `--profile` CLI switch that prints the top 5 tasks by time spent in your algorithm using `cProfile`.
- Support an optional `--max-parallelism` flag that, when set, outputs a naive parallel runtime estimate assuming up to `p` tasks can execute simultaneously when dependencies are satisfied.

## Constraints & Expectations
- Target completion window: 60–75 minutes including light testing.
- Avoid recursion limits—prefer iterative topological ordering or explicit stacks.
- Handle missing dependency records gracefully (assume missing tasks have zero cost but keep them in the critical path output).

## Deliverables
- Implement logic in `starter.py`.
- Document runtime/memory analysis plus profiling takeaways in `NOTES.md`.
- Prepare to defend how the memoization strategy scales when agent graphs change rapidly in production.

