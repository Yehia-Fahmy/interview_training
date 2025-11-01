# 6-Week Preparation Roadmap

The timeline below scaffolds your preparation around the three interview pillars. Adjust pacing to match your schedule—each week lists the minimum deliverables and pointers to exercises inside this repository.

## Week 0 – Orientation & Baseline
- **Logistics**: Install the Zoom desktop client, test remote-control permissions, and verify camera/mic and network stability.
- **Assessment**: Complete the `code_challenge/easy` prompt under realistic timing (~45 min) to benchmark comfort with Python fundamentals.
- **Reflection**: Capture strengths/gaps in a personal journal; translate gaps into goals for Weeks 1–4.

## Week 1 – Python Systems Thinking
- **Focus**: Data structures, complexity analysis, and memory-aware optimizations.
- **Exercises**:
  - Finish `code_challenge/easy/starter.py` and attempt the stretch goal noted in the prompt.
  - Start `code_challenge/medium` and profile your solution using `cProfile` or `time.perf_counter`.
- **Checkpoint**: Summarize the micro-optimizations you applied (e.g., pre-allocation, avoiding temporaries) and why they matter for agent infrastructure.

## Week 2 – Performance & Tooling
- **Focus**: Low-level reasoning, streaming data, and diagnostics.
- **Exercises**:
  - Complete `code_challenge/medium` under timed conditions.
  - Implement at least one alternative approach and compare runtime/memory charts.
- **Stretch**: Skim `resources.md` section on CPython internals or PyPy to deepen understanding of interpreter behaviors.
- **Checkpoint**: Draft a mock retrospective that you could share with an interviewer about trade-offs you considered.

## Week 3 – ML Experimentation Fundamentals
- **Focus**: Exploratory data analysis, baseline modeling, and evaluation discipline.
- **Exercises**:
  - Run the workflow in `data_ml_coding/foundational`; emphasize reproducible notebooks/scripts.
  - Track metrics in a simple experiment log (CSV, MLflow, or Weighted Logging).
- **Checkpoint**: Prepare a 5-minute explanation of dataset assumptions, metric choice, and failure modes.

## Week 4 – Production-Ready ML & LLM Ops
- **Focus**: Iterative modeling, drift monitoring, prompt/tool evaluation.
- **Exercises**:
  - Attempt `data_ml_coding/intermediate` with the suggested open dataset.
  - Implement the provided evaluation harness; capture unit tests or assertions around data contracts.
  - Optional: Explore the LLM evaluation framework reference in `resources.md`.
- **Checkpoint**: Write a README snippet describing how you would productionize and monitor your solution inside Software Factory.

## Week 5 – Advanced ML Systems & Automation
- **Focus**: Agent evaluation, reinforcement learning from feedback, automation of retraining loops.
- **Exercises**:
  - Tackle `data_ml_coding/advanced` focusing on measurement strategy (success, safety, latency).
  - Pair it with the `system_design/observability` prompt to reason about architecture.
- **Checkpoint**: Conduct a mock design review with a friend or by recording yourself; critique clarity and depth.

## Week 6 – System Design Mastery & Full Rehearsal
- **Focus**: End-to-end architecture storytelling, SLOs, and operational playbooks.
- **Exercises**:
  - Work through two `system_design` prompts end-to-end, producing diagrams plus written trade-off analysis.
  - Re-run `code_challenge/hard` and `data_ml_coding/intermediate` back-to-back to simulate interview fatigue.
- **Checkpoint**: Assemble a final summary of lessons learned, remaining risks, and targeted questions for your interviewers.

## Post-Roadmap Maintenance
- Rotate through one exercise from each pillar every 7–10 days to keep skills sharp.
- Keep annotating new insights, libraries, or patterns in your personal notes; iteratively update practice prompts as you discover new weak spots.

You can compress or extend this schedule—treat it as a template and modify the per-week workload to keep yourself challenged but not overwhelmed.

