# Advanced Exercise – Adaptive Agent Evaluation & Automation

**Goal**: Design and partially implement an automated evaluation and retraining loop for a multi-agent workflow inside Software Factory. Focus on measurement strategy, drift handling, and safety automation.

## Scenario
An agent pair collaborates: one agent drafts infrastructure change plans, another agent reviews and approves. Failures include unsafe infrastructure commands and regression of approval accuracy. You must create an offline evaluation harness plus automation to trigger retraining or rollback.

## Tasks
1. **Data Modeling**
   - Define a schema for agent interactions: prompts, intermediate actions, tool calls, approvals, human feedback, and outcome labels.
   - Implement ingestion code that loads historical interactions (mock via JSONL) and materializes features for evaluation.
2. **Evaluation Suite**
   - Implement metric calculators covering success rate, safety violation rate, latency, and disagreement between agents.
   - Add scenario-weighted scoring: some tickets count double depending on severity.
   - Provide confidence intervals via bootstrap or Bayesian estimation.
3. **Automation Hooks**
   - Build a policy function that, given rolling metrics, decides to `continue`, `rollback`, or `trigger_retraining`.
   - Persist decisions and underlying metrics to an audit log.
4. **Retraining Stub**
   - Sketch a retraining pipeline (pseudo-code acceptable) that fine-tunes the drafting agent using preference or supervised data while respecting safety guidelines.
5. **Reporting**
   - Produce a `PLAYBOOK.md` describing how the system runs daily, alarms, dashboards, and on-call response steps.

## Stretch Goals
- Integrate statistical drift detection (e.g., Kolmogorov–Smirnov on prompt features).
- Add counterfactual evaluation: simulate alternative policies and log expected gains.
- Containerize the pipeline or provide a `make` target for end-to-end execution.

## Deliverables
- Organize code in modules (`evaluation.py`, `policy.py`, `retrieval.py`, etc.) with type hints and docstrings.
- Include unit tests covering metric calculations and policy thresholds.
- Provide sample data under `data/sample_interactions.jsonl` to validate the pipeline.

## Reflection Prompts
- Which metrics would you promote to service-level indicators (SLIs) for Software Factory customers?
- How do you prevent evaluation gaming or reward hacking from the agents?
- What human-in-the-loop checkpoints remain necessary even after automation?

