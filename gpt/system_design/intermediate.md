# Intermediate Prompt – Evaluation Service for Agent Releases

**Scenario**: Before shipping a new agent release, 8090.ai runs regression suites comparing the candidate against the current production agent. Design an evaluation service that orchestrates replay experiments, stores results, and publishes scorecards to stakeholders.

## Requirements
- Run replay experiments on demand or on schedule across historical datasets (tens of millions of interactions).
- Support plug-ins for varied evaluation types: deterministic unit checks, stochastic generative scoring, safety audits.
- Produce signed report artifacts and surface them through a dashboard.
- Gate promotion: only allow deployment if metrics stay within guardrail thresholds.

## Discussion Topics
- Workload orchestration (Kubernetes jobs, Ray, Airflow) and resource isolation.
- Artifact management (model checkpoints, prompts, datasets) with lineage tracking.
- Result storage and querying (OLAP warehouse, document store, vector DB for embeddings).
- Confidence interval computation and statistical significance testing.
- Notifying downstream systems (deployment pipeline, Slack, PagerDuty) with actionable summaries.
- Cost controls and strategies to parallelize while respecting budgets.

## Deliverable
- End-to-end architecture highlighting control plane vs. data plane, plus mitigation strategies when evaluations fail, flake, or exceed SLA.

