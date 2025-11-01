# Intermediate Exercise – Retrieval-Augmented Resolution Assistant

**Goal**: Prototype an evaluation harness for an LLM agent that suggests fixes for regression tickets. Focus on reproducibility, guardrails, and measurement.

## Scenario
Software Factory ships weekly updates. Before rollout, an internal agent reads regression reports (markdown snippets) and proposes remediation steps. You need to evaluate the agent’s suggestions against historical resolutions to estimate usefulness and risk.

## Dataset
- Use the `bug_report` subset of Hugging Face `ai2_arc` or any open bug/ticket dataset.
- Create a derived dataset with columns: `ticket_id`, `summary`, `details`, `resolution`.
- Construct a knowledge base by chunking the `details` field and indexing with FAISS or a lightweight BM25 implementation.

## Tasks
1. Build a retrieval layer that, given a ticket, returns top-k relevant chunks.
2. Use an open-source instruction-tuned model (e.g., `mistralai/Mistral-7B-Instruct`, `meta-llama/Meta-Llama-3-8B-Instruct`, or a local smaller model via `transformers`) to draft remediation steps conditioned on retrieved context.
3. Implement automatic evaluations:
   - Similarity to ground-truth resolutions (BLEU, ROUGE-L, semantic similarity via sentence transformers).
   - Safety heuristics: flag if the suggestion contains words from a configurable deny-list.
   - Latency tracking per agent call.
4. Log every run (config, metrics, artifacts) to an experiment tracker. Persist raw generations for human audit.
5. Produce a runbook-style `REPORT.md` capturing methodology, results, limitations, and rollout recommendation.

## Stretch Goals
- Add a lightweight human-in-the-loop review simulation: randomly sample predictions and log qualitative annotations.
- Integrate guardrails (e.g., OpenAI Evals, Guardrails AI, or custom regex) to enforce format contracts.

## Deliverables
- Place orchestration code in `pipeline.py` (or a notebook) plus helper modules as needed.
- Include configuration files (YAML/JSON) to run the experiment with different model checkpoints.
- Commit a `tests/` directory with at least one unit or smoke test covering retrieval and evaluation.

## Reflection Prompts
- How do you ensure deterministic replays of an experiment when model outputs are stochastic?
- Which online signals (feedback, automatic grading) would you monitor after deployment?
- How can you sandbox the agent to avoid destructive recommendations while still iterating quickly?

