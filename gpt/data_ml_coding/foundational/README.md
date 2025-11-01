# Foundational Exercise – Customer Ticket Triage

**Goal**: Build a lightweight classifier that routes support tickets from Software Factory customers into latency, hallucination, or integration queues. Emphasize data understanding and reproducibility.

## Dataset
- Use the Kaggle “Customer Support on Twitter” dataset or Hugging Face `tweet_eval` (sentiment subset). Treat labels `negative`, `neutral`, `positive` as proxies for severity buckets.
- If you cannot access the datasets, stub synthetic data with CSV generation but document assumptions explicitly.

## Tasks
1. Create a notebook or script that:
   - Loads the data reproducibly (fixed random seeds, environment capture).
   - Performs minimal text cleaning (lowercasing, punctuation stripping, optional tokenization).
   - Splits into train/validation/test with stratification.
2. Train a baseline model (logistic regression or linear SVM) using scikit-learn.
3. Evaluate with macro F1 and per-class precision/recall; surface a confusion matrix.
4. Log metrics and parameters to a lightweight experiment tracker (MLflow local, Weights & Biases, or an append-only CSV).
5. Draft a short “Ship-It” note explaining the model’s role, limitations, and immediate next steps.

## Deliverables
- Place code in `notebook.ipynb` or `pipeline.py` (your choice) plus a `REPORT.md` summarizing results.
- Ensure rerunning end-to-end produces the same tables/metrics.

## Reflection Prompts
- How would you detect if ticket distribution drifts post-deployment?
- What guardrails keep false negatives (high-severity tickets mis-labeled) below 1%?
- How does this align with the Improvement Engineer’s focus on measurement and reliability?

