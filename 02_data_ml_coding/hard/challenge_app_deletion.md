# Challenge: App Deletion Risk Prediction

## Scenario
You are part of the lifecycle analytics team for a consumer mobile app. Growth has stalled because too many users uninstall the app within a week of installing it. Product managers want a reliable deletion-risk score that can be refreshed daily so they can trigger retention campaigns before users churn. You are given a thin behavioral snapshot per user that mixes demographics, recent interaction stats, contextual usage patterns, and high-level app metadata. Your job is to build a model that predicts whether a user will delete the app within the next 7 days (`label = 1`).

## Dataset schema
Each row corresponds to one active user snapshot captured at midnight. All numeric features are either counts or normalized averages. Missing categorical values are encoded with `"U"` (unknown). Example columns:

| Feature block | Columns | Type | Notes |
| --- | --- | --- | --- |
| Identity | `user_id` | string | Unique but should only be used for grouping/splitting. |
| Demographics | `country`, `gender`, `age` | categorical, categorical, numeric | Age is capped at 90. |
| Acquisition | `download_channel` | categorical (`email`, `push`, `organic`, `paid`) | |
| Behavior | `interactions_7d`, `interactions_14d`, `interactions_30d`, `daily_time_spent`, `time_since_last_use` | numeric | Interaction counts are non-decreasing windows. |
| Context | `time_of_day`, `day_of_week`, `platform`, `os_version` | numeric (0–23), numeric (0–6), categorical (`ios`,`android`), categorical | |
| App | `app_version` | categorical | Semantic version, keep as string. |
| Target | `label` | binary | 1 if user deletes app within 7 days, else 0. |

### Additional notes
- Assume the dataset ships as `data.csv` (10–50K rows). Expect slight class imbalance (≈20% positive).
- Time-series leakage is minimal because the snapshot already aggregates past activity. Avoid grouping future data when you create folds—split by `user_id` to keep duplicates in the same set.

## Goal
Design and train a PyTorch-based churn detector that outputs a probability of deletion for each user snapshot. You are responsible for every stage of the workflow: deciding how to split the data, how to encode the heterogeneous features, what model family to use, and how to monitor its performance over time. Optimize for both discrimination (ROC-AUC) and operational usefulness (F1 at a threshold you justify).

## Requirements
1. **Data handling**
   - Define your own reproducible train/validation/test strategy that respects `user_id` groupings.
   - Document feature engineering choices, normalization strategies, and how you handle missing or rare values.
2. **Modeling**
   - Build the full PyTorch training stack from scratch (dataset class, dataloaders, model definition, optimizer, scheduler, etc.).
   - Justify the architecture and regularization you pick. Feel free to experiment beyond simple MLPs if you think the data warrants it.
3. **Evaluation**
   - Track ROC-AUC, PR-AUC, F1, and any calibration metric of your choice. Explain how you pick the decision threshold.
   - Provide diagnostics such as confusion matrices or PR curves to show trade-offs.
4. **Deliverables**
   - A runnable training script or notebook plus clear instructions in a README.
   - Saved model weights and any preprocessing assets required for inference.
   - Brief discussion of limitations and next steps.

## Stretch goals
- Incorporate temporal decay features (e.g., weighted interaction sums) without additional raw data.
- Build a small feature importance dashboard (permutation or SHAP).
- Package the model in a lightweight FastAPI service exposing `/score` and `/health`.
- Add automated evaluation on the provided baseline splits via GitHub Actions.

## Evaluation rubric
| Dimension | Weight | What we look for |
| --- | --- | --- |
| Data hygiene | 25% | Leak-free splits, consistent preprocessing, thorough EDA. |
| Modeling quality | 35% | Architecture choices, handling of imbalance, metrics > baseline. |
| Experiment discipline | 20% | Reproducible configs, logging, checkpoints, clarity. |
| Communication | 20% | Clear README, thoughtful discussion of trade-offs and next steps. |

Document assumptions. If you deviate (e.g., choose a different metric), explain why and show evidence.

