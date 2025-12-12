# Challenge: App Deletion Risk Prediction

## Problem Statement

You are a machine learning engineer at a mobile app company experiencing high user churn. The product team wants to proactively identify users who are likely to delete the app within the next **n weeks** so they can trigger retention campaigns (push notifications, emails, in-app offers) before users churn.

Your task is to build a **binary classification system** that predicts the probability of a user deleting the app. The output should be a probability score that can be used to prioritize intervention efforts.

---

## Business Context

- **Goal**: Reduce app uninstalls by identifying at-risk users early
- **Intervention**: Users with high deletion probability receive targeted retention campaigns
- **Constraint**: Limited budget for interventions, so precision matters
- **Success Metric**: Improve retention rate while minimizing false positives (annoying loyal users)

---

## Dataset Schema

The dataset (`data.csv`) contains user snapshots with the following features:

| Column | Feature Name | Type | Description |
|--------|--------------|------|-------------|
| F1 | country | categorical | User's country (US, UK, BR, IN, AR, etc.) |
| F2 | gender | categorical | M (Male), F (Female), U (Unknown) |
| F3 | age | numeric | User age (17-90) |
| F4 | download_channel | categorical | E (Email), P (Push), O (Organic), PA (Paid) |
| F5 | interactions_7d | numeric | Number of app interactions in last 7 days |
| F6 | interactions_14d | numeric | Number of app interactions in last 14 days |
| F7 | interactions_30d | numeric | Number of app interactions in last 30 days |
| F8 | daily_time_spent | numeric | Average daily time spent on app (minutes) |
| F9 | time_since_last_use | numeric | Days since last app use |
| F10 | time_of_day | numeric | Typical usage hour (0-23) |
| F11 | day_of_week | numeric | Typical usage day (0-6, Monday=0) |
| F12 | platform | categorical | I (iOS), A (Android) |
| F13 | app_version | numeric | App version number |
| F14 | os_version | numeric | OS version number |
| Label | label | binary | 1 = User deletes app, 0 = User retains app |

### Data Characteristics

- **Size**: ~10,000-20,000 rows
- **Class Imbalance**: Approximately 20% positive class (deletions)
- **Missing Values**: Present in several columns (real-world scenario)
  - Some users have missing gender, age, or download channel information
  - Missing values may be encoded as empty strings or NaN
- **Feature Correlations**: Behavioral features (interactions, time spent) are correlated with deletion probability

---

## Requirements

### 1. Data Exploration & Preprocessing
- Load and explore the dataset
- Handle missing values appropriately (document your strategy)
- Encode categorical variables
- Normalize/standardize numeric features if needed
- Address class imbalance (if you choose to)

### 2. Feature Engineering (Optional)
- Create derived features (e.g., interaction ratios, engagement scores)
- Consider temporal patterns

### 3. Model Development
- Implement at least one classification model
- Split data appropriately (train/validation/test)
- Tune hyperparameters
- Handle class imbalance in your approach

### 4. Evaluation
- Report appropriate metrics for imbalanced classification:
  - ROC-AUC
  - Precision-Recall AUC
  - F1-Score
  - Confusion Matrix
- Justify your choice of decision threshold
- Discuss precision/recall tradeoffs for this business problem

### 5. Deliverables
- Clean, documented code
- Model training script
- Brief analysis of results and feature importance
- Recommendations for deployment

---

## Evaluation Rubric

| Dimension | Weight | What We Look For |
|-----------|--------|------------------|
| Data Handling | 25% | Proper handling of missing values, encoding, and preprocessing |
| Modeling | 35% | Appropriate model choice, handling of imbalance, good performance |
| Evaluation | 20% | Correct metrics, threshold selection, business-aware analysis |
| Code Quality | 20% | Clean, readable, well-documented code |

---

## Stretch Goals

- Implement multiple models and compare performance
- Build a feature importance analysis (SHAP, permutation importance)
- Create a simple prediction API endpoint
- Discuss how you would monitor this model in production
- Propose an A/B testing strategy for the intervention campaigns

---

## Getting Started

1. Load the dataset from `data.csv`
2. Explore the data and understand the distributions
3. Implement your preprocessing pipeline
4. Train and evaluate your model
5. Document your findings and decisions

Good luck!
