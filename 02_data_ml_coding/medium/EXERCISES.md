# Data/ML Coding - Medium Exercises

These exercises focus on practical ML challenges that require thoughtful problem-solving, feature engineering, and understanding of ML concepts. They are designed to prepare you for interviews where AI assistance is allowed, emphasizing code quality, design rationale, and technical depth.

---

## Exercise 1: Feature Engineering & Selection Pipeline

**Difficulty:** Medium  
**Time Limit:** 60-75 minutes (with AI assistance allowed)  
**Focus:** Feature engineering, feature selection, pipeline design, code quality

### Problem

Build a comprehensive feature engineering and selection pipeline for a classification problem. You'll work with messy, real-world-like data that requires:

1. **Handling missing values** intelligently
2. **Creating derived features** from existing ones
3. **Encoding categorical variables** appropriately
4. **Selecting the most important features** to avoid overfitting
5. **Designing a reusable pipeline** that can be applied to new data

This challenge tests your ability to think critically about data preprocessing and feature engineering decisions.

### Requirements

1. Create a `FeatureEngineeringPipeline` class that:
   - Handles missing values with multiple strategies (mean/median/mode, forward fill, etc.)
   - Creates interaction features (e.g., product, ratio, difference of numeric features)
   - Encodes categorical features (one-hot, target encoding, frequency encoding)
   - Scales/normalizes features appropriately
   - Can be fit on training data and transform new data

2. Create a `FeatureSelector` class that:
   - Implements multiple selection methods (correlation-based, mutual information, recursive feature elimination)
   - Can select top-k features based on importance scores
   - Provides feature importance rankings
   - Works with the pipeline seamlessly

3. Design considerations:
   - **Avoid data leakage**: Ensure feature engineering doesn't leak target information
   - **Handle unseen categories**: What happens when new data has categories not seen in training?
   - **Memory efficiency**: Can your pipeline handle large datasets?
   - **Reproducibility**: Ensure transformations are deterministic

### Hints (Don't peek too early!)

<details>
<summary>Hint 1: Missing Value Strategy</summary>
Consider different strategies for different types of features. For example, time-series features might benefit from forward-fill, while numeric features might use median imputation. Think about when to use mode vs. mean vs. median.
</details>

<details>
<summary>Hint 2: Feature Interactions</summary>
Creating interaction features can be powerful but also lead to feature explosion. Consider which interactions are most likely to be meaningful (e.g., ratios of related features, products of features that might interact).
</details>

<details>
<summary>Hint 3: Categorical Encoding</summary>
One-hot encoding is simple but can create high dimensionality. Target encoding can be powerful but requires careful cross-validation to avoid overfitting. Consider when each approach is appropriate.
</details>

<details>
<summary>Hint 4: Feature Selection</summary>
Mutual information captures non-linear relationships better than correlation. Recursive feature elimination can be computationally expensive but often finds better feature subsets. Consider combining multiple methods.
</details>

### Key Learning Points

- Understanding when and how to engineer features
- Avoiding common pitfalls (data leakage, overfitting)
- Designing reusable, production-ready pipelines
- Making informed trade-offs between model complexity and performance

### Design Considerations to Explain

- Why did you choose specific imputation strategies?
- How do you prevent data leakage in feature engineering?
- What's your rationale for feature selection thresholds?
- How would you handle production data that differs from training data?

---

## Exercise 2: Handling Imbalanced Classification

**Difficulty:** Medium  
**Time Limit:** 60-75 minutes (with AI assistance allowed)  
**Focus:** Imbalanced data, sampling strategies, evaluation metrics, model calibration

### Problem

Build a classification system for highly imbalanced data (e.g., fraud detection, rare disease diagnosis). The challenge is to:

1. **Handle severe class imbalance** (e.g., 1:100 or worse)
2. **Choose appropriate evaluation metrics** (accuracy is misleading!)
3. **Implement sampling strategies** (SMOTE, undersampling, class weights)
4. **Calibrate probability predictions** for reliable confidence scores
5. **Design a robust evaluation framework** that reflects real-world performance

This challenge tests your understanding of when standard ML practices fail and how to adapt.

### Requirements

1. Create an `ImbalancedClassifier` wrapper that:
   - Accepts any sklearn-compatible classifier
   - Applies sampling strategies (SMOTE, ADASYN, random undersampling)
   - Handles class weights appropriately
   - Calibrates probability predictions using Platt scaling or isotonic regression

2. Create an `ImbalancedEvaluator` class that:
   - Computes appropriate metrics (precision-recall curve, F1, balanced accuracy, PR-AUC)
   - Generates visualizations (confusion matrix, ROC curve, PR curve)
   - Provides detailed classification reports
   - Handles multiple evaluation scenarios (cost-sensitive, threshold optimization)

3. Design considerations:
   - **Metric selection**: Why is accuracy insufficient? What metrics matter for imbalanced data?
   - **Sampling trade-offs**: When does oversampling help vs. hurt?
   - **Threshold tuning**: How do you find optimal decision thresholds?
   - **Production considerations**: How do sampling strategies affect inference?

### Hints (Don't peek too early!)

<details>
<summary>Hint 1: Evaluation Metrics</summary>
For imbalanced data, focus on metrics that don't depend on class distribution: precision, recall, F1-score, PR-AUC, and balanced accuracy. ROC-AUC can be misleading when classes are severely imbalanced.
</details>

<details>
<summary>Hint 2: Sampling Strategies</summary>
SMOTE creates synthetic samples in feature space, but can create unrealistic samples if features are discrete or have specific constraints. Consider when to use SMOTE vs. ADASYN vs. simple oversampling.
</details>

<details>
<summary>Hint 3: Class Weights</summary>
Using class weights in the loss function is often simpler than sampling and doesn't require modifying the dataset. However, it may not be as effective for extreme imbalances. Consider combining both approaches.
</details>

<details>
<summary>Hint 4: Probability Calibration</summary>
Many models (especially tree-based) produce poorly calibrated probabilities. Calibration is crucial when you need reliable confidence scores. Platt scaling works well for most cases, isotonic regression for more complex patterns.
</details>

### Key Learning Points

- Understanding why standard metrics fail for imbalanced data
- Knowing when and how to apply sampling techniques
- Importance of probability calibration
- Designing evaluation frameworks that reflect business needs

### Design Considerations to Explain

- Why did you choose specific metrics over others?
- How do you decide between sampling strategies?
- What's the trade-off between precision and recall in your domain?
- How would you deploy this model in production?

---

## Exercise 3: Time Series Forecasting Pipeline

**Difficulty:** Medium  
**Time Limit:** 60-75 minutes (with AI assistance allowed)  
**Focus:** Time series analysis, feature engineering, model selection, evaluation

### Problem

Build a time series forecasting pipeline that can handle:

1. **Multiple time series** with different characteristics
2. **Missing values and outliers** in time series data
3. **Feature engineering** for time series (lag features, rolling statistics, seasonality)
4. **Multiple forecasting models** (ARIMA, exponential smoothing, ML-based)
5. **Proper evaluation** using time-aware cross-validation
6. **Ensemble predictions** from multiple models

This challenge tests your understanding of temporal dependencies and time series-specific challenges.

### Requirements

1. Create a `TimeSeriesPreprocessor` class that:
   - Handles missing values (interpolation, forward fill, etc.)
   - Detects and handles outliers
   - Creates time-based features (day of week, month, holidays, etc.)
   - Creates lag features and rolling window statistics
   - Handles multiple time series with different frequencies

2. Create a `TimeSeriesForecaster` class that:
   - Implements or wraps multiple forecasting methods (ARIMA, Exponential Smoothing, Prophet, ML models)
   - Supports univariate and multivariate forecasting
   - Handles seasonality detection and modeling
   - Provides confidence intervals for predictions

3. Create a `TimeSeriesEvaluator` class that:
   - Implements time-aware cross-validation (no future data leakage!)
   - Computes appropriate metrics (MAE, RMSE, MAPE, MASE)
   - Handles multiple forecast horizons
   - Compares model performance across different time periods

4. Design considerations:
   - **Temporal dependencies**: How do you prevent data leakage in time series?
   - **Stationarity**: When and how do you handle non-stationary series?
   - **Seasonality**: How do you detect and model seasonal patterns?
   - **Multiple horizons**: How do predictions change for short vs. long horizons?

### Hints (Don't peek too early!)

<details>
<summary>Hint 1: Time-Aware Cross-Validation</summary>
Standard k-fold CV leaks future information! Use walk-forward validation or time series cross-validation where you always train on past data and test on future data. Each fold should be chronologically ordered.
</details>

<details>
<summary>Hint 2: Feature Engineering</summary>
Lag features (y_{t-1}, y_{t-7}, etc.) are crucial. Rolling statistics (mean, std over windows) capture trends. Time-based features (day of week, month) capture seasonality. Consider interactions between these.
</details>

<details>
<summary>Hint 3: Stationarity</summary>
Many time series models assume stationarity. Use differencing or transformations (log, Box-Cox) to achieve stationarity. ADF test can check for stationarity, but visual inspection is also valuable.
</details>

<details>
<summary>Hint 4: Model Selection</summary>
ARIMA is great for univariate series with clear patterns. Exponential smoothing handles trends and seasonality well. ML models (XGBoost, LSTM) can capture complex patterns but need careful feature engineering. Consider ensemble methods.
</details>

### Key Learning Points

- Understanding temporal dependencies and data leakage
- Feature engineering for time series
- Choosing appropriate forecasting models
- Proper evaluation of time series models

### Design Considerations to Explain

- How do you prevent data leakage in time series?
- When would you use ARIMA vs. ML models?
- How do you handle multiple forecast horizons?
- What's your strategy for handling non-stationary series?

---

## Exercise 4: Model Interpretability & Explainability

**Difficulty:** Medium  
**Time Limit:** 60-75 minutes (with AI assistance allowed)  
**Focus:** Model interpretability, SHAP values, feature importance, explainable AI

### Problem

Build a comprehensive model interpretability system that can:

1. **Explain individual predictions** (local interpretability)
2. **Explain overall model behavior** (global interpretability)
3. **Handle different model types** (linear, tree-based, neural networks)
4. **Provide actionable insights** for stakeholders
5. **Detect potential model issues** (bias, feature interactions)

This challenge tests your ability to make ML models transparent and trustworthy, which is crucial for production systems.

### Requirements

1. Create a `ModelInterpreter` class that:
   - Computes feature importance (permutation importance, SHAP values)
   - Generates local explanations for individual predictions
   - Creates global explanations (feature importance plots, partial dependence plots)
   - Handles different model types appropriately
   - Detects feature interactions and dependencies

2. Create an `ExplanationVisualizer` class that:
   - Generates SHAP summary plots
   - Creates waterfall plots for individual predictions
   - Visualizes partial dependence plots
   - Creates feature interaction plots
   - Generates human-readable explanations

3. Create a `ModelAuditor` class that:
   - Detects potential bias (demographic parity, equalized odds)
   - Identifies problematic feature interactions
   - Flags predictions with high uncertainty
   - Validates model behavior against domain knowledge

4. Design considerations:
   - **Computational efficiency**: SHAP can be slow for large datasets
   - **Model-agnostic vs. model-specific**: When to use each approach?
   - **Actionability**: How do you make explanations useful for decision-makers?
   - **Trust**: How do you validate that explanations are correct?

### Hints (Don't peek too early!)

<details>
<summary>Hint 1: SHAP Values</summary>
SHAP (SHapley Additive exPlanations) provides unified framework for model interpretability. TreeSHAP is fast for tree models, KernelSHAP is model-agnostic but slower. Consider sampling strategies for large datasets.
</details>

<details>
<summary>Hint 2: Feature Importance</summary>
Permutation importance measures how much performance degrades when a feature is shuffled. It's model-agnostic but can be slow. Tree-based models have built-in importance, but it can be biased toward high-cardinality features.
</details>

<details>
<summary>Hint 3: Partial Dependence</summary>
Partial dependence plots show how a feature affects predictions while averaging over other features. They reveal non-linear relationships but can be misleading if features are correlated.
</details>

<details>
<summary>Hint 4: Local Explanations</summary>
For individual predictions, SHAP values show how each feature contributed. LIME is an alternative but less theoretically grounded. Consider which features to highlight and how to present them clearly.
</details>

### Key Learning Points

- Understanding different interpretability methods
- Knowing when to use model-specific vs. model-agnostic approaches
- Making explanations actionable for stakeholders
- Detecting and addressing model bias

### Design Considerations to Explain

- Why did you choose SHAP over LIME or other methods?
- How do you handle computational complexity for large models?
- How do you validate that explanations are correct?
- How would you present explanations to non-technical stakeholders?

---

## General Guidelines

### Code Quality Expectations

- **Clean, readable code** with proper structure and naming
- **Comprehensive documentation** (docstrings, comments where needed)
- **Error handling** for edge cases
- **Type hints** for better code clarity
- **Modular design** that allows for extension

### Design Rationale

Be prepared to explain:
- **Why** you made specific design choices
- **Trade-offs** you considered
- **Alternatives** you evaluated
- **Production considerations** (scalability, maintainability, etc.)

### Testing Your Solutions

Each exercise includes a `test_all.py` script that will validate your implementation. However, the goal isn't just to pass tests—focus on:

1. **Code quality** and maintainability
2. **Understanding** the underlying concepts
3. **Explaining** your design decisions
4. **Considering** production deployment scenarios

### Using AI Assistance

Since AI assistance is allowed in your interview:
- Use AI tools to help with implementation details
- But **think critically** about the solutions AI suggests
- **Understand** what the code does, don't just copy-paste
- **Explain** your design choices and rationale
- **Consider** edge cases and production concerns

Good luck! 🚀
