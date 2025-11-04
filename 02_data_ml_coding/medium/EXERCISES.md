# Data/ML Coding - Medium Exercises

These exercises focus on production ML concerns: model deployment, monitoring, A/B testing, and advanced pipelines.

---

## Exercise 1: Model Versioning and Registry

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Production ML infrastructure, model management

### Problem

Build a simple model registry system that can:
1. Store and version models
2. Track metadata (metrics, training date, features)
3. Promote models (dev → staging → production)
4. Rollback to previous versions

This is critical for the "Improvement Engineer" role which involves managing the ML lifecycle.

### Requirements

1. Create a `ModelRegistry` class with:
   - `register_model()` - Store model with metadata
   - `get_model()` - Retrieve model by version or stage
   - `promote_model()` - Move model between stages
   - `list_models()` - Query models by criteria

2. Track metadata:
   - Model version
   - Training timestamp
   - Performance metrics
   - Feature list
   - Model type and hyperparameters
   - Stage (dev, staging, production)

3. Support persistence (file-based or database)

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 2: Model Monitoring and Drift Detection

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Production ML monitoring, detecting model degradation

### Problem

Implement a monitoring system that detects:
1. **Data drift** - Distribution shift in input features
2. **Prediction drift** - Change in prediction distribution
3. **Concept drift** - Relationship between features and target changes

This is critical for an Improvement Engineer role that involves ensuring model reliability.

### Requirements

1. Create a `ModelMonitor` class that:
   - Tracks baseline statistics (from training data)
   - Compares current data to baseline
   - Detects drift using statistical tests
   - Generates alerts when drift detected

2. Implement drift detection methods:
   - Kolmogorov-Smirnov test for distributions
   - Chi-square test for categorical features
   - Monitoring prediction distributions

3. Create a simple dashboard/reporting system

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 3: A/B Testing Framework for ML Models

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Experimentation, statistical testing, production ML

### Problem

Build an A/B testing framework for comparing ML models in production. This is essential for an Improvement Engineer who needs to validate model improvements.

### Requirements

1. Create a framework that:
   - Splits traffic between models (A/B or A/B/C)
   - Tracks metrics for each variant
   - Performs statistical significance testing
   - Determines winner based on criteria

2. Support:
   - Different traffic splits (50/50, 80/20, etc.)
   - Multiple metrics (accuracy, F1, latency, etc.)
   - Statistical tests (t-test, Mann-Whitney, etc.)
   - Minimum sample size calculations

3. Generate reports with:
   - Metric comparisons
   - Statistical significance
   - Recommendation (which model to use)

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 4: ML Experiment Tracking System

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Experiment management, reproducibility, MLflow-like functionality

### Problem

Build a simple experiment tracking system similar to MLflow. Track:
- Hyperparameters
- Metrics
- Artifacts (models, plots)
- Code versions
- Environment info

### Requirements

1. Create an `ExperimentTracker` class with:
   - `start_run()` - Begin tracking an experiment
   - `log_params()` - Log hyperparameters
   - `log_metrics()` - Log metrics (supports multiple steps)
   - `log_artifact()` - Save files (models, plots)
   - `end_run()` - Complete the run

2. Support:
   - Multiple runs per experiment
   - Querying runs by criteria
   - Comparison of runs
   - Simple visualization

3. Persist to disk (JSON + file system)

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

