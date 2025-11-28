# ML Concepts Reference

Key concepts for the Data/ML Coding interview.

## Fundamental ML Algorithms

### K-Nearest Neighbors (KNN)
- **Type**: Supervised learning (classification/regression)
- **How it works**: Predicts based on k nearest training examples
- **Distance metric**: Euclidean distance (squared)
- **Implementation**: See `knn.py` and `test_knn.py` in repository root
- **Key characteristics**:
  - Lazy learning (no explicit training phase)
  - Non-parametric
  - Sensitive to k value and distance metric

### K-Means Clustering
- **Type**: Unsupervised learning (clustering)
- **How it works**: Partitions data into k clusters by minimizing within-cluster variance
- **Algorithm**: Iterative optimization of centroids
- **Implementation**: See `k_means.py` and `test_kmeans.py` in repository root
- **Key characteristics**:
  - Requires specifying k (number of clusters)
  - Sensitive to initialization
  - Converges to local optimum
  - Metrics: Within-Cluster Sum of Squares (WCSS)

## Model Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Cross-validation**: K-fold, stratified
- **Holdout sets**: Train/validation/test splits

## Production ML

- **Model Lifecycle**: Training → Validation → Deployment → Monitoring
- **Model Versioning**: Track versions, rollback capability
- **A/B Testing**: Statistical significance, sample sizes
- **Monitoring**: Drift detection, performance tracking

## LLM Evaluation (8090 Focus)

- **BLEU**: N-gram overlap
- **ROUGE**: Recall-oriented metrics
- **Perplexity**: Language model quality
- **Semantic Similarity**: Embedding-based evaluation
- **Task-specific**: Classification accuracy, etc.

## Feature Engineering

- **Missing Values**: Imputation strategies
- **Categorical Encoding**: One-hot, label encoding, embeddings
- **Scaling**: Standardization, normalization
- **Feature Selection**: Correlation, importance-based

## MLOps

- **Experiment Tracking**: MLflow, Weights & Biases
- **Model Registry**: Centralized model storage
- **Feature Stores**: Online and offline features
- **Model Serving**: APIs, batching, optimization

