"""
Exercise 2: Handling Imbalanced Classification

Build a classification system for highly imbalanced data with appropriate
sampling strategies, evaluation metrics, and probability calibration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Warning: imbalanced-learn not available. Some sampling methods will be limited.")


class ImbalancedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper for handling imbalanced classification.
    
    Applies sampling strategies, class weights, and probability calibration.
    """
    
    def __init__(self, 
                 base_estimator,
                 sampling_strategy: Optional[str] = None,  # 'smote', 'adasyn', 'undersample', 'oversample', None
                 class_weight: Optional[str] = 'balanced',  # 'balanced', dict, None
                 calibrate_probabilities: bool = True,
                 calibration_method: str = 'isotonic'):  # 'isotonic', 'sigmoid'
        """
        Initialize imbalanced classifier.
        
        Args:
            base_estimator: Base sklearn-compatible classifier
            sampling_strategy: Sampling method to apply
            class_weight: Class weight strategy
            calibrate_probabilities: Whether to calibrate probabilities
            calibration_method: Calibration method ('isotonic' or 'sigmoid')
        """
        self.base_estimator = base_estimator
        self.sampling_strategy = sampling_strategy
        self.class_weight = class_weight
        self.calibrate_probabilities = calibrate_probabilities
        self.calibration_method = calibration_method
        
        self.sampler_ = None
        self.estimator_ = None
        self.classes_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier with appropriate handling for imbalance.
        
        Args:
            X: Training features
            y: Training target
        
        Returns:
            self
        """
        # TODO: Implement fitting
        # 1. Apply sampling strategy if specified
        # 2. Set class weights on base estimator if specified
        # 3. Fit the base estimator
        # 4. Calibrate probabilities if requested
        # 5. Store classes
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        # TODO: Return predictions from fitted estimator
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probability estimates"""
        # TODO: Return probability predictions (calibrated if requested)
        pass
    
    def _apply_sampling(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply sampling strategy"""
        # TODO: Implement sampling
        # SMOTE: Use imblearn's SMOTE if available
        # ADASYN: Use imblearn's ADASYN if available
        # Undersampling: Random undersampling of majority class
        # Oversampling: Random oversampling of minority class
        # Return resampled X, y
        pass


class ImbalancedEvaluator:
    """
    Comprehensive evaluation for imbalanced classification.
    
    Computes appropriate metrics and generates visualizations.
    """
    
    def __init__(self, positive_class: int = 1):
        """
        Initialize evaluator.
        
        Args:
            positive_class: Label of positive/minority class
        """
        self.positive_class = positive_class
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary of metrics
        """
        # TODO: Implement metric computation
        # Compute: accuracy, balanced_accuracy, precision, recall, F1
        # If y_proba provided: ROC-AUC, PR-AUC
        # Return dictionary of metrics
        pass
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray,
                              metric: str = 'f1') -> float:
        """
        Find optimal decision threshold.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced')
        
        Returns:
            Optimal threshold value
        """
        # TODO: Implement threshold optimization
        # Try different thresholds
        # Compute metric for each threshold
        # Return threshold that maximizes metric
        pass
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             ax: Optional[plt.Axes] = None):
        """Plot confusion matrix"""
        # TODO: Create confusion matrix visualization
        pass
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   ax: Optional[plt.Axes] = None):
        """Plot precision-recall curve"""
        # TODO: Create PR curve visualization
        pass
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                       ax: Optional[plt.Axes] = None):
        """Plot ROC curve"""
        # TODO: Create ROC curve visualization
        pass
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       y_proba: Optional[np.ndarray] = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
        
        Returns:
            Formatted report string
        """
        # TODO: Generate human-readable report
        # Include all metrics, class distribution, recommendations
        pass


# Usage example
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Create highly imbalanced dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        random_state=42
    )
    X = pd.DataFrame(X)
    
    print(f"Class distribution:")
    print(f"  Class 0: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Class 1: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train baseline model (no handling)
    baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    
    print("\n=== Baseline Model (No Imbalance Handling) ===")
    baseline_eval = ImbalancedEvaluator()
    baseline_metrics = baseline_eval.evaluate(y_test, baseline_pred, baseline_proba)
    print(baseline_eval.generate_report(y_test, baseline_pred, baseline_proba))
    
    # Train with imbalance handling
    print("\n=== Model with Imbalance Handling ===")
    imbalanced_clf = ImbalancedClassifier(
        base_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
        sampling_strategy='smote' if IMBLEARN_AVAILABLE else None,
        class_weight='balanced',
        calibrate_probabilities=True
    )
    imbalanced_clf.fit(X_train, y_train)
    imbalanced_pred = imbalanced_clf.predict(X_test)
    imbalanced_proba = imbalanced_clf.predict_proba(X_test)[:, 1]
    
    imbalanced_metrics = baseline_eval.evaluate(y_test, imbalanced_pred, imbalanced_proba)
    print(baseline_eval.generate_report(y_test, imbalanced_pred, imbalanced_proba))
    
    # Find optimal threshold
    optimal_threshold = baseline_eval.find_optimal_threshold(y_test, imbalanced_proba, metric='f1')
    print(f"\nOptimal threshold (F1): {optimal_threshold:.4f}")
    
    # Predict with optimal threshold
    optimal_pred = (imbalanced_proba >= optimal_threshold).astype(int)
    optimal_metrics = baseline_eval.evaluate(y_test, optimal_pred, imbalanced_proba)
    print(f"\nMetrics with optimal threshold:")
    print(f"  Precision: {optimal_metrics.get('precision', 0):.4f}")
    print(f"  Recall: {optimal_metrics.get('recall', 0):.4f}")
    print(f"  F1: {optimal_metrics.get('f1', 0):.4f}")
