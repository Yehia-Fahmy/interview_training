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
        cm = confusion_matrix(y_true, y_pred)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Get class labels
        classes = np.unique(np.concatenate([y_true, y_pred]))
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       horizontalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        
        return ax
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   ax: Optional[plt.Axes] = None):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avg_precision:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return ax
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                       ax: Optional[plt.Axes] = None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return ax
    
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
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("IMBALANCED CLASSIFICATION EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Class distribution
        unique_classes, counts = np.unique(y_true, return_counts=True)
        total = len(y_true)
        report_lines.append("CLASS DISTRIBUTION:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total) * 100
            report_lines.append(f"  Class {cls}: {count} ({percentage:.2f}%)")
        report_lines.append("")
        
        # Compute metrics
        metrics = self.evaluate(y_true, y_pred, y_proba)
        
        # Basic metrics
        report_lines.append("CLASSIFICATION METRICS:")
        report_lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        if 'balanced_accuracy' in metrics:
            report_lines.append(f"  Balanced Accuracy: {metrics.get('balanced_accuracy', 0):.4f}")
        report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
        report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
        report_lines.append(f"  F1-Score: {metrics.get('f1', 0):.4f}")
        report_lines.append("")
        
        # Probability-based metrics
        if y_proba is not None:
            report_lines.append("PROBABILITY-BASED METRICS:")
            if 'roc_auc' in metrics:
                report_lines.append(f"  ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
            if 'pr_auc' in metrics:
                report_lines.append(f"  PR-AUC: {metrics.get('pr_auc', 0):.4f}")
            report_lines.append("")
        
        # Confusion matrix summary
        cm = confusion_matrix(y_true, y_pred)
        report_lines.append("CONFUSION MATRIX:")
        report_lines.append(f"                Predicted")
        report_lines.append(f"                0      1")
        report_lines.append(f"  Actual  0    {cm[0,0]:5d}  {cm[0,1]:5d}")
        report_lines.append(f"          1    {cm[1,0]:5d}  {cm[1,1]:5d}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if metrics.get('recall', 0) < 0.5:
            report_lines.append("  ⚠ Low recall detected. Consider:")
            report_lines.append("    - Using sampling strategies (SMOTE, ADASYN)")
            report_lines.append("    - Adjusting class weights")
            report_lines.append("    - Lowering decision threshold")
        
        if metrics.get('precision', 0) < 0.5:
            report_lines.append("  ⚠ Low precision detected. Consider:")
            report_lines.append("    - Increasing decision threshold")
            report_lines.append("    - Feature engineering to improve separability")
        
        if y_proba is not None and metrics.get('roc_auc', 0) < 0.7:
            report_lines.append("  ⚠ Low ROC-AUC detected. Consider:")
            report_lines.append("    - Feature selection or engineering")
            report_lines.append("    - Trying different algorithms")
            report_lines.append("    - Collecting more training data")
        
        if metrics.get('f1', 0) > 0.7:
            report_lines.append("  ✓ Good F1-score achieved!")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


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
