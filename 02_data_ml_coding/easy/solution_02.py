"""
Solution for Exercise 2: Model Evaluation Metrics

This file contains the reference solution.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ClassificationMetrics:
    """
    Comprehensive classification metrics calculator.
    
    This implementation focuses on clarity and correctness over using
    sklearn directly, though sklearn can be used for validation.
    """
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_proba: Optional[np.ndarray] = None):
        """
        Initialize with true labels and predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC-AUC)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = y_proba if y_proba is not None else None
        self.classes = np.unique(np.concatenate([self.y_true, self.y_pred]))
        self.n_classes = len(self.classes)
    
    def confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix from scratch."""
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        
        # Map class labels to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for true_label, pred_label in zip(self.y_true, self.y_pred):
            true_idx = class_to_idx[true_label]
            pred_idx = class_to_idx[pred_label]
            cm[true_idx, pred_idx] += 1
        
        return cm
    
    def accuracy(self) -> float:
        """Compute accuracy."""
        correct = np.sum(self.y_true == self.y_pred)
        return correct / len(self.y_true)
    
    def precision_per_class(self) -> Dict[int, float]:
        """Compute precision for each class."""
        cm = self.confusion_matrix()
        precision_dict = {}
        
        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_dict[int(cls)] = precision
        
        return precision_dict
    
    def recall_per_class(self) -> Dict[int, float]:
        """Compute recall for each class."""
        cm = self.confusion_matrix()
        recall_dict = {}
        
        for i, cls in enumerate(self.classes):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_dict[int(cls)] = recall
        
        return recall_dict
    
    def f1_per_class(self) -> Dict[int, float]:
        """Compute F1-score for each class."""
        precision = self.precision_per_class()
        recall = self.recall_per_class()
        f1_dict = {}
        
        for cls in self.classes:
            p = precision[int(cls)]
            r = recall[int(cls)]
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
            f1_dict[int(cls)] = f1
        
        return f1_dict
    
    def macro_averaged_metrics(self) -> Dict[str, float]:
        """Compute macro-averaged precision, recall, F1."""
        precision = self.precision_per_class()
        recall = self.recall_per_class()
        f1 = self.f1_per_class()
        
        return {
            'precision': np.mean(list(precision.values())),
            'recall': np.mean(list(recall.values())),
            'f1': np.mean(list(f1.values()))
        }
    
    def roc_auc(self) -> Optional[float]:
        """Compute ROC-AUC (binary classification only)."""
        if self.n_classes != 2 or self.y_proba is None:
            return None
        
        # For binary classification, use the positive class probabilities
        positive_class = self.classes[1] if self.classes[0] == 0 else self.classes[0]
        
        # Get binary labels (0 for negative, 1 for positive)
        y_true_binary = (self.y_true == positive_class).astype(int)
        
        # Sort by probability (descending)
        sorted_indices = np.argsort(self.y_proba)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        y_proba_sorted = self.y_proba[sorted_indices]
        
        # Calculate TPR and FPR at each threshold
        tpr = []
        fpr = []
        
        # Number of positive and negative samples
        num_pos = np.sum(y_true_binary)
        num_neg = len(y_true_binary) - num_pos
        
        if num_pos == 0 or num_neg == 0:
            return None
        
        # Calculate TPR and FPR for each unique threshold
        thresholds = np.unique(y_proba_sorted)
        thresholds = np.append(thresholds, 1.0)  # Add threshold at 1.0
        
        for threshold in thresholds:
            # Predictions at this threshold
            y_pred_threshold = (y_proba_sorted >= threshold).astype(int)
            
            # True positives and false positives
            tp = np.sum((y_pred_threshold == 1) & (y_true_sorted == 1))
            fp = np.sum((y_pred_threshold == 1) & (y_true_sorted == 0))
            
            tpr.append(tp / num_pos if num_pos > 0 else 0.0)
            fpr.append(fp / num_neg if num_neg > 0 else 0.0)
        
        # Calculate AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        return auc
    
    def summary(self) -> Dict:
        """Return comprehensive metrics summary."""
        summary = {
            'accuracy': self.accuracy(),
            'per_class': {
                'precision': self.precision_per_class(),
                'recall': self.recall_per_class(),
                'f1': self.f1_per_class()
            },
            'macro_averaged': self.macro_averaged_metrics(),
            'confusion_matrix': self.confusion_matrix().tolist()
        }
        
        if self.n_classes == 2:
            roc_auc = self.roc_auc()
            if roc_auc is not None:
                summary['roc_auc'] = roc_auc
        
        return summary

