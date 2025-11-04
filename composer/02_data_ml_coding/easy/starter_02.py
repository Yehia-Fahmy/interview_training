"""
Exercise 2: Model Evaluation Metrics

Implement comprehensive evaluation metrics for classification models from scratch.
Create a clean, reusable evaluation module.
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
        conf_matrix = np.zeros((self.n_classes, self.n_classes))
        for i in range(len(self.y_true)):
            true_idx = np.where(self.classes == self.y_true[i])[0][0]
            pred_idx = np.where(self.classes == self.y_pred[i])[0][0]
            conf_matrix[true_idx][pred_idx] += 1
        return conf_matrix

    
    def accuracy(self) -> float:
        """Compute accuracy."""
        total_correct = 0
        total = len(self.y_true)
        for i in range(total):
            if self.y_pred[i] == self.y_true[i]: total_correct += 1
        return total_correct / total
    
    def precision_per_class(self) -> Dict[int, float]:
        """Compute precision for each class."""
        TP = [0 for _ in range(self.n_classes)]
        AP = [0 for _ in range(self.n_classes)]
        conf_matrix = self.confusion_matrix()
        d = {}
        for i in range(self.n_classes):
            AP[i] += np.sum(conf_matrix[:, i])
            TP[i] = conf_matrix[i][i]
            d[self.classes[i]] = TP[i] / AP[i] if AP[i] > 0 else 0.0
        return d
    
    def recall_per_class(self) -> Dict[int, float]:
        """Compute recall for each class."""
        # TODO: Implement recall = TP / (TP + FN) for each class
        # Use confusion matrix to compute TP and FN
        pass
    
    def f1_per_class(self) -> Dict[int, float]:
        """Compute F1-score for each class."""
        # TODO: Implement F1 = 2 * (precision * recall) / (precision + recall)
        # Use precision_per_class() and recall_per_class()
        pass
    
    def macro_averaged_metrics(self) -> Dict[str, float]:
        """Compute macro-averaged precision, recall, F1."""
        # TODO: Implement macro-averaged metrics
        # Average the per-class metrics across all classes
        pass
    
    def roc_auc(self) -> Optional[float]:
        """Compute ROC-AUC (binary classification only)."""
        if self.n_classes != 2 or self.y_proba is None:
            return None
        
        # TODO: Implement ROC-AUC from scratch
        # 1. Sort predictions by probability (descending)
        # 2. Calculate TPR and FPR at each threshold
        # 3. Use trapezoidal rule to compute area under ROC curve
        # Hint: You can use np.trapz for integration
        pass
    
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


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3,
                               n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)
    
    # Train a model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
    
    # Evaluate
    metrics = ClassificationMetrics(y_test, y_pred, y_proba)
    summary = metrics.summary()
    
    print("Classification Metrics Summary:")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro-averaged F1: {summary['macro_averaged']['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(metrics.confusion_matrix())

