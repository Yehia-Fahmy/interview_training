# Exercise 2: Model Evaluation Metrics

**Difficulty:** Easy  
**Time Limit:** 30 minutes  
**Focus:** Understanding evaluation metrics, implementation quality

## Problem

Implement comprehensive evaluation metrics for classification models. Create a clean, reusable evaluation module.

## Requirements

1. Implement the following metrics from scratch:
   - Accuracy
   - Precision (per class and macro-averaged)
   - Recall (per class and macro-averaged)
   - F1-score (per class and macro-averaged)
   - Confusion matrix
   - ROC-AUC (for binary classification)

2. Create a class that:
   - Computes all metrics
   - Formats results nicely
   - Handles binary and multiclass scenarios

3. Write unit tests to validate your implementation

## Solution Template

```python
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        """Compute confusion matrix."""
        # Your implementation from scratch
        pass
    
    def accuracy(self) -> float:
        """Compute accuracy."""
        # Your implementation
        pass
    
    def precision_per_class(self) -> Dict[int, float]:
        """Compute precision for each class."""
        # Your implementation
        pass
    
    def recall_per_class(self) -> Dict[int, float]:
        """Compute recall for each class."""
        # Your implementation
        pass
    
    def f1_per_class(self) -> Dict[int, float]:
        """Compute F1-score for each class."""
        # Your implementation
        pass
    
    def macro_averaged_metrics(self) -> Dict[str, float]:
        """Compute macro-averaged precision, recall, F1."""
        # Your implementation
        pass
    
    def roc_auc(self) -> Optional[float]:
        """Compute ROC-AUC (binary classification only)."""
        if self.n_classes != 2 or self.y_proba is None:
            return None
        # Your implementation
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
```

## Key Learning Points

1. **Metric Understanding:** Know what each metric means
2. **Implementation Quality:** Clean, tested code
3. **Edge Cases:** Handle binary vs multiclass, missing probabilities

## Design Considerations

- Why implement from scratch vs using sklearn? (Understanding)
- How to handle edge cases (single class, perfect predictions)?
- Should metrics handle class imbalance?

