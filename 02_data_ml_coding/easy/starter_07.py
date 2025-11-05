"""
Exercise 7: Model Training Pipeline with Scikit-learn

Create a comprehensive model training pipeline using scikit-learn that supports
multiple model types, proper data splitting, and evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class ModelTrainer:
    """
    Comprehensive model training pipeline using scikit-learn.
    
    Supports both classification and regression tasks.
    """
    
    def __init__(self, 
                 task_type: str = 'classification',
                 model_type: str = 'random_forest',
                 model_params: Optional[Dict] = None,
                 use_scaling: bool = True):
        """
        Initialize model trainer.
        
        Args:
            task_type: 'classification' or 'regression'
            model_type: Type of model to use
            model_params: Hyperparameters for the model
            use_scaling: Whether to scale features
        """
        self.task_type = task_type
        self.model_type = model_type
        self.model_params = model_params or {}
        self.use_scaling = use_scaling
        
        self.model = None
        self.pipeline = None
        self.history = {}
    
    def _create_model(self):
        """Create the model based on task type and model type."""
        # TODO: Create appropriate model based on task_type and model_type
        # Classification: RandomForestClassifier, LogisticRegression
        # Regression: RandomForestRegressor, LinearRegression
        # Use self.model_params to set hyperparameters
        pass
    
    def train(self, X: pd.DataFrame, y: pd.Series,
              test_size: float = 0.2, 
              val_size: float = 0.2,
              random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model with proper train/val/test splits.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of test set
            val_size: Proportion of validation set (from training set)
            random_state: Random seed
        
        Returns:
            Dictionary with training history and metrics
        """
        # TODO: Implement training pipeline
        # 1. Split data (train/test, then train/val from training set)
        # 2. Create preprocessing pipeline (scaler if use_scaling)
        # 3. Create model using _create_model()
        # 4. Build sklearn Pipeline with preprocessing and model
        # 5. Fit pipeline on training data
        # 6. Evaluate on validation and test sets with appropriate metrics
        # 7. Store results in self.history
        # 8. Return self.history
        
        return self.history
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on given data."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # TODO: Make predictions and compute appropriate metrics
        # Classification: accuracy, precision, recall, f1
        # Regression: mse, mae, r2
        # Return dictionary of metrics
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.pipeline is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.pipeline.predict(X)


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    
    # Test classification
    X_clf, y_clf = make_classification(n_samples=1000, n_features=10, 
                                       n_informative=5, random_state=42)
    X_clf = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(10)])
    
    trainer_clf = ModelTrainer(task_type='classification', 
                               model_type='random_forest',
                               model_params={'n_estimators': 100, 'random_state': 42})
    results_clf = trainer_clf.train(X_clf, pd.Series(y_clf))
    
    print("Classification Results:")
    print(results_clf)
    
    # Test regression
    X_reg, y_reg = make_regression(n_samples=1000, n_features=10, 
                                   n_informative=5, random_state=42)
    X_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(10)])
    
    trainer_reg = ModelTrainer(task_type='regression',
                               model_type='random_forest',
                               model_params={'n_estimators': 100, 'random_state': 42})
    results_reg = trainer_reg.train(X_reg, pd.Series(y_reg))
    
    print("\nRegression Results:")
    print(results_reg)

