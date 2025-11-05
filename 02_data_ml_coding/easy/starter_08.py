"""
Exercise 8: Cross-validation and Hyperparameter Tuning

Implement a comprehensive hyperparameter tuning system that performs cross-validation,
grid search, random search, and model comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning with cross-validation.
    
    Supports grid search, random search, and model comparison.
    """
    
    def __init__(self,
                 task_type: str = 'classification',
                 cv_folds: int = 5,
                 scoring: Optional[str] = None,
                 random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            task_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric (None for default)
            random_state: Random seed
        """
        self.task_type = task_type
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        
        self.best_model = None
        self.best_params = None
        self.cv_results = {}
    
    def grid_search(self,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model,
                   param_grid: Dict[str, List],
                   use_pipeline: bool = True) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            model: Model instance or class
            param_grid: Dictionary of parameter grids
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            Dictionary with best model and results
        """
        # TODO: Implement grid search
        # 1. Create pipeline if needed (scaler + model)
        # 2. Set up GridSearchCV with cv_folds and scoring
        # 3. Fit GridSearchCV
        # 4. Extract best model, best parameters, best score
        # 5. Return dictionary with results
        pass
    
    def random_search(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     model,
                     param_distributions: Dict[str, List],
                     n_iter: int = 50,
                     use_pipeline: bool = True) -> Dict[str, Any]:
        """
        Perform random search for hyperparameter tuning.
        
        Args:
            X: Features
            y: Target
            model: Model instance or class
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of iterations
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            Dictionary with best model and results
        """
        # TODO: Implement random search
        # Similar to grid_search but using RandomizedSearchCV
        # 1. Create pipeline if needed
        # 2. Set up RandomizedSearchCV
        # 3. Fit and extract results
        # 4. Return dictionary with results
        pass
    
    def compare_models(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      models: Dict[str, Any],
                      use_pipeline: bool = True) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation.
        
        Args:
            X: Features
            y: Target
            models: Dictionary of model_name -> (model_class, param_dict)
            use_pipeline: Whether to use pipeline with preprocessing
        
        Returns:
            DataFrame with comparison results
        """
        # TODO: Compare multiple models
        # 1. For each model in models dictionary:
        #    - Create pipeline if needed
        #    - Perform cross-validation using cross_val_score
        #    - Collect mean and std of scores
        # 2. Create DataFrame with model names, mean scores, std scores
        # 3. Return comparison DataFrame
        pass


# Test
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=500, n_features=10, 
                               n_informative=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    
    tuner = HyperparameterTuner(task_type='classification', cv_folds=5)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    
    results = tuner.grid_search(X, pd.Series(y), 
                                 RandomForestClassifier(random_state=42),
                                 param_grid)
    
    print("Best Parameters:", results['best_params'])
    print("Best CV Score:", results['best_score'])
    print("\nCV Results Summary:")
    print(results['cv_results'])

