"""
Solution for Exercise 8: Cross-validation and Hyperparameter Tuning

This file contains the reference solution.
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
        # Create pipeline if needed
        if use_pipeline:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            # Adjust param_grid keys for pipeline
            param_grid_pipeline = {f'model__{k}': v for k, v in param_grid.items()}
            estimator = pipeline
            param_grid_final = param_grid_pipeline
        else:
            estimator = model
            param_grid_final = param_grid
        
        # Set up GridSearchCV
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid_final,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        # Fit
        grid_search.fit(X, y)
        
        # Extract results
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        results = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        self.cv_results = results
        return results
    
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
        # Create pipeline if needed
        if use_pipeline:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            param_dist_pipeline = {f'model__{k}': v for k, v in param_distributions.items()}
            estimator = pipeline
            param_dist_final = param_dist_pipeline
        else:
            estimator = model
            param_dist_final = param_distributions
        
        # Set up RandomizedSearchCV
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        random_search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist_final,
            n_iter=n_iter,
            cv=cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=self.random_state,
            return_train_score=True
        )
        
        # Fit
        random_search.fit(X, y)
        
        # Extract results
        self.best_model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        results = {
            'best_model': random_search.best_estimator_,
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': pd.DataFrame(random_search.cv_results_)
        }
        
        self.cv_results = results
        return results
    
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
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        results = []
        
        for model_name, (model_class, param_dict) in models.items():
            # Create model with parameters
            model = model_class(**param_dict)
            
            # Create pipeline if needed
            if use_pipeline:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
                estimator = pipeline
            else:
                estimator = model
            
            # Perform cross-validation
            scores = cross_val_score(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv,
                scoring=self.scoring,
                n_jobs=-1
            )
            
            results.append({
                'model': model_name,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'min_score': scores.min(),
                'max_score': scores.max()
            })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('mean_score', ascending=False)
        
        return comparison_df


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
    print(results['cv_results'][['param_n_estimators', 'param_max_depth', 'mean_test_score', 'std_test_score']].head())

