"""
Exercise 4: End-to-End ML Pipeline

Create a complete, production-ready ML pipeline that:
1. Loads and splits data
2. Engineers features
3. Trains a model
4. Evaluates performance
5. Saves the model

The code should be:
- Well-organized (separate functions/classes)
- Documented
- Easy to modify and extend
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class MLPipeline:
    """
    End-to-end machine learning pipeline.
    
    Handles data loading, preprocessing, training, evaluation, and model saving.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary with model settings, paths, etc.
        """
        self.config = config
        self.model = None
        self.feature_pipeline = None  # From previous exercise
        self.train_history = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from file.
        
        Args:
            file_path: Path to data file (CSV, JSON, etc.)
        
        Returns:
            Loaded DataFrame
        """
        # TODO: Implement data loading
        # Support CSV and JSON formats
        # Handle errors gracefully
        pass
    
    def split_data(self, df: pd.DataFrame, target_col: str,
                   test_size: float = 0.2) -> tuple:
        """
        Split data into train/test sets.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            test_size: Proportion of data for test set
        
        Returns:
            (X_train, X_test, y_train, y_test) tuple
        """
        # TODO: Implement train/test split
        # Use train_test_split from sklearn
        # Return features and target separately
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Training history/metrics
        """
        # TODO: Implement training
        # 1. Engineer features (fit and transform feature pipeline)
        # 2. Train model based on config
        # 3. Evaluate on validation set if provided
        # 4. Return training metrics
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            X: Features
            y: True labels
        
        Returns:
            Dictionary of metrics
        """
        # TODO: Implement evaluation
        # 1. Transform features using pipeline
        # 2. Make predictions
        # 3. Compute metrics (accuracy, etc.)
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predictions array
        """
        # TODO: Implement prediction
        # 1. Transform features using pipeline
        # 2. Return model predictions
        pass
    
    def save(self, model_dir: Path):
        """
        Save model and pipeline to disk.
        
        Args:
            model_dir: Directory to save model
        """
        # TODO: Save model, feature pipeline, and config
        # Create directory if it doesn't exist
        # Save model as pickle
        # Save config as JSON
        pass
    
    @classmethod
    def load(cls, model_dir: Path) -> 'MLPipeline':
        """
        Load saved pipeline.
        
        Args:
            model_dir: Directory containing saved model
        
        Returns:
            Loaded MLPipeline instance
        """
        # TODO: Load model, pipeline, and config
        # Load config from JSON
        # Create pipeline instance
        # Load model and feature pipeline
        pass


# Example usage
if __name__ == "__main__":
    config = {
        'model_type': 'random_forest',
        'model_params': {'n_estimators': 100, 'random_state': 42},
        'test_size': 0.2,
        'random_state': 42
    }
    
    pipeline = MLPipeline(config)
    
    # Load and split data
    # df = pipeline.load_data('data.csv')
    # X_train, X_test, y_train, y_test = pipeline.split_data(df, 'target')
    
    # Train
    # history = pipeline.train(X_train, y_train)
    
    # Evaluate
    # metrics = pipeline.evaluate(X_test, y_test)
    # print("Test Metrics:", metrics)
    
    # Save
    # pipeline.save(Path('models/my_model'))
    
    # Load later
    # loaded_pipeline = MLPipeline.load(Path('models/my_model'))

