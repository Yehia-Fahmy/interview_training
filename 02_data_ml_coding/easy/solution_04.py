"""
Solution for Exercise 4: End-to-End ML Pipeline

This file contains the reference solution.
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
        self.feature_pipeline = None
        self.train_history = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        file_path = Path(file_path)
        
        if file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def split_data(self, df: pd.DataFrame, target_col: str,
                   test_size: float = 0.2) -> tuple:
        """Split data into train/test sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        random_state = self.config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train the model."""
        # Initialize feature pipeline (simplified - you could use FeaturePipeline from exercise 3)
        # For this solution, we'll use a simple approach
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        
        # Fit and transform training data
        X_train_imputed = self.imputer.fit_transform(X_train.select_dtypes(include=[np.number]))
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        
        # Initialize and train model
        model_type = self.config.get('model_type', 'random_forest')
        model_params = self.config.get('model_params', {})
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set if provided
        history = {}
        if X_val is not None and y_val is not None:
            X_val_imputed = self.imputer.transform(X_val.select_dtypes(include=[np.number]))
            X_val_scaled = self.scaler.transform(X_val_imputed)
            y_val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            history['val_accuracy'] = val_accuracy
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        history['train_accuracy'] = train_accuracy
        
        self.train_history = history
        return history
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on given data."""
        # Transform features
        X_imputed = self.imputer.transform(X.select_dtypes(include=[np.number]))
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        # Transform features
        X_imputed = self.imputer.transform(X.select_dtypes(include=[np.number]))
        X_scaled = self.scaler.transform(X_imputed)
        
        # Return predictions
        return self.model.predict(X_scaled)
    
    def save(self, model_dir: Path):
        """Save model and pipeline to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save preprocessing components
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(model_dir / 'imputer.pkl', 'wb') as f:
            pickle.dump(self.imputer, f)
        
        # Save config
        with open(model_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training history
        with open(model_dir / 'train_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)
    
    @classmethod
    def load(cls, model_dir: Path) -> 'MLPipeline':
        """Load saved pipeline."""
        model_dir = Path(model_dir)
        
        # Load config
        with open(model_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create pipeline instance
        pipeline = cls(config)
        
        # Load model
        with open(model_dir / 'model.pkl', 'rb') as f:
            pipeline.model = pickle.load(f)
        
        # Load preprocessing components
        with open(model_dir / 'scaler.pkl', 'rb') as f:
            pipeline.scaler = pickle.load(f)
        
        with open(model_dir / 'imputer.pkl', 'rb') as f:
            pipeline.imputer = pickle.load(f)
        
        # Load training history
        if (model_dir / 'train_history.json').exists():
            with open(model_dir / 'train_history.json', 'r') as f:
                pipeline.train_history = json.load(f)
        
        return pipeline

