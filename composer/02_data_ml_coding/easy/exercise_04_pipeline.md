# Exercise 4: End-to-End ML Pipeline

**Difficulty:** Easy  
**Time Limit:** 50 minutes  
**Focus:** Complete ML workflow, code organization

## Problem

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

## Requirements

1. Create a pipeline class that orchestrates the entire workflow
2. Support for different model types (configurable)
3. Save/load functionality for models and pipelines
4. Comprehensive logging and progress tracking

## Solution Template

```python
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

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
        """Load data from file."""
        # Your implementation - support CSV, JSON, etc.
        pass
    
    def split_data(self, df: pd.DataFrame, target_col: str,
                   test_size: float = 0.2) -> tuple:
        """Split data into train/test sets."""
        # Your implementation
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training history/metrics
        """
        # Your implementation
        # 1. Engineer features
        # 2. Train model
        # 3. Evaluate on validation set if provided
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on given data."""
        # Your implementation
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        # Your implementation
        pass
    
    def save(self, model_dir: Path):
        """Save model and pipeline to disk."""
        # Save model
        # Save feature pipeline
        # Save config
        pass
    
    @classmethod
    def load(cls, model_dir: Path) -> 'MLPipeline':
        """Load saved pipeline."""
        # Load model, pipeline, config
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
    df = pipeline.load_data('data.csv')
    X_train, X_test, y_train, y_test = pipeline.split_data(df, 'target')
    
    # Train
    history = pipeline.train(X_train, y_train)
    
    # Evaluate
    metrics = pipeline.evaluate(X_test, y_test)
    print("Test Metrics:", metrics)
    
    # Save
    pipeline.save(Path('models/my_model'))
    
    # Load later
    loaded_pipeline = MLPipeline.load(Path('models/my_model'))
```

## Key Learning Points

1. **Pipeline Architecture:** Organizing ML workflows
2. **Configuration Management:** Externalizing settings
3. **Model Persistence:** Saving and loading models
4. **Code Reusability:** Making components reusable

## Design Considerations

- How to structure configuration?
- Should preprocessing be part of pipeline or separate?
- How to version models and pipelines?
- What metadata should be saved?

