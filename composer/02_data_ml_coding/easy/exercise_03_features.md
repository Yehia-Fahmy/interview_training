# Exercise 3: Feature Engineering Pipeline

**Difficulty:** Easy  
**Time Limit:** 40 minutes  
**Focus:** Clean pipeline design, feature engineering best practices

## Problem

Create a reusable feature engineering pipeline that handles:
1. Missing value imputation
2. Categorical encoding
3. Numerical scaling
4. Feature selection (optional)

The pipeline should be:
- Easy to use (sklearn-style fit/transform)
- Well-documented
- Testable

## Requirements

1. Create a `FeaturePipeline` class with:
   - `fit()` method to learn transformations
   - `transform()` method to apply transformations
   - Support for both numerical and categorical features

2. Implement common transformations:
   - Numerical: Standardization, normalization, imputation
   - Categorical: One-hot encoding, label encoding, target encoding

3. Write example usage with real-world data

## Solution Template

```python
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class FeaturePipeline:
    """
    Reusable feature engineering pipeline.
    
    Handles missing values, encoding, and scaling in a clean pipeline.
    """
    
    def __init__(self, 
                 numerical_features: List[str],
                 categorical_features: List[str],
                 imputation_strategy: str = 'mean',
                 scaling: bool = True):
        """
        Initialize feature pipeline.
        
        Args:
            numerical_features: List of numerical column names
            categorical_features: List of categorical column names
            imputation_strategy: Strategy for missing values ('mean', 'median', 'mode')
            scaling: Whether to scale numerical features
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.imputation_strategy = imputation_strategy
        self.scaling = scaling
        
        # Initialize transformers
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}  # One encoder per categorical feature
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (DataFrame)
            y: Training labels (optional, for target encoding)
        """
        # Fit numerical imputer
        if self.numerical_features:
            self.numerical_imputer = SimpleImputer(strategy=self.imputation_strategy)
            self.numerical_imputer.fit(X[self.numerical_features])
            
            # Fit scaler
            if self.scaling:
                self.scaler = StandardScaler()
                imputed = self.numerical_imputer.transform(X[self.numerical_features])
                self.scaler.fit(imputed)
        
        # Fit categorical imputer and encoders
        if self.categorical_features:
            self.categorical_imputer = SimpleImputer(strategy='most_frequent')
            self.categorical_imputer.fit(X[self.categorical_features])
            
            for col in self.categorical_features:
                self.encoders[col] = LabelEncoder()
                imputed_col = self.categorical_imputer.transform(X[[col]])[:, 0]
                # Only fit on non-null values
                non_null_mask = pd.notna(X[col])
                if non_null_mask.any():
                    self.encoders[col].fit(imputed_col[non_null_mask])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform (DataFrame)
        
        Returns:
            Transformed features (numpy array)
        """
        transformed_parts = []
        
        # Transform numerical features
        if self.numerical_features:
            imputed = self.numerical_imputer.transform(X[self.numerical_features])
            if self.scaling:
                imputed = self.scaler.transform(imputed)
            transformed_parts.append(imputed)
        
        # Transform categorical features
        if self.categorical_features:
            imputed = self.categorical_imputer.transform(X[self.categorical_features])
            
            # Encode each categorical feature
            encoded_features = []
            for i, col in enumerate(self.categorical_features):
                encoded = self.encoders[col].transform(imputed[:, i])
                # One-hot encode
                n_classes = len(self.encoders[col].classes_)
                one_hot = np.zeros((len(encoded), n_classes))
                one_hot[np.arange(len(encoded)), encoded] = 1
                encoded_features.append(one_hot)
            
            if encoded_features:
                transformed_parts.append(np.hstack(encoded_features))
        
        return np.hstack(transformed_parts) if transformed_parts else np.array([])

# Test
if __name__ == "__main__":
    # Create sample data with missing values
    data = pd.DataFrame({
        'age': [25, 30, None, 35, 40, None],
        'salary': [50000, 60000, 70000, None, 80000, 90000],
        'city': ['NYC', 'SF', None, 'NYC', 'SF', 'NYC'],
        'department': ['Engineering', 'Sales', 'Engineering', None, 'Sales', 'Engineering']
    })
    
    pipeline = FeaturePipeline(
        numerical_features=['age', 'salary'],
        categorical_features=['city', 'department'],
        scaling=True
    )
    
    pipeline.fit(data)
    transformed = pipeline.transform(data)
    print("Transformed shape:", transformed.shape)
    print("Sample transformed features:\n", transformed[:3])
```

## Key Learning Points

1. **Pipeline Design:** sklearn-style API for consistency
2. **Separation of Concerns:** Fit learns, transform applies
3. **Documentation:** Clear docstrings and type hints

## Design Considerations

- Why separate fit and transform?
- How to handle unseen categories in transform?
- Should you support different encoding strategies?

