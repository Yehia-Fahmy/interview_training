"""
Exercise 3: Feature Engineering Pipeline

Create a reusable feature engineering pipeline that handles:
1. Missing value imputation
2. Categorical encoding
3. Numerical scaling
4. Feature selection (optional)

The pipeline should be:
- Easy to use (sklearn-style fit/transform)
- Well-documented
- Testable
"""

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
        
        # Initialize transformers (to be fitted)
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
        # TODO: Fit numerical imputer
        # TODO: Fit scaler if scaling is enabled
        # TODO: Fit categorical imputer
        # TODO: Fit encoders for each categorical feature
        pass
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform (DataFrame)
        
        Returns:
            Transformed features (numpy array)
        """
        transformed_parts = []
        
        # TODO: Transform numerical features
        # 1. Impute missing values
        # 2. Scale if enabled
        # 3. Append to transformed_parts
        
        # TODO: Transform categorical features
        # 1. Impute missing values
        # 2. Encode each categorical feature (use label encoding, then one-hot)
        # 3. Append to transformed_parts
        
        # TODO: Combine numerical and categorical features
        # Return as a single numpy array
        
        return np.array([])  # Placeholder
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)


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

