"""
Exercise 6: Advanced Feature Engineering with Pandas and Scikit-learn

Create a feature engineering module that generates advanced features from raw data:
temporal features, aggregated features, interaction features, binning, and target encoding.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering pipeline.
    
    Generates temporal, aggregated, interaction, and encoded features.
    """
    
    def __init__(self,
                 date_columns: Optional[List[str]] = None,
                 groupby_columns: Optional[List[str]] = None,
                 aggregate_columns: Optional[List[str]] = None,
                 create_interactions: bool = True,
                 create_polynomials: bool = False,
                 n_bins: int = 5):
        """
        Initialize feature engineer.
        
        Args:
            date_columns: Columns to extract temporal features from
            groupby_columns: Columns to use for groupby aggregations
            aggregate_columns: Columns to aggregate
            create_interactions: Whether to create interaction features
            create_polynomials: Whether to create polynomial features
            n_bins: Number of bins for discretization
        """
        self.date_columns = date_columns or []
        self.groupby_columns = groupby_columns or []
        self.aggregate_columns = aggregate_columns or []
        self.create_interactions = create_interactions
        self.create_polynomials = create_polynomials
        self.n_bins = n_bins
        
        self.polynomial_transformer = None
        self.discretizer = None
        self.target_encodings = {}
        self.aggregated_features = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the feature engineer on training data.
        
        Args:
            X: Training features
            y: Training target (optional, needed for target encoding)
        """
        # TODO: Fit transformers and compute encodings
        # 1. Fit polynomial transformer if enabled (on numeric columns)
        # 2. Fit discretizer if binning is needed (on numeric columns)
        # 3. Compute target encodings if y is provided (for categorical columns)
        # 4. Store aggregation statistics for groupby features
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        """
        X_transformed = X.copy()
        
        # TODO: Apply all transformations
        # 1. Extract temporal features from date columns (year, month, day, day_of_week, etc.)
        # 2. Add aggregated features using groupby (mean, std, count, etc.)
        # 3. Create interaction features (multiplication of numeric columns)
        # 4. Apply polynomial transformation if enabled
        # 5. Apply discretization to numeric columns if enabled
        # 6. Apply target encoding to categorical columns if available
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


# Test
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'customer_id': np.random.randint(1, 10, 100),
        'product_category': np.random.choice(['A', 'B', 'C'], 100),
        'amount': np.random.uniform(10, 1000, 100),
        'quantity': np.random.randint(1, 10, 100)
    })
    
    target = pd.Series(np.random.randint(0, 2, 100))
    
    engineer = FeatureEngineer(
        date_columns=['transaction_date'],
        groupby_columns=['customer_id'],
        aggregate_columns=['amount'],
        create_interactions=True
    )
    
    features = engineer.fit_transform(data, target)
    print(f"Original shape: {data.shape}")
    print(f"Engineered shape: {features.shape}")
    print(f"\nNew columns: {set(features.columns) - set(data.columns)}")

