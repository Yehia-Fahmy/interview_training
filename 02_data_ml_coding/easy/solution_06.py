"""
Solution for Exercise 6: Advanced Feature Engineering with Pandas and Scikit-learn

This file contains the reference solution.
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
        # Fit polynomial transformer
        if self.create_polynomials:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.polynomial_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
                self.polynomial_transformer.fit(X[numeric_cols])
        
        # Fit discretizer
        if self.n_bins > 1:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
                self.discretizer.fit(X[numeric_cols])
        
        # Compute target encodings
        if y is not None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                encoding = y.groupby(X[col]).mean()
                self.target_encodings[col] = encoding
        
        # Store aggregation statistics for groupby features
        if self.groupby_columns and self.aggregate_columns:
            for group_col in self.groupby_columns:
                if group_col in X.columns:
                    for agg_col in self.aggregate_columns:
                        if agg_col in X.columns:
                            key = f"{group_col}_{agg_col}"
                            self.aggregated_features[key] = {
                                'mean': X.groupby(group_col)[agg_col].mean(),
                                'std': X.groupby(group_col)[agg_col].std(),
                                'count': X.groupby(group_col)[agg_col].count()
                            }
        
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
        
        # Extract temporal features
        for col in self.date_columns:
            if col in X_transformed.columns:
                if pd.api.types.is_datetime64_any_dtype(X_transformed[col]):
                    X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                    X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                    X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                    X_transformed[f'{col}_dayofweek'] = X_transformed[col].dt.dayofweek
                    X_transformed[f'{col}_dayofyear'] = X_transformed[col].dt.dayofyear
                    X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
        
        # Add aggregated features
        for group_col in self.groupby_columns:
            if group_col in X_transformed.columns:
                for agg_col in self.aggregate_columns:
                    if agg_col in X_transformed.columns:
                        key = f"{group_col}_{agg_col}"
                        if key in self.aggregated_features:
                            X_transformed[f'{agg_col}_mean_by_{group_col}'] = X_transformed[group_col].map(
                                self.aggregated_features[key]['mean']
                            )
                            X_transformed[f'{agg_col}_std_by_{group_col}'] = X_transformed[group_col].map(
                                self.aggregated_features[key]['std']
                            )
        
        # Create interaction features
        if self.create_interactions:
            numeric_cols = X_transformed.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                for i, col1 in enumerate(numeric_cols):
                    for col2 in numeric_cols[i+1:]:
                        X_transformed[f'{col1}_x_{col2}'] = X_transformed[col1] * X_transformed[col2]
        
        # Apply polynomial transformation
        if self.create_polynomials and self.polynomial_transformer is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                poly_features = self.polynomial_transformer.transform(X_transformed[numeric_cols])
                poly_df = pd.DataFrame(
                    poly_features,
                    columns=self.polynomial_transformer.get_feature_names_out(numeric_cols),
                    index=X_transformed.index
                )
                X_transformed = pd.concat([X_transformed, poly_df], axis=1)
        
        # Apply discretization
        if self.discretizer is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                discretized = self.discretizer.transform(X_transformed[numeric_cols])
                discretized_df = pd.DataFrame(
                    discretized,
                    columns=[f'{col}_binned' for col in numeric_cols],
                    index=X_transformed.index
                )
                X_transformed = pd.concat([X_transformed, discretized_df], axis=1)
        
        # Apply target encoding
        for col, encoding in self.target_encodings.items():
            if col in X_transformed.columns:
                X_transformed[f'{col}_target_encoded'] = X_transformed[col].map(encoding)
                X_transformed[f'{col}_target_encoded'].fillna(encoding.mean(), inplace=True)
        
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

