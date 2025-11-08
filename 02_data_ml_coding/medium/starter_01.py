"""
Exercise 1: Feature Engineering & Selection Pipeline

Build a comprehensive feature engineering and selection pipeline for classification.
Handle missing values, create derived features, encode categoricals, and select important features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (
    mutual_info_classif,
    f_classif,
    SelectKBest,
    RFE
)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering pipeline.
    
    Handles missing values, creates interaction features, encodes categoricals,
    and scales features. Designed to prevent data leakage and handle unseen data.
    """
    
    def __init__(self, 
                 missing_strategy: Dict[str, str] = None,
                 create_interactions: bool = True,
                 interaction_pairs: Optional[List[Tuple[str, str]]] = None,
                 categorical_encoding: str = 'onehot',  # 'onehot', 'target', 'frequency'
                 scaling: str = 'standard',  # 'standard', 'minmax', None
                 handle_unseen: str = 'ignore'):  # 'ignore', 'error', 'most_frequent'
        """
        Initialize feature engineering pipeline.
        
        Args:
            missing_strategy: Dict mapping feature names to imputation strategy
                            ('mean', 'median', 'mode', 'forward_fill', 'constant')
            create_interactions: Whether to create interaction features
            interaction_pairs: Specific pairs of features to create interactions for
            categorical_encoding: Method for encoding categorical features
            scaling: Scaling method for numeric features
            handle_unseen: How to handle unseen categories in test data
        """
        self.missing_strategy = missing_strategy or {}
        self.create_interactions = create_interactions
        self.interaction_pairs = interaction_pairs
        self.categorical_encoding = categorical_encoding
        self.scaling = scaling
        self.handle_unseen = handle_unseen
        
        # Store learned parameters during fit
        self.imputation_values_ = {}
        self.categorical_encoders_ = {}
        self.scaler_ = None
        self.feature_names_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features
            y: Training target (optional, needed for target encoding)
        
        Returns:
            self
        """
        # TODO: Implement pipeline fitting
        # 1. Identify numeric vs categorical features
        # 2. Learn imputation values for missing data
        # 3. Fit encoders for categorical features (if target encoding, need y)
        # 4. Fit scaler if needed
        # 5. Store feature names and transformations
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using learned parameters.
        
        Args:
            X: Data to transform
        
        Returns:
            Transformed DataFrame
        """
        # TODO: Implement transformation
        # 1. Handle missing values using learned strategies
        # 2. Create interaction features
        # 3. Encode categorical features (handle unseen categories!)
        # 4. Scale numeric features
        # 5. Ensure consistent feature order
        
        pass
    
    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify numeric and categorical features"""
        # TODO: Implement feature type identification
        # Numeric: int64, float64
        # Categorical: object, category, or low-cardinality numeric features
        pass
    
    def _handle_missing_values(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Handle missing values according to strategy"""
        # TODO: Implement missing value handling
        # For each feature:
        #   - If strategy specified, use it
        #   - If numeric and no strategy: use median
        #   - If categorical and no strategy: use mode
        #   - Store learned values during fit
        #   - Use stored values during transform
        pass
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # TODO: Implement interaction feature creation
        # If interaction_pairs specified, create interactions for those pairs
        # Otherwise, create interactions for all numeric feature pairs (be careful of explosion!)
        # Consider: product, ratio, difference, sum
        # Only create interactions that make sense (e.g., ratio where denominator != 0)
        pass
    
    def _encode_categoricals(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                            is_fit: bool = False) -> pd.DataFrame:
        """Encode categorical features"""
        # TODO: Implement categorical encoding
        # One-hot encoding: Create binary columns for each category
        # Target encoding: Encode by mean target value per category (need y during fit!)
        # Frequency encoding: Encode by frequency of category
        # Handle unseen categories appropriately
        pass
    
    def _scale_features(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Scale numeric features"""
        # TODO: Implement feature scaling
        # Standard scaling: (x - mean) / std
        # MinMax scaling: (x - min) / (max - min)
        # Only scale numeric features
        pass


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection using multiple methods.
    
    Supports correlation-based, mutual information, and recursive feature elimination.
    """
    
    def __init__(self, 
                 method: str = 'mutual_info',  # 'mutual_info', 'f_test', 'rfe', 'correlation'
                 k: int = 10,
                 threshold: Optional[float] = None,
                 correlation_threshold: float = 0.95):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method to use
            k: Number of features to select (if threshold not specified)
            threshold: Minimum importance score threshold
            correlation_threshold: Threshold for removing highly correlated features
        """
        self.method = method
        self.k = k
        self.threshold = threshold
        self.correlation_threshold = correlation_threshold
        
        self.selected_features_ = []
        self.feature_importances_ = {}
        self.selector_ = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit feature selector.
        
        Args:
            X: Features
            y: Target
        
        Returns:
            self
        """
        # TODO: Implement feature selection fitting
        # 1. Remove highly correlated features (if method includes this)
        # 2. Apply selection method (mutual_info, f_test, RFE)
        # 3. Store selected features and importance scores
        # 4. Determine k based on threshold if specified
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from data.
        
        Args:
            X: Data to transform
        
        Returns:
            DataFrame with selected features
        """
        # TODO: Return only selected features
        pass
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features"""
        # TODO: Implement correlation-based removal
        # Calculate correlation matrix
        # Identify feature pairs above threshold
        # Remove one feature from each pair (keep the one with higher variance or importance)
        # Return filtered DataFrame and list of removed features
        pass
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """Get features ranked by importance"""
        # TODO: Return DataFrame with features and importance scores, sorted
        pass


# Usage example
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create synthetic dataset with missing values and categoricals
    np.random.seed(42)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Add some missing values
    missing_indices = np.random.choice(X.index, size=100, replace=False)
    X.loc[missing_indices, 'feature_0'] = np.nan
    
    # Add categorical features
    X['category'] = np.random.choice(['A', 'B', 'C', 'D'], size=1000)
    X['category2'] = np.random.choice(['X', 'Y'], size=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit pipeline
    pipeline = FeatureEngineeringPipeline(
        missing_strategy={'feature_0': 'median'},
        create_interactions=True,
        categorical_encoding='onehot',
        scaling='standard'
    )
    
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Transformed features: {X_train_transformed.shape[1]}")
    
    # Feature selection
    selector = FeatureSelector(method='mutual_info', k=15)
    X_train_selected = selector.fit_transform(X_train_transformed, y_train)
    X_test_selected = selector.transform(X_test_transformed)
    
    print(f"Selected features: {X_train_selected.shape[1]}")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_selected, y_train)
    
    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train_selected))
    test_acc = accuracy_score(y_test, model.predict(X_test_selected))
    
    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Feature importance
    importance_df = selector.get_feature_importance_ranking()
    print("\nTop 10 Features:")
    print(importance_df.head(10))
