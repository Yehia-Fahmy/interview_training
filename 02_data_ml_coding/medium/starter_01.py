"""
Exercise 1: Feature Engineering & Selection Pipeline

Build a comprehensive feature engineering and selection pipeline for classification.
Handle missing values, create derived features, encode categoricals, and select important features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals._packaging.version import collections
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

RANDOM_STATE = 89

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
        self.scaler_columns_ = []  # Store which columns to scale
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
        self._identify_feature_types(X)
        
        # 2. Handle missing values - learn imputation strategies
        X = self._handle_missing_values(X, is_fit=True)
        
        # 3. Create interaction features (if enabled)
        if self.create_interactions:
            interaction_features = self._create_interaction_features(X)
            # Concatenate interaction features with original features
            X = pd.concat([X, interaction_features], axis=1)
        
        # 4. Encode categorical features - learn encoding parameters
        X = self._encode_categoricals(X, y, is_fit=True)
        
        # 5. Scale numeric features - learn scaling parameters
        X = self._scale_features(X, is_fit=True)
        
        # 6. Store feature names after all transformations
        self.feature_names_ = list(X.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using learned parameters.
        
        Args:
            X: Data to transform
        
        Returns:
            Transformed DataFrame
        """
        # 1. Handle missing values using learned strategies
        X = self._handle_missing_values(X, is_fit=False)
        
        # 2. Create interaction features
        if self.create_interactions:
            interaction_features = self._create_interaction_features(X)
            # Concatenate interaction features with original features
            X = pd.concat([X, interaction_features], axis=1)
        
        # 3. Encode categorical features (handle unseen categories!)
        X = self._encode_categoricals(X, y=None, is_fit=False)
        
        # 4. Scale numeric features
        X = self._scale_features(X, is_fit=False)
        
        # 5. Ensure consistent feature order
        # Reorder columns to match the order learned during fit
        if hasattr(self, 'feature_names_') and len(self.feature_names_) > 0:
            # Only keep columns that exist in both X and feature_names_
            available_features = [f for f in self.feature_names_ if f in X.columns]
            # Add any new columns that weren't in training (shouldn't happen, but be safe)
            new_features = [f for f in X.columns if f not in self.feature_names_]
            X = X[available_features + new_features]
        
        return X
    
    def _identify_feature_types(self, X: pd.DataFrame):
        """Identify numeric and categorical features"""
        # TODO: Implement feature type identification
        # HINT: Iterate through X.columns (not rows!)
        #   - Check X[col].dtype to determine type
        #   - Numeric: dtype in ['int64', 'float64'] or np.issubdtype(X[col].dtype, np.number)
        #   - Categorical: dtype == 'object' or dtype.name == 'category'
        #   - Optional: Consider low-cardinality numeric features (< 10 unique values) as categorical
        #   - Append to self.numeric_features_ and self.categorical_features_ lists
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                self.numeric_features_.append(col)
            else:
                self.categorical_features_.append(col)
    
    def _handle_missing_values(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Handle missing values according to strategy"""
        # TODO: Implement missing value handling
        # For each feature:
        #   - If strategy specified, use it
        #   - If numeric and no strategy: use median
        #   - If categorical and no strategy: use mode
        #   - Store learned values during fit
        #   - Use stored values during transform
        
        X = X.copy()  # Work on a copy to avoid modifying original
        
        # Handle numeric features
        for feature in self.numeric_features_:
            if X[feature].isna().any():  # Only process if there are missing values
                strategy = self.missing_strategy.get(feature, 'median')  # Default to median
                
                if is_fit:
                    # During fit: learn and store the imputation value
                    if strategy == 'mean':
                        imputation_value = X[feature].mean()
                    elif strategy == 'median':
                        imputation_value = X[feature].median()
                    elif strategy == 'mode':
                        imputation_value = X[feature].mode()[0] if not X[feature].mode().empty else 0
                    elif strategy == 'forward_fill':
                        imputation_value = None  # Forward fill doesn't need a stored value
                    elif strategy == 'constant':
                        imputation_value = self.missing_strategy.get(f'{feature}_constant', 0)
                    else:
                        imputation_value = X[feature].median()  # Default fallback
                    
                    if strategy != 'forward_fill':
                        self.imputation_values_[feature] = imputation_value
                        # Also fill missing values in training data using learned value
                        X[feature] = X[feature].fillna(imputation_value)
                    else:
                        # Forward fill for training data
                        X[feature] = X[feature].ffill().bfill()
                else:
                    # During transform: use stored imputation value
                    if strategy == 'forward_fill':
                        X[feature] = X[feature].ffill().bfill()
                    else:
                        imputation_value = self.imputation_values_.get(feature)
                        if imputation_value is not None:
                            X[feature] = X[feature].fillna(imputation_value)
        
        # Handle categorical features
        for feature in self.categorical_features_:
            if X[feature].isna().any():  # Only process if there are missing values
                strategy = self.missing_strategy.get(feature, 'mode')  # Default to mode
                
                if is_fit:
                    # During fit: learn and store the imputation value
                    if strategy == 'mode':
                        mode_values = X[feature].mode()
                        imputation_value = mode_values[0] if not mode_values.empty else 'unknown'
                    elif strategy == 'constant':
                        imputation_value = self.missing_strategy.get(f'{feature}_constant', 'unknown')
                    elif strategy == 'forward_fill':
                        imputation_value = None  # Forward fill doesn't need a stored value
                    else:
                        # For other strategies on categorical, default to mode
                        mode_values = X[feature].mode()
                        imputation_value = mode_values[0] if not mode_values.empty else 'unknown'
                    
                    if strategy != 'forward_fill':
                        self.imputation_values_[feature] = imputation_value
                        # Also fill missing values in training data using learned value
                        X[feature] = X[feature].fillna(imputation_value)
                    else:
                        # Forward fill for training data
                        X[feature] = X[feature].ffill().bfill()
                else:
                    # During transform: use stored imputation value
                    if strategy == 'forward_fill':
                        X[feature] = X[feature].ffill().bfill()
                    else:
                        imputation_value = self.imputation_values_.get(feature, 'unknown')
                        X[feature] = X[feature].fillna(imputation_value)
        return X
    
    def _create_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # TODO: Implement interaction feature creation
        # If interaction_pairs specified, create interactions for those pairs
        # Otherwise, create interactions for all numeric feature pairs (be careful of explosion!)
        # Consider: product, ratio, difference, sum
        # Only create interactions that make sense (e.g., ratio where denominator != 0)
        
        # Check if interactions should be created
        if not self.create_interactions:
            return pd.DataFrame(index=X.index)
        
        # Determine which pairs to use
        if self.interaction_pairs is None or len(self.interaction_pairs) == 0:
            # Generate all pairs from numeric features if not specified
            from itertools import combinations
            # Filter to only features that exist in X and are numeric
            available_numeric = [f for f in self.numeric_features_ if f in X.columns]
            pairs = list(combinations(available_numeric, 2))
        else:
            # Filter pairs to only those where both features exist in X
            pairs = [(f1, f2) for f1, f2 in self.interaction_pairs 
                    if f1 in X.columns and f2 in X.columns]
        
        # Return empty DataFrame if no valid pairs
        if len(pairs) == 0:
            return pd.DataFrame(index=X.index)
        
        # Initialize DataFrame to store interaction features
        # Create column names for each interaction type (product, ratio, difference, sum)
        interaction_columns = []
        for feature1, feature2 in pairs:
            interaction_columns.extend([
                f'{feature1}_{feature2}_product',
                f'{feature1}_{feature2}_ratio',
                f'{feature1}_{feature2}_difference',
                f'{feature1}_{feature2}_sum'
            ])
        
        # Initialize DataFrame with same index as X, filled with NaN
        interaction_df = pd.DataFrame(
            index=X.index,
            columns=interaction_columns,
            dtype=float
        )
        
        for feature1, feature2 in pairs:
            interaction_df[f'{feature1}_{feature2}_product'] = X[feature1] * X[feature2]
            # Avoid division by zero: set ratio to NaN where denominator is zero
            interaction_df[f'{feature1}_{feature2}_ratio'] = np.where(
                X[feature2] != 0,
                X[feature1] / X[feature2],
                np.nan
            )
            interaction_df[f'{feature1}_{feature2}_difference'] = X[feature1] - X[feature2]
            interaction_df[f'{feature1}_{feature2}_sum'] = X[feature1] + X[feature2]
        return interaction_df
    
    def _handle_unseen_categories(self, feature: str, X: pd.DataFrame, 
                                  mapping: dict, encoder_info: dict, default_value: float) -> pd.Series:
        """Helper to handle unseen categories for mapping-based encodings"""
        encoded_values = X[feature].map(mapping)
        
        if encoded_values.isna().any():
            if self.handle_unseen == 'error':
                unseen = X[feature][encoded_values.isna()].unique()
                raise ValueError(f"Unseen categories in feature '{feature}': {unseen.tolist()}")
            elif self.handle_unseen == 'most_frequent':
                most_freq = encoder_info.get('most_frequent') or encoder_info.get('most_frequent_category')
                fill_value = mapping.get(most_freq, default_value) if most_freq else default_value
            else:  # 'ignore'
                fill_value = default_value
            encoded_values = encoded_values.fillna(fill_value)
        
        return encoded_values
    
    def _encode_categoricals(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                            is_fit: bool = False) -> pd.DataFrame:
        """Encode categorical features"""
        X = X.copy()
        if len(self.categorical_features_) == 0:
            return X
        
        encoded_dfs = []
        features_to_drop = []
        
        for feature in self.categorical_features_:
            if feature not in X.columns:
                continue
            
            if self.categorical_encoding == 'onehot':
                if is_fit:
                    encoded = pd.get_dummies(X[feature], prefix=feature, dummy_na=False)
                    self.categorical_encoders_[feature] = {
                        'type': 'onehot',
                        'categories': set(encoded.columns),
                        'most_frequent': X[feature].mode()[0] if not X[feature].mode().empty else None
                    }
                else:
                    encoder_info = self.categorical_encoders_.get(feature, {})
                    known_categories = encoder_info.get('categories', set())
                    
                    # Handle unseen categories
                    if self.handle_unseen == 'most_frequent':
                        most_freq = encoder_info.get('most_frequent')
                        if most_freq:
                            seen_cats = {col.replace(f'{feature}_', '') for col in known_categories}
                            X.loc[~X[feature].isin(seen_cats), feature] = most_freq
                    elif self.handle_unseen == 'error':
                        seen_cats = {col.replace(f'{feature}_', '') for col in known_categories}
                        unseen = X[feature][~X[feature].isin(seen_cats)].unique()
                        if len(unseen) > 0:
                            raise ValueError(f"Unseen categories in feature '{feature}': {unseen.tolist()}")
                    
                    encoded = pd.get_dummies(X[feature], prefix=feature, dummy_na=False)
                    # Ensure all known categories present, remove unknown if ignore
                    for cat_col in known_categories:
                        if cat_col not in encoded.columns:
                            encoded[cat_col] = 0
                    if self.handle_unseen == 'ignore':
                        encoded = encoded[[col for col in encoded.columns if col in known_categories]]
                
                encoded_dfs.append(encoded)
                features_to_drop.append(feature)
                
            elif self.categorical_encoding == 'target':
                if is_fit:
                    if y is None:
                        raise ValueError("Target encoding requires y (target variable) during fit")
                    temp_df = pd.DataFrame({feature: X[feature], 'target': y})
                    target_means = temp_df.groupby(feature)['target'].mean()
                    self.categorical_encoders_[feature] = {
                        'type': 'target',
                        'mapping': target_means.to_dict(),
                        'global_mean': y.mean(),
                        'most_frequent_category': X[feature].mode()[0] if not X[feature].mode().empty else None
                    }
                    X[feature] = X[feature].map(target_means).fillna(y.mean())
                else:
                    encoder_info = self.categorical_encoders_.get(feature, {})
                    X[feature] = self._handle_unseen_categories(
                        feature, X, encoder_info.get('mapping', {}), encoder_info,
                        encoder_info.get('global_mean', 0)
                    )
                
            elif self.categorical_encoding == 'frequency':
                if is_fit:
                    frequencies = X[feature].value_counts(normalize=True).to_dict()
                    self.categorical_encoders_[feature] = {
                        'type': 'frequency',
                        'mapping': frequencies,
                        'most_frequent': X[feature].mode()[0] if not X[feature].mode().empty else None
                    }
                    X[feature] = X[feature].map(frequencies)
                else:
                    encoder_info = self.categorical_encoders_.get(feature, {})
                    mapping = encoder_info.get('mapping', {})
                    default_freq = mapping.get(encoder_info.get('most_frequent'), 0)
                    X[feature] = self._handle_unseen_categories(
                        feature, X, mapping, encoder_info, default_freq
                    )
        
        X = X.drop(columns=features_to_drop, errors='ignore')
        if encoded_dfs:
            X = pd.concat([X] + encoded_dfs, axis=1)
        return X
    
    def _scale_features(self, X: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Scale numeric features"""
        # If no scaling requested, return X unchanged
        if self.scaling is None or self.scaling == 'none':
            return X
        
        X = X.copy()  # Work on a copy to avoid modifying original
        
        if is_fit:
            # During fit: identify and learn scaling parameters
            # Identify numeric columns to scale (exclude binary/one-hot columns)
            numeric_cols_to_scale = []
            for col in X.columns:
                if col not in self.numeric_features_ and X[col].dtype not in ['int64', 'float64']:
                    if not np.issubdtype(X[col].dtype, np.number):
                        continue
                # Skip binary columns (only 0 and 1) - these are likely one-hot encoded
                unique_vals = X[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
                    continue
                numeric_cols_to_scale.append(col)
            
            if len(numeric_cols_to_scale) == 0:
                return X  # No numeric features to scale
            
            # Store which columns to scale for transform phase
            self.scaler_columns_ = numeric_cols_to_scale
            
            # Create and fit scaler
            if self.scaling == 'standard':
                self.scaler_ = StandardScaler()
            elif self.scaling == 'minmax':
                self.scaler_ = MinMaxScaler()
            else:
                return X  # Unknown scaling method
            
            # Fit scaler on numeric columns
            self.scaler_.fit(X[numeric_cols_to_scale])
            
            # Apply scaling to training data
            X_scaled = self.scaler_.transform(X[numeric_cols_to_scale])
            X[numeric_cols_to_scale] = X_scaled
        else:
            # During transform: use stored scaler and columns
            if self.scaler_ is None:
                return X  # No scaler was fitted
            
            # Use the same columns that were used during fit
            numeric_cols_to_scale = [col for col in self.scaler_columns_ if col in X.columns]
            
            if len(numeric_cols_to_scale) == 0:
                return X  # None of the scaled columns exist in X
            
            # Apply scaling using stored scaler
            X_scaled = self.scaler_.transform(X[numeric_cols_to_scale])
            X[numeric_cols_to_scale] = X_scaled
        
        return X


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
        
        # Step 1: Remove highly correlated features if method is 'correlation' or if needed
        if self.method == 'correlation':
            X_filtered, _ = self._remove_correlated_features(X)
            # For correlation method, just select remaining features (up to k)
            self.selected_features_ = list(X_filtered.columns[:self.k])
            self.feature_importances_ = {f: 1.0 for f in self.selected_features_}
            return self
        
        # Step 2: Apply selection method
        # Convert to numpy arrays (handle both pandas and numpy inputs)
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        if self.method == 'mutual_info':
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X_array, y_array, random_state=RANDOM_STATE)
            feature_scores = dict(zip(X.columns, mi_scores))
            self.feature_importances_ = feature_scores
            
            # Select top k features or features above threshold
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            if self.threshold is not None:
                selected = [f for f, score in sorted_features if score >= self.threshold]
                self.selected_features_ = selected[:self.k] if len(selected) > self.k else selected
            else:
                self.selected_features_ = [f for f, _ in sorted_features[:self.k]]
                
        elif self.method == 'f_test':
            # Calculate F-test scores
            f_scores, _ = f_classif(X_array, y_array)
            feature_scores = dict(zip(X.columns, f_scores))
            self.feature_importances_ = feature_scores
            
            # Select top k features or features above threshold
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            if self.threshold is not None:
                selected = [f for f, score in sorted_features if score >= self.threshold]
                self.selected_features_ = selected[:self.k] if len(selected) > self.k else selected
            else:
                self.selected_features_ = [f for f, _ in sorted_features[:self.k]]
                
        elif self.method == 'rfe':
            # Use Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
            self.selector_ = RFE(estimator, n_features_to_select=self.k)
            self.selector_.fit(X_array, y_array)
            self.selected_features_ = X.columns[self.selector_.support_].tolist()
            # Store rankings as importances (lower rank = more important)
            self.feature_importances_ = dict(zip(X.columns, self.selector_.ranking_))
            
        else:
            # Default: select all features
            self.selected_features_ = list(X.columns)
            self.feature_importances_ = {f: 1.0 for f in X.columns}
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from data.
        
        Args:
            X: Data to transform
        
        Returns:
            DataFrame with selected features
        """
        if len(self.selected_features_) == 0:
            return pd.DataFrame(index=X.index)
        
        # Return only features that exist in both selected_features_ and X
        available_features = [f for f in self.selected_features_ if f in X.columns]
        return X[available_features] if available_features else pd.DataFrame(index=X.index)
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features, keeping the feature with higher variance from each pair."""
        corr_matrix = X.corr().abs()
        features_to_remove = set()
        
        # Find highly correlated pairs and mark lower-variance feature for removal
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= self.correlation_threshold:
                    feat1, feat2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    # Skip if already marked for removal
                    if feat1 in features_to_remove or feat2 in features_to_remove:
                        continue
                    # Remove feature with lower variance
                    if X[feat1].var() >= X[feat2].var():
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
        
        removed_features = list(features_to_remove)
        X_filtered = X.drop(columns=removed_features) if removed_features else X.copy()
        return X_filtered, removed_features
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """Get features ranked by importance"""
        if not self.feature_importances_:
            return pd.DataFrame(columns=['feature', 'importance'])
        
        # Create DataFrame with features and importance scores, sorted descending
        ranking_df = pd.DataFrame([
            {'feature': feat, 'importance': score}
            for feat, score in self.feature_importances_.items()
        ])
        ranking_df = ranking_df.sort_values('importance', ascending=False).reset_index(drop=True)
        return ranking_df


# Usage example
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    # Create synthetic dataset with missing values and categoricals
    np.random.seed(RANDOM_STATE)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                              n_redundant=5, random_state=RANDOM_STATE)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    
    # Add some missing values
    missing_indices = np.random.choice(X.index, size=100, replace=False)
    X.loc[missing_indices, 'feature_0'] = np.nan
    
    # Add categorical features
    X['category'] = np.random.choice(['A', 'B', 'C', 'D'], size=1000)
    X['category2'] = np.random.choice(['X', 'Y'], size=1000)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
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
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
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
