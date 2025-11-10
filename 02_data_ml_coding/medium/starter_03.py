"""
Exercise 3: Time Series Forecasting Pipeline

Build a time series forecasting pipeline that handles multiple series,
missing values, feature engineering, and proper evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ARIMA and Exponential Smoothing will be limited.")


class TimeSeriesPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for time series data.
    
    Handles missing values, outliers, and creates time-based features.
    """
    
    def __init__(self,
                 handle_missing: str = 'interpolate',  # 'interpolate', 'forward_fill', 'backward_fill', 'mean'
                 handle_outliers: bool = True,
                 outlier_method: str = 'iqr',  # 'iqr', 'zscore'
                 create_time_features: bool = True,
                 create_lag_features: bool = True,
                 lag_periods: List[int] = None,
                 create_rolling_features: bool = True,
                 rolling_windows: List[int] = None):
        """
        Initialize preprocessor.
        
        Args:
            handle_missing: Method for handling missing values
            handle_outliers: Whether to detect and handle outliers
            outlier_method: Method for outlier detection
            create_time_features: Whether to create time-based features
            create_lag_features: Whether to create lag features
            lag_periods: List of lag periods (e.g., [1, 7, 30])
            create_rolling_features: Whether to create rolling statistics
            rolling_windows: List of window sizes for rolling features
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.create_time_features = create_time_features
        self.create_lag_features = create_lag_features
        self.lag_periods = lag_periods or [1, 7, 30]
        self.create_rolling_features = create_rolling_features
        self.rolling_windows = rolling_windows or [7, 30]
        
        self.outlier_thresholds_ = {}
        self.feature_names_ = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit preprocessor on training data.
        
        Args:
            X: Time series data with datetime index
            y: Target series (optional)
        
        Returns:
            self
        """
        # TODO: Implement fitting
        # 1. Learn outlier thresholds if handling outliers
        # 2. Store feature names
        # Note: Most preprocessing is stateless, but outlier detection needs thresholds
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform time series data.
        
        Args:
            X: Time series data with datetime index
        
        Returns:
            Transformed DataFrame with engineered features
        """
        X = X.copy()
        
        # 1. Handle missing values (initial cleanup)
        X = self._handle_missing_values(X)
        
        # 2. Handle outliers
        if self.handle_outliers:
            X = self._handle_outliers(X)
        
        # 3. Create time-based features (day of week, month, etc.)
        if self.create_time_features:
            X = self._create_time_features(X)
        
        # 4. Create lag features (creates NaNs at beginning)
        if self.create_lag_features:
            X = self._create_lag_features(X)
        
        # 5. Create rolling statistics (may have NaNs at beginning)
        if self.create_rolling_features:
            X = self._create_rolling_features(X)
        
        # 6. Handle missing values again (after lag/rolling features create NaNs)
        X = self._handle_missing_values(X)
        
        return X
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        X = X.copy()
        
        if self.handle_missing == 'interpolate':
            X = X.interpolate(method='linear')
        elif self.handle_missing == 'forward_fill':
            X = X.ffill()
        elif self.handle_missing == 'backward_fill':
            X = X.bfill()
        elif self.handle_missing == 'mean':
            X = X.fillna(X.mean())
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers"""        
        X = X.copy()
        
        for col in X.select_dtypes(include=[np.number]).columns:
            if self.outlier_method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
            elif self.outlier_method == 'zscore':
                mean = X[col].mean()
                std = X[col].std()
                if std > 0:
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    X[col] = X[col].clip(lower=lower_bound, upper=upper_bound)
        
        return X
    
    def _create_time_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""        
        X = X.copy()
        index = X.index
        
        if not isinstance(index, pd.DatetimeIndex):
            return X
        
        X['day_of_week'] = index.dayofweek
        X['day_of_month'] = index.day
        X['month'] = index.month
        X['quarter'] = index.quarter
        X['year'] = index.year
        X['is_weekend'] = (index.dayofweek >= 5).astype(int)
        X['is_month_start'] = index.is_month_start.astype(int)
        X['is_month_end'] = index.is_month_end.astype(int)
        
        # Cyclical encoding for periodic features
        X['day_of_week_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X['day_of_week_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        X['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        
        return X
    
    def _create_lag_features(self, X: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Create lag features"""        
        X = X.copy()
        
        # Determine which columns to create lags for
        if target_col and target_col in X.columns:
            cols_to_lag = [target_col]
        else:
            cols_to_lag = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create lag features for each column and lag period
        for col in cols_to_lag:
            for lag in self.lag_periods:
                X[f'{col}_lag_{lag}'] = X[col].shift(lag)
        
        return X
    
    def _create_rolling_features(self, X: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Create rolling window statistics"""        
        X = X.copy()
        
        # Determine which columns to create rolling features for
        if target_col and target_col in X.columns:
            cols_to_roll = [target_col]
        else:
            cols_to_roll = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create rolling statistics for each column and window size
        for col in cols_to_roll:
            for window in self.rolling_windows:
                rolling = X[col].rolling(window=window, min_periods=1)
                X[f'{col}_rolling_mean_{window}'] = rolling.mean()
                X[f'{col}_rolling_std_{window}'] = rolling.std()
                X[f'{col}_rolling_min_{window}'] = rolling.min()
                X[f'{col}_rolling_max_{window}'] = rolling.max()
            
            # Exponential weighted moving average
            X[f'{col}_ewm'] = X[col].ewm(span=7, adjust=False).mean()
        
        return X


class TimeSeriesForecaster:
    """
    Time series forecasting with multiple methods.
    
    Supports ARIMA, Exponential Smoothing, and ML-based forecasting.
    """
    
    def __init__(self, 
                 method: str = 'arima',  # 'arima', 'exponential_smoothing', 'ml'
                 model_params: Optional[Dict] = None,
                 forecast_horizon: int = 1):
        """
        Initialize forecaster.
        
        Args:
            method: Forecasting method
            model_params: Parameters for the forecasting model
            forecast_horizon: Number of steps ahead to forecast
        """
        self.method = method
        self.model_params = model_params or {}
        self.forecast_horizon = forecast_horizon
        
        self.model_ = None
        self.fitted_ = False
    
    def fit(self, y: pd.Series):
        """
        Fit forecasting model.
        
        Args:
            y: Time series to fit (with datetime index)
        
        Returns:
            self
        """
        # TODO: Implement model fitting
        # ARIMA: Fit ARIMA model (need to determine order)
        # Exponential Smoothing: Fit ETS model
        # ML: This would require feature engineering first (use preprocessor)
        pass
    
    def predict(self, n_periods: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts.
        
        Args:
            n_periods: Number of periods to forecast (defaults to forecast_horizon)
        
        Returns:
            (forecasts, confidence_intervals) tuple
        """
        # TODO: Generate forecasts
        # Return point forecasts and confidence intervals if available
        pass
    
    def _fit_arima(self, y: pd.Series):
        """Fit ARIMA model"""
        # TODO: Implement ARIMA fitting
        # Auto-select order or use provided order
        # Handle non-stationarity (differencing)
        pass
    
    def _fit_exponential_smoothing(self, y: pd.Series):
        """Fit Exponential Smoothing model"""
        # TODO: Implement ETS fitting
        # Detect trend and seasonality
        pass


class TimeSeriesEvaluator:
    """
    Evaluator for time series forecasting with time-aware cross-validation.
    """
    
    def __init__(self, 
                 cv_method: str = 'walk_forward',  # 'walk_forward', 'expanding_window'
                 n_splits: int = 5,
                 test_size: Optional[int] = None):
        """
        Initialize evaluator.
        
        Args:
            cv_method: Cross-validation method
            n_splits: Number of CV splits
            test_size: Size of test set in each split
        """
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.test_size = test_size
    
    def time_series_cv_split(self, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-aware cross-validation splits.
        
        Args:
            y: Time series data
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        # TODO: Implement time-aware CV
        # Walk-forward: Fixed train size, moving test window
        # Expanding window: Growing train size, moving test window
        # CRITICAL: Never use future data to predict past!
        pass
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        # TODO: Compute time series metrics
        # MAE, RMSE, MAPE, MASE (Mean Absolute Scaled Error)
        # Handle division by zero in MAPE
        pass
    
    def cross_validate(self, forecaster: TimeSeriesForecaster, 
                      y: pd.Series) -> Dict[str, List[float]]:
        """
        Perform time-aware cross-validation.
        
        Args:
            forecaster: Fitted forecaster
            y: Time series data
        
        Returns:
            Dictionary of metrics across CV folds
        """
        # TODO: Implement cross-validation
        # Split data using time_series_cv_split
        # Train and evaluate on each fold
        # Return metrics for each fold
        pass


# Usage example
if __name__ == "__main__":
    # Create synthetic time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Create series with trend and seasonality
    trend = np.linspace(100, 150, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + noise
    
    ts_data = pd.Series(values, index=dates, name='value')
    
    # Add some missing values
    missing_indices = np.random.choice(ts_data.index, size=50, replace=False)
    ts_data.loc[missing_indices] = np.nan
    
    print(f"Time series length: {len(ts_data)}")
    print(f"Missing values: {ts_data.isna().sum()}")
    
    # Preprocess
    preprocessor = TimeSeriesPreprocessor(
        handle_missing='interpolate',
        handle_outliers=True,
        create_time_features=True,
        create_lag_features=True,
        lag_periods=[1, 7, 30],
        create_rolling_features=True,
        rolling_windows=[7, 30]
    )
    
    # Convert to DataFrame for preprocessing
    ts_df = pd.DataFrame({'value': ts_data})
    ts_df_transformed = preprocessor.fit_transform(ts_df)
    
    print(f"\nOriginal features: {ts_df.shape[1]}")
    print(f"Transformed features: {ts_df_transformed.shape[1]}")
    
    # Split into train/test (last 20% for testing)
    split_idx = int(len(ts_data) * 0.8)
    train_data = ts_data[:split_idx]
    test_data = ts_data[split_idx:]
    
    # Forecast
    if STATSMODELS_AVAILABLE:
        forecaster = TimeSeriesForecaster(method='arima', forecast_horizon=len(test_data))
        forecaster.fit(train_data)
        forecasts, conf_intervals = forecaster.predict(n_periods=len(test_data))
        
        # Evaluate
        evaluator = TimeSeriesEvaluator()
        metrics = evaluator.evaluate(test_data.values, forecasts)
        
        print("\n=== Forecasting Results ===")
        print(f"MAE: {metrics.get('mae', 0):.4f}")
        print(f"RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"MAPE: {metrics.get('mape', 0):.4f}%")
    else:
        print("\nStatsmodels not available. Install with: pip install statsmodels")
