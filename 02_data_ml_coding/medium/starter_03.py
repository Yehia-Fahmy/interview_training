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
        # 1. Learn outlier thresholds if handling outliers
        if self.handle_outliers:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.outlier_method == 'iqr':
                    Q1 = X[col].quantile(0.25)
                    Q3 = X[col].quantile(0.75)
                    IQR = Q3 - Q1
                    # Handle edge case where IQR is 0 (constant values)
                    if IQR == 0:
                        self.outlier_thresholds_[col] = {
                            'lower': Q1,
                            'upper': Q3
                        }
                    else:
                        self.outlier_thresholds_[col] = {
                            'lower': Q1 - 1.5 * IQR,
                            'upper': Q3 + 1.5 * IQR
                        }
                elif self.outlier_method == 'zscore':
                    mean = X[col].mean()
                    std = X[col].std()
                    if std > 0:
                        self.outlier_thresholds_[col] = {
                            'lower': mean - 3 * std,
                            'upper': mean + 3 * std
                        }
                    else:
                        self.outlier_thresholds_[col] = {
                            'lower': mean,
                            'upper': mean
                        }
        
        # 2. Store original feature names
        self.feature_names_ = X.columns.tolist()
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform time series data.
        
        Args:
            X: Time series data with datetime index
        
        Returns:
            Transformed DataFrame with engineered features
        """
        if X.empty:
            return X.copy()
        
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
        # Use forward fill for lag/rolling NaNs, then apply configured method
        X = X.ffill().bfill()  # Handle NaNs from lag/rolling first
        X = self._handle_missing_values(X)  # Apply configured method for any remaining
        
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
            # Only compute mean for original numeric columns (exclude time features)
            original_numeric_cols = [col for col in self.feature_names_ 
                                     if col in X.columns and pd.api.types.is_numeric_dtype(X[col])]
            if original_numeric_cols:
                means = X[original_numeric_cols].mean()
                X[original_numeric_cols] = X[original_numeric_cols].fillna(means)
            # For other columns (time features, lag features, etc.), use forward fill
            X = X.ffill().bfill()
        
        return X
    
    def _handle_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using thresholds learned during fit"""
        X = X.copy()
        
        # Use stored thresholds from fit
        for col in X.select_dtypes(include=[np.number]).columns:
            if col in self.outlier_thresholds_:
                thresholds = self.outlier_thresholds_[col]
                X[col] = X[col].clip(lower=thresholds['lower'], upper=thresholds['upper'])
            # Skip columns that weren't seen during fit (new columns won't have thresholds)
        
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
            # Exclude time features and cyclical encodings from lagging
            time_feature_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                                'is_weekend', 'is_month_start', 'is_month_end',
                                'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_lag = [col for col in numeric_cols if col not in time_feature_cols]
        
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
            # Exclude time features and cyclical encodings from rolling stats
            time_feature_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                                'is_weekend', 'is_month_start', 'is_month_end',
                                'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_roll = [col for col in numeric_cols if col not in time_feature_cols]
        
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
        if self.method == 'arima':
            self._fit_arima(y)
        elif self.method == 'exponential_smoothing':
            self._fit_exponential_smoothing(y)
        elif self.method == 'ml':
            raise NotImplementedError("ML-based forecasting requires feature engineering. Use preprocessor first.")
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return self
    
    def predict(self, n_periods: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate forecasts.
        
        Args:
            n_periods: Number of periods to forecast (defaults to forecast_horizon)
        
        Returns:
            (forecasts, confidence_intervals) tuple
            - forecasts: array of point forecasts
            - confidence_intervals: array of (lower, upper) bounds, or zeros if not available
        """
        if not self.fitted_ or self.model_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        n_periods = n_periods or self.forecast_horizon
        
        if self.method == 'arima':
            # ARIMA provides forecasts and confidence intervals
            forecast_result = self.model_.get_forecast(steps=n_periods)
            forecasts = forecast_result.predicted_mean.values
            conf_int = forecast_result.conf_int().values
        elif self.method == 'exponential_smoothing':
            # Exponential Smoothing provides forecasts
            forecasts = self.model_.forecast(steps=n_periods).values
            # Try to get confidence intervals if available
            try:
                forecast_result = self.model_.get_prediction(start=len(self.model_.fittedvalues), 
                                                             end=len(self.model_.fittedvalues) + n_periods - 1)
                conf_int = forecast_result.conf_int().values
            except:
                # If confidence intervals not available, return zeros
                conf_int = np.zeros((n_periods, 2))
        else:
            raise ValueError(f"Prediction not implemented for method: {self.method}")
        
        return forecasts, conf_int
    
    def _fit_arima(self, y: pd.Series):
        """Fit ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA. Install with: pip install statsmodels")
        
        # Get ARIMA order from model_params, or use auto-selection
        order = self.model_params.get('order', None)
        
        if order is None:
            # Auto-select order: try common combinations and select best AIC
            # For simplicity, use a reasonable default or simple grid search
            # Common orders: (1,1,1), (2,1,2), (0,1,1)
            orders_to_try = [
                (1, 1, 1),  # Most common default
                (2, 1, 2),
                (0, 1, 1),  # Simple differencing + MA
                (1, 1, 0),  # AR + differencing
                (0, 1, 0),  # Simple random walk
            ]
            
            best_aic = np.inf
            best_order = (1, 1, 1)
            best_model = None
            
            for order_candidate in orders_to_try:
                try:
                    model = ARIMA(y, order=order_candidate)
                    fitted_model = model.fit()
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = order_candidate
                        best_model = fitted_model
                except:
                    continue
            
            if best_model is None:
                # Fallback to default if all fail
                order = (1, 1, 1)
                self.model_ = ARIMA(y, order=order).fit()
            else:
                self.model_ = best_model
                self.model_params['order'] = best_order
        else:
            # Use provided order
            self.model_ = ARIMA(y, order=order).fit()
        
        self.fitted_ = True
    
    def _fit_exponential_smoothing(self, y: pd.Series):
        """Fit Exponential Smoothing model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for Exponential Smoothing. Install with: pip install statsmodels")
        
        # Get parameters from model_params or use defaults
        trend = self.model_params.get('trend', 'add')  # 'add', 'mul', or None
        seasonal = self.model_params.get('seasonal', None)  # 'add', 'mul', or None
        seasonal_periods = self.model_params.get('seasonal_periods', None)
        
        # Auto-detect seasonal period from data frequency if not provided
        if seasonal_periods is None and seasonal is not None:
            if isinstance(y.index, pd.DatetimeIndex):
                freq = pd.infer_freq(y.index)
                if freq:
                    # Map common frequencies to seasonal periods
                    freq_map = {
                        'D': 7,      # Daily -> weekly seasonality
                        'W': 52,     # Weekly -> yearly
                        'M': 12,     # Monthly -> yearly
                        'Q': 4,      # Quarterly -> yearly
                        'H': 24,     # Hourly -> daily
                    }
                    seasonal_periods = freq_map.get(freq[:1], None)
        
        # Fit model
        self.model_ = ExponentialSmoothing(
            y,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        ).fit()
        
        self.fitted_ = True


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
        n = len(y)
        splits = []
        
        if self.test_size is None:
            test_size = max(1, n // (self.n_splits + 1))
        else:
            test_size = self.test_size
        
        if self.cv_method == 'walk_forward':
            # Fixed train size, moving test window
            train_size = n - test_size * self.n_splits
            if train_size < test_size:
                train_size = test_size
            
            for i in range(self.n_splits):
                train_end = train_size + i * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n)
                
                if test_start >= n:
                    break
                
                train_indices = np.arange(train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))
        
        elif self.cv_method == 'expanding_window':
            # Growing train size, moving test window
            initial_train_size = max(test_size, n - test_size * self.n_splits)
            
            for i in range(self.n_splits):
                train_end = initial_train_size + i * test_size
                test_start = train_end
                test_end = min(test_start + test_size, n)
                
                if test_start >= n:
                    break
                
                train_indices = np.arange(train_end)
                test_indices = np.arange(test_start, test_end)
                splits.append((train_indices, test_indices))
        
        return splits
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Remove NaN values from both arrays (align indices)
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'mase': np.nan}
        
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # MAPE (Mean Absolute Percentage Error)
        mask_nonzero = y_true != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        else:
            mape = np.nan
        
        # MASE (Mean Absolute Scaled Error)
        # Uses naive forecast (shift by 1) as baseline
        if len(y_true) > 1:
            naive_forecast = np.abs(np.diff(y_true))
            if naive_forecast.sum() > 0:
                mase = np.mean(np.abs(y_true - y_pred)) / np.mean(naive_forecast)
            else:
                mase = np.nan
        else:
            mase = np.nan
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mase': mase
        }
    
    def cross_validate(self, forecaster: TimeSeriesForecaster, 
                      y: pd.Series) -> Dict[str, List[float]]:
        """
        Perform time-aware cross-validation.
        
        Args:
            forecaster: Forecaster instance (will be fitted on each fold)
            y: Time series data
        
        Returns:
            Dictionary of metrics across CV folds
        """
        splits = self.time_series_cv_split(y)
        
        all_metrics = {'mae': [], 'rmse': [], 'mape': [], 'mase': []}
        
        for train_indices, test_indices in splits:
            # Split data
            y_train = y.iloc[train_indices]
            y_test = y.iloc[test_indices]
            
            # Fit forecaster on training data
            forecaster.fit(y_train)
            
            # Predict on test data
            n_periods = len(test_indices)
            y_pred, _ = forecaster.predict(n_periods=n_periods)
            
            # Evaluate
            metrics = self.evaluate(y_test.values, y_pred)
            
            # Store metrics
            for key in all_metrics:
                value = metrics.get(key, np.nan)
                if not np.isnan(value):
                    all_metrics[key].append(value)
        
        return all_metrics


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
        # Clean train_data (remove NaNs) - ARIMA can't handle missing values
        train_data_clean = train_data.dropna()
        # Preserve frequency information for statsmodels
        # After dropna(), we need to recreate index with frequency if possible
        if isinstance(train_data_clean.index, pd.DatetimeIndex):
            freq = train_data.index.freq or pd.infer_freq(train_data.index)
            if freq and train_data_clean.index.freq is None:
                # Recreate index with frequency - use integer index if frequency doesn't match
                try:
                    # Try to set frequency - if it fails, use integer index
                    train_data_clean = train_data_clean.copy()
                    train_data_clean.index = pd.DatetimeIndex(train_data_clean.index, freq=freq)
                except (ValueError, TypeError):
                    # If frequency can't be set (due to gaps), use integer index
                    # ARIMA will work fine with integer index
                    pass
        
        if len(train_data_clean) == 0:
            print("Error: No valid training data after removing NaNs")
        else:
            forecaster = TimeSeriesForecaster(method='arima', forecast_horizon=len(test_data))
            forecaster.fit(train_data_clean)
            forecasts, conf_intervals = forecaster.predict(n_periods=len(test_data))
            
            # Evaluate
            evaluator = TimeSeriesEvaluator()
            metrics = evaluator.evaluate(test_data.values, forecasts)
            
            print("\n=== Forecasting Results ===")
            print(f"MAE: {metrics.get('mae', np.nan):.4f}" if not np.isnan(metrics.get('mae', np.nan)) else "MAE: nan")
            print(f"RMSE: {metrics.get('rmse', np.nan):.4f}" if not np.isnan(metrics.get('rmse', np.nan)) else "RMSE: nan")
            print(f"MAPE: {metrics.get('mape', np.nan):.4f}%" if not np.isnan(metrics.get('mape', np.nan)) else "MAPE: nan%")
    else:
        print("\nStatsmodels not available. Install with: pip install statsmodels")
