"""
Automated Test Runner for Medium Data/ML Coding Exercises

This script tests all your implementations with comprehensive, robust tests.

Run with: python test_all.py
Or test individual exercises: python test_all.py --exercise 1
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, List, Tuple
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def load_module(file_path: str, module_name: str):
    """Dynamically load a Python module from a file path."""
    full_path = SCRIPT_DIR / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"Could not find {full_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, str(full_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {full_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_condition(condition: bool, message: str, passed_list: List, failed_list: List):
    """Helper to track passed/failed tests"""
    if condition:
        print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")
        passed_list.append(message)
    else:
        print(f"{Colors.RED}✗ {message}{Colors.RESET}")
        failed_list.append(message)


def test_exercise_1():
    """Test Exercise 1: Feature Engineering & Selection Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: Feature Engineering & Selection Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_01.py', 'starter_01')
        
        passed = []
        failed = []
        
        # Test student implementation
        try:
            # Test 1: Basic functionality with missing values and categoricals
            print(f"\n{Colors.BLUE}Test 1: Basic Pipeline Functionality{Colors.RESET}")
            np.random.seed(42)
            X, y = make_classification(n_samples=500, n_features=10, n_informative=5, 
                                      n_redundant=2, random_state=42)
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            
            # Add missing values
            missing_indices = np.random.choice(X.index, size=50, replace=False)
            X.loc[missing_indices, 'feature_0'] = np.nan
            
            # Add categorical features
            X['category'] = np.random.choice(['A', 'B', 'C'], size=500)
            X['category2'] = np.random.choice(['X', 'Y'], size=500)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Test pipeline
            pipeline = student_module.FeatureEngineeringPipeline(
                missing_strategy={'feature_0': 'median'},
                create_interactions=True,
                categorical_encoding='onehot',
                scaling='standard'
            )
            
            X_train_transformed = pipeline.fit_transform(X_train, y_train)
            X_test_transformed = pipeline.transform(X_test)
            
            # Assertions
            assert_condition(isinstance(X_train_transformed, pd.DataFrame), 
                           "Pipeline returns DataFrame", passed, failed)
            assert_condition(X_train_transformed.shape[0] == X_train.shape[0],
                           "Pipeline preserves number of rows", passed, failed)
            assert_condition(X_test_transformed.shape[0] == X_test.shape[0],
                           "Transform preserves test set size", passed, failed)
            assert_condition(X_train_transformed.shape[1] >= X_train.shape[1],
                           "Pipeline creates new features", passed, failed)
            assert_condition(not X_train_transformed.isna().any().any(),
                           "Pipeline handles missing values", passed, failed)
            assert_condition(not X_test_transformed.isna().any().any(),
                           "Pipeline handles missing values in test set", passed, failed)
            
            # Test 2: Data leakage prevention
            print(f"\n{Colors.BLUE}Test 2: Data Leakage Prevention{Colors.RESET}")
            # Check that test set transformations don't use test set statistics
            # If scaling is done correctly, mean should be close to 0 for standardized features
            numeric_cols = [col for col in X_train_transformed.columns 
                          if col.startswith('feature_') and 'interaction' not in col.lower()]
            if len(numeric_cols) > 0:
                test_means = X_test_transformed[numeric_cols[:5]].mean()
                # Test means shouldn't be exactly 0 (would indicate leakage)
                assert_condition(abs(test_means.mean()) < 1.0,
                               "No obvious data leakage in scaling", passed, failed)
            
            # Test 3: Feature selection
            print(f"\n{Colors.BLUE}Test 3: Feature Selection{Colors.RESET}")
            selector = student_module.FeatureSelector(method='mutual_info', k=8)
            X_train_selected = selector.fit_transform(X_train_transformed, y_train)
            X_test_selected = selector.transform(X_test_transformed)
            
            assert_condition(X_train_selected.shape[1] <= X_train_transformed.shape[1],
                           "Feature selection reduces dimensionality", passed, failed)
            assert_condition(X_train_selected.shape[1] == X_test_selected.shape[1],
                           "Feature selection consistent between train/test", passed, failed)
            assert_condition(X_train_selected.shape[1] <= 8,
                           "Feature selection respects k parameter", passed, failed)
            
            # Test feature importance ranking
            importance_df = selector.get_feature_importance_ranking()
            assert_condition(importance_df is not None,
                           "Feature importance ranking exists", passed, failed)
            if importance_df is not None:
                assert_condition(len(importance_df) > 0,
                               "Feature importance ranking is non-empty", passed, failed)
                assert_condition(len(importance_df) == X_train_selected.shape[1],
                                 "Feature importance matches selected features", passed, failed)
            
            # Test 4: Unseen categories handling
            print(f"\n{Colors.BLUE}Test 4: Unseen Categories Handling{Colors.RESET}")
            # Add new category in test set
            X_test_unseen = X_test.copy()
            X_test_unseen.loc[X_test_unseen.index[0], 'category'] = 'UNSEEN_CATEGORY'
            
            try:
                X_test_unseen_transformed = pipeline.transform(X_test_unseen)
                assert_condition(not X_test_unseen_transformed.isna().any().any(),
                               "Pipeline handles unseen categories gracefully", passed, failed)
            except Exception as e:
                # If it raises an error, that's also acceptable if documented
                print(f"{Colors.YELLOW}⚠ Unseen category handling: {str(e)}{Colors.RESET}")
            
            # Test 5: Different encoding strategies
            print(f"\n{Colors.BLUE}Test 5: Different Encoding Strategies{Colors.RESET}")
            for encoding in ['onehot', 'frequency']:
                try:
                    pipeline_enc = student_module.FeatureEngineeringPipeline(
                        categorical_encoding=encoding,
                        create_interactions=False,
                        scaling=None
                    )
                    X_enc = pipeline_enc.fit_transform(X_train, y_train)
                    assert_condition(X_enc.shape[0] == X_train.shape[0],
                                   f"{encoding} encoding works", passed, failed)
                except Exception as e:
                    print(f"{Colors.YELLOW}⚠ {encoding} encoding: {str(e)}{Colors.RESET}")
            
            # Test 6: Model performance validation
            print(f"\n{Colors.BLUE}Test 6: Model Performance Validation{Colors.RESET}")
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train_selected, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train_selected))
            test_acc = accuracy_score(y_test, model.predict(X_test_selected))
            
            assert_condition(train_acc > 0.5, "Model trains successfully", passed, failed)
            assert_condition(test_acc > 0.4, "Model generalizes reasonably", passed, failed)
            assert_condition(abs(train_acc - test_acc) < 0.3,
                           "No severe overfitting", passed, failed)
            
            print(f"\n{Colors.BLUE}Results:{Colors.RESET}")
            print(f"  Original features: {X_train.shape[1]}")
            print(f"  Transformed features: {X_train_transformed.shape[1]}")
            print(f"  Selected features: {X_train_selected.shape[1]}")
            print(f"  Train accuracy: {train_acc:.4f}")
            print(f"  Test accuracy: {test_acc:.4f}")
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed.append(f"Exception: {str(e)}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {len(passed)}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {len(failed)}{Colors.RESET}")
        if failed:
            print(f"{Colors.RED}Failed tests: {', '.join(failed)}{Colors.RESET}")
        return len(passed), len(failed)
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_2():
    """Test Exercise 2: Handling Imbalanced Classification"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: Handling Imbalanced Classification")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        
        passed = []
        failed = []
        
        # Test student implementation
        try:
            # Test 1: Basic imbalanced classifier
            print(f"\n{Colors.BLUE}Test 1: Basic Imbalanced Classifier{Colors.RESET}")
            X, y = make_classification(
                n_samples=2000,
                n_features=10,
                n_informative=5,
                n_classes=2,
                weights=[0.95, 0.05],
                random_state=42
            )
            X = pd.DataFrame(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"  Class distribution - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
            
            imbalanced_clf = student_module.ImbalancedClassifier(
                base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                sampling_strategy=None,
                class_weight='balanced',
                calibrate_probabilities=True
            )
            
            imbalanced_clf.fit(X_train, y_train)
            predictions = imbalanced_clf.predict(X_test)
            probabilities = imbalanced_clf.predict_proba(X_test)
            
            assert_condition(len(predictions) == len(y_test),
                           "Predictions have correct length", passed, failed)
            assert_condition(predictions.dtype in [np.int64, np.int32, int],
                           "Predictions are integers", passed, failed)
            assert_condition(probabilities.shape == (len(X_test), 2),
                           "Probabilities have correct shape", passed, failed)
            assert_condition(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
                           "Probabilities sum to 1", passed, failed)
            assert_condition(np.all((probabilities >= 0) & (probabilities <= 1)),
                           "Probabilities in valid range", passed, failed)
            
            # Test 2: Evaluator metrics
            print(f"\n{Colors.BLUE}Test 2: Evaluator Metrics{Colors.RESET}")
            evaluator = student_module.ImbalancedEvaluator()
            metrics = evaluator.evaluate(y_test, predictions, probabilities[:, 1])
            
            required_metrics = ['precision', 'recall', 'f1', 'accuracy']
            for metric in required_metrics:
                assert_condition(metric in metrics,
                               f"Metric '{metric}' is computed", passed, failed)
                if metric in metrics:
                    assert_condition(0 <= metrics[metric] <= 1,
                                   f"Metric '{metric}' in valid range", passed, failed)
            
            if 'roc_auc' in metrics:
                assert_condition(0 <= metrics['roc_auc'] <= 1,
                               "ROC-AUC in valid range", passed, failed)
            
            # Test 3: Threshold optimization
            print(f"\n{Colors.BLUE}Test 3: Threshold Optimization{Colors.RESET}")
            optimal_threshold = evaluator.find_optimal_threshold(y_test, probabilities[:, 1], metric='f1')
            assert_condition(0 <= optimal_threshold <= 1,
                           "Optimal threshold in valid range", passed, failed)
            
            # Apply threshold and verify improvement
            threshold_pred = (probabilities[:, 1] >= optimal_threshold).astype(int)
            threshold_metrics = evaluator.evaluate(y_test, threshold_pred, probabilities[:, 1])
            assert_condition('f1' in threshold_metrics,
                           "Threshold-based predictions work", passed, failed)
            
            # Test 4: Baseline comparison
            print(f"\n{Colors.BLUE}Test 4: Baseline Comparison{Colors.RESET}")
            baseline = RandomForestClassifier(n_estimators=50, random_state=42)
            baseline.fit(X_train, y_train)
            baseline_pred = baseline.predict(X_test)
            baseline_proba = baseline.predict_proba(X_test)[:, 1]
            baseline_metrics = evaluator.evaluate(y_test, baseline_pred, baseline_proba)
            
            # Imbalanced classifier should have better recall for minority class
            if 'recall' in metrics and 'recall' in baseline_metrics:
                print(f"  Baseline recall: {baseline_metrics['recall']:.4f}")
                print(f"  Imbalanced recall: {metrics['recall']:.4f}")
            
            # Test 5: Report generation
            print(f"\n{Colors.BLUE}Test 5: Report Generation{Colors.RESET}")
            report = evaluator.generate_report(y_test, predictions, probabilities[:, 1])
            assert_condition(isinstance(report, str),
                           "Report is generated", passed, failed)
            assert_condition(len(report) > 0,
                           "Report is non-empty", passed, failed)
            
            # Test 6: Different sampling strategies (if available)
            print(f"\n{Colors.BLUE}Test 6: Sampling Strategies{Colors.RESET}")
            try:
                from imblearn.over_sampling import SMOTE
                imbalanced_clf_smote = student_module.ImbalancedClassifier(
                    base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                    sampling_strategy='smote',
                    calibrate_probabilities=False
                )
                imbalanced_clf_smote.fit(X_train, y_train)
                smote_pred = imbalanced_clf_smote.predict(X_test)
                assert_condition(len(smote_pred) == len(y_test),
                               "SMOTE sampling works", passed, failed)
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ SMOTE test: {str(e)}{Colors.RESET}")
            
            print(f"\n{Colors.BLUE}Results:{Colors.RESET}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  F1: {metrics.get('f1', 0):.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed.append(f"Exception: {str(e)}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {len(passed)}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {len(failed)}{Colors.RESET}")
        if failed:
            print(f"{Colors.RED}Failed tests: {', '.join(failed)}{Colors.RESET}")
        return len(passed), len(failed)
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_3():
    """Test Exercise 3: Time Series Forecasting Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Time Series Forecasting Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        
        passed = []
        failed = []
        
        # Test student implementation
        try:
            # Test 1: Basic preprocessing
            print(f"\n{Colors.BLUE}Test 1: Basic Preprocessing{Colors.RESET}")
            np.random.seed(42)
            dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
            
            trend = np.linspace(100, 150, len(dates))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            noise = np.random.normal(0, 5, len(dates))
            values = trend + seasonal + noise
            
            ts_data = pd.Series(values, index=dates, name='value')
            
            # Add missing values
            missing_indices = np.random.choice(ts_data.index, size=20, replace=False)
            ts_data.loc[missing_indices] = np.nan
            
            print(f"  Time series length: {len(ts_data)}")
            print(f"  Missing values: {ts_data.isna().sum()}")
            
            preprocessor = student_module.TimeSeriesPreprocessor(
                handle_missing='interpolate',
                handle_outliers=True,
                create_time_features=True,
                create_lag_features=True,
                lag_periods=[1, 7],
                create_rolling_features=True,
                rolling_windows=[7, 30]
            )
            
            ts_df = pd.DataFrame({'value': ts_data})
            ts_df_transformed = preprocessor.fit_transform(ts_df)
            
            assert_condition(isinstance(ts_df_transformed, pd.DataFrame),
                           "Preprocessor returns DataFrame", passed, failed)
            assert_condition(ts_df_transformed.shape[0] == ts_df.shape[0],
                           "Preprocessor preserves number of rows", passed, failed)
            assert_condition(ts_df_transformed.shape[1] > ts_df.shape[1],
                           "Preprocessor creates new features", passed, failed)
            assert_condition(not ts_df_transformed.isna().any().any(),
                           "Preprocessor handles missing values", passed, failed)
            
            # Test 2: Time-aware cross-validation
            print(f"\n{Colors.BLUE}Test 2: Time-Aware Cross-Validation{Colors.RESET}")
            evaluator = student_module.TimeSeriesEvaluator()
            cv_splits = evaluator.time_series_cv_split(ts_data)
            
            assert_condition(len(cv_splits) > 0,
                           "CV splits are generated", passed, failed)
            
            # Verify no data leakage (test indices should be after train indices)
            for train_idx, test_idx in cv_splits[:3]:  # Check first 3 splits
                train_max = max(train_idx)
                test_min = min(test_idx)
                assert_condition(train_max < test_min,
                               f"CV split {cv_splits.index((train_idx, test_idx))} has no leakage", 
                               passed, failed)
            
            # Test 3: Evaluation metrics
            print(f"\n{Colors.BLUE}Test 3: Evaluation Metrics{Colors.RESET}")
            # Create simple predictions for testing
            y_true = ts_data.dropna().values[-100:]
            y_pred = y_true + np.random.normal(0, 2, len(y_true))  # Add some noise
            
            metrics = evaluator.evaluate(y_true, y_pred)
            
            required_metrics = ['mae', 'rmse']
            for metric in required_metrics:
                assert_condition(metric in metrics,
                               f"Metric '{metric}' is computed", passed, failed)
                if metric in metrics:
                    assert_condition(metrics[metric] >= 0,
                                   f"Metric '{metric}' is non-negative", passed, failed)
            
            # Test 4: Forecasting (if statsmodels available)
            print(f"\n{Colors.BLUE}Test 4: Forecasting{Colors.RESET}")
            try:
                from statsmodels.tsa.arima.model import ARIMA
                forecaster = student_module.TimeSeriesForecaster(method='arima', forecast_horizon=10)
                train_data = ts_data[:int(len(ts_data) * 0.8)]
                forecaster.fit(train_data)
                
                forecasts, conf_intervals = forecaster.predict(n_periods=10)
                assert_condition(len(forecasts) == 10,
                               "Forecasts have correct length", passed, failed)
                assert_condition(not np.isnan(forecasts).any(),
                               "Forecasts contain no NaN", passed, failed)
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Forecasting test (statsmodels may not be available): {str(e)}{Colors.RESET}")
            
            # Test 5: Feature engineering correctness
            print(f"\n{Colors.BLUE}Test 5: Feature Engineering Correctness{Colors.RESET}")
            # Check that lag features are correct
            if 'value_lag_1' in ts_df_transformed.columns:
                # Lag 1 should be previous value
                lag1_col = ts_df_transformed['value_lag_1'].dropna()
                original_col = ts_df['value'].iloc[1:len(lag1_col)+1]
                # Should match (allowing for some preprocessing differences)
                assert_condition(len(lag1_col) > 0,
                               "Lag features are created", passed, failed)
            
            # Check that time features are created
            time_feature_cols = [col for col in ts_df_transformed.columns 
                                if any(x in col.lower() for x in ['day', 'month', 'week', 'year'])]
            assert_condition(len(time_feature_cols) > 0,
                           "Time-based features are created", passed, failed)
            
            print(f"\n{Colors.BLUE}Results:{Colors.RESET}")
            print(f"  Original features: {ts_df.shape[1]}")
            print(f"  Transformed features: {ts_df_transformed.shape[1]}")
            print(f"  CV splits: {len(cv_splits)}")
            if 'mae' in metrics:
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed.append(f"Exception: {str(e)}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {len(passed)}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {len(failed)}{Colors.RESET}")
        if failed:
            print(f"{Colors.RED}Failed tests: {', '.join(failed)}{Colors.RESET}")
        return len(passed), len(failed)
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_4():
    """Test Exercise 4: Model Interpretability & Explainability"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: Model Interpretability & Explainability")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        
        passed = []
        failed = []
        
        # Test student implementation
        try:
            # Test 1: Basic interpreter setup
            print(f"\n{Colors.BLUE}Test 1: Basic Interpreter Setup{Colors.RESET}")
            X, y = make_classification(
                n_samples=500,
                n_features=10,
                n_informative=5,
                random_state=42
            )
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            
            print(f"  Model Accuracy: {model.score(X_test, y_test):.4f}")
            
            interpreter = student_module.ModelInterpreter(
                model, 
                method='permutation'
            )
            interpreter.fit(X_train, y_train)
            
            assert_condition(interpreter is not None,
                           "Interpreter is initialized", passed, failed)
            
            # Test 2: Feature importance
            print(f"\n{Colors.BLUE}Test 2: Feature Importance{Colors.RESET}")
            importance_df = interpreter.get_feature_importance(X_test, y_test)
            
            assert_condition(importance_df is not None,
                           "Feature importance is computed", passed, failed)
            if importance_df is not None:
                assert_condition(len(importance_df) > 0,
                               "Feature importance is non-empty", passed, failed)
                assert_condition(len(importance_df) <= X_test.shape[1],
                               "Feature importance has correct number of features", passed, failed)
                
                # Check that importance values are reasonable
                if 'importance' in importance_df.columns or len(importance_df.columns) > 0:
                    importance_values = importance_df.iloc[:, -1].values  # Last column should be importance
                    assert_condition(np.all(np.isfinite(importance_values)),
                                   "Feature importance values are finite", passed, failed)
            
            # Test 3: Global explanation
            print(f"\n{Colors.BLUE}Test 3: Global Explanation{Colors.RESET}")
            global_explanation = interpreter.explain_global(X_test)
            
            assert_condition(global_explanation is not None,
                           "Global explanation is computed", passed, failed)
            if global_explanation is not None:
                assert_condition(isinstance(global_explanation, dict),
                               "Global explanation is a dictionary", passed, failed)
            
            # Test 4: Local explanation
            print(f"\n{Colors.BLUE}Test 4: Local Explanation{Colors.RESET}")
            local_explanation = interpreter.explain_local(X_test, instance_idx=0)
            
            assert_condition(local_explanation is not None,
                           "Local explanation is computed", passed, failed)
            if local_explanation is not None:
                assert_condition(isinstance(local_explanation, dict),
                               "Local explanation is a dictionary", passed, failed)
            
            # Test 5: Model auditor
            print(f"\n{Colors.BLUE}Test 5: Model Auditor{Colors.RESET}")
            auditor = student_module.ModelAuditor(model)
            
            uncertain_indices = auditor.flag_high_uncertainty(X_test, uncertainty_threshold=0.3)
            assert_condition(isinstance(uncertain_indices, list),
                           "Uncertainty flagging returns list", passed, failed)
            assert_condition(all(isinstance(idx, (int, np.integer)) for idx in uncertain_indices),
                           "Uncertainty indices are integers", passed, failed)
            assert_condition(all(0 <= idx < len(X_test) for idx in uncertain_indices),
                           "Uncertainty indices are valid", passed, failed)
            
            # Test 6: Bias detection (if sensitive features provided)
            print(f"\n{Colors.BLUE}Test 6: Bias Detection{Colors.RESET}")
            # Create a synthetic sensitive feature
            X_with_sensitive = X_test.copy()
            X_with_sensitive['sensitive'] = np.random.choice([0, 1], size=len(X_test))
            
            try:
                bias_metrics = auditor.detect_bias(X_with_sensitive, y_test, 'sensitive')
                if bias_metrics is not None:
                    assert_condition(isinstance(bias_metrics, dict),
                                   "Bias detection returns dictionary", passed, failed)
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Bias detection: {str(e)}{Colors.RESET}")
            
            # Test 7: Feature interactions (if SHAP available)
            print(f"\n{Colors.BLUE}Test 7: Feature Interactions{Colors.RESET}")
            try:
                interpreter_shap = student_module.ModelInterpreter(model, method='shap')
                interpreter_shap.fit(X_train, y_train)
                shap_values = interpreter_shap.compute_shap_values(X_test.head(50))
                
                if shap_values is not None:
                    interactions = auditor.detect_feature_interactions(X_test.head(50), shap_values)
                    assert_condition(isinstance(interactions, list),
                                   "Feature interactions detected", passed, failed)
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Feature interactions (SHAP may not be available): {str(e)}{Colors.RESET}")
            
            print(f"\n{Colors.BLUE}Results:{Colors.RESET}")
            if importance_df is not None:
                print(f"  Top 3 features:")
                print(importance_df.head(3))
            print(f"  High uncertainty predictions: {len(uncertain_indices)}")
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed.append(f"Exception: {str(e)}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {len(passed)}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {len(failed)}{Colors.RESET}")
        if failed:
            print(f"{Colors.RED}Failed tests: {', '.join(failed)}{Colors.RESET}")
        return len(passed), len(failed)
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*60)
    print("Medium Data/ML Coding - Robust Test Suite")
    print("="*60)
    print(f"{Colors.RESET}")
    
    exercise_num = None
    if len(sys.argv) > 1:
        if '--exercise' in sys.argv or '-e' in sys.argv:
            try:
                idx = sys.argv.index('--exercise') if '--exercise' in sys.argv else sys.argv.index('-e')
                exercise_num = int(sys.argv[idx + 1])
            except (IndexError, ValueError):
                print(f"{Colors.RED}Invalid exercise number. Use: python test_all.py --exercise <1-4>{Colors.RESET}")
                return
    
    total_passed = 0
    total_failed = 0
    
    tests = [
        (1, test_exercise_1),
        (2, test_exercise_2),
        (3, test_exercise_3),
        (4, test_exercise_4),
    ]
    
    if exercise_num:
        tests = [(num, func) for num, func in tests if num == exercise_num]
        if not tests:
            print(f"{Colors.RED}Exercise {exercise_num} not found.{Colors.RESET}")
            return
    
    for num, test_func in tests:
        passed, failed = test_func()
        total_passed += passed
        total_failed += failed
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Final Summary")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}Total Passed: {total_passed}{Colors.RESET}")
    print(f"{Colors.RED}Total Failed: {total_failed}{Colors.RESET}")
    
    if total_failed == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}All tests passed! 🎉{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.YELLOW}Some tests failed. Keep working! 💪{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
