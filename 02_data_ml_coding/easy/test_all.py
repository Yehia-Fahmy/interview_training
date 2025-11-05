"""
Automated Test Runner for Easy Data/ML Coding Exercises

This script tests all your implementations and compares them against:
1. Expected/ideal outputs (from reference solutions)

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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


def test_exercise_1():
    """Test Exercise 1: Classification Model from Scratch"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: Classification Model from Scratch")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_01.py', 'starter_01')
        solution_module = load_module('solution_01.py', 'solution_01')
        
        # Generate test data
        X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                                   n_redundant=1, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            student_model = student_module.LogisticRegression(learning_rate=0.1, max_iter=1000)
            student_model.fit(X_train, y_train)
            student_pred = student_model.predict(X_test)
            student_accuracy = accuracy_score(y_test, student_pred)
            student_proba = student_model.predict_proba(X_test)
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Accuracy: {student_accuracy:.4f}")
            print(f"  Predictions shape: {student_pred.shape}")
            print(f"  Probabilities shape: {student_proba.shape}")
            print(f"  Final loss: {student_model.loss_history[-1]:.4f if student_model.loss_history else 'N/A'}")
            
            # Check if predictions are valid
            if student_pred.shape == y_test.shape and np.all((student_pred == 0) | (student_pred == 1)):
                print(f"{Colors.GREEN}✓ Predictions are valid{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Predictions are invalid{Colors.RESET}")
                failed += 1
            
            # Check if probabilities are valid
            if student_proba.shape == y_test.shape and np.all((student_proba >= 0) & (student_proba <= 1)):
                print(f"{Colors.GREEN}✓ Probabilities are valid{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Probabilities are invalid{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_model = solution_module.LogisticRegression(learning_rate=0.1, max_iter=1000)
            ref_model.fit(X_train, y_train)
            ref_pred = ref_model.predict(X_test)
            ref_accuracy = accuracy_score(y_test, ref_pred)
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Accuracy: {ref_accuracy:.4f}")
            print(f"  Final loss: {ref_model.loss_history[-1]:.4f}")
            
            # Compare accuracy (allow some tolerance)
            if 'student_accuracy' in locals():
                accuracy_diff = abs(student_accuracy - ref_accuracy)
                if accuracy_diff < 0.1:  # Allow 10% difference
                    print(f"{Colors.GREEN}✓ Accuracy is close to reference (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Accuracy differs significantly (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_2():
    """Test Exercise 2: Model Evaluation Metrics"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: Model Evaluation Metrics")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        solution_module = load_module('solution_02.py', 'solution_02')
        
        # Generate test data
        X, y = make_classification(n_samples=200, n_features=10, n_classes=3,
                                   n_informative=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            student_metrics = student_module.ClassificationMetrics(y_test, y_pred, y_proba)
            
            student_accuracy = student_metrics.accuracy()
            student_cm = student_metrics.confusion_matrix()
            student_precision = student_metrics.precision_per_class()
            student_recall = student_metrics.recall_per_class()
            student_f1 = student_metrics.f1_per_class()
            student_macro = student_metrics.macro_averaged_metrics()
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Accuracy: {student_accuracy:.4f}")
            print(f"  Confusion Matrix shape: {student_cm.shape}")
            print(f"  Precision per class: {student_precision}")
            print(f"  Macro-averaged F1: {student_macro.get('f1', 'N/A'):.4f if student_macro else 'N/A'}")
            
            # Validate metrics
            if 0 <= student_accuracy <= 1:
                print(f"{Colors.GREEN}✓ Accuracy is valid{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Accuracy is invalid{Colors.RESET}")
                failed += 1
            
            if student_cm.shape[0] == student_cm.shape[1] == len(np.unique(y_test)):
                print(f"{Colors.GREEN}✓ Confusion matrix shape is correct{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Confusion matrix shape is incorrect{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_metrics = solution_module.ClassificationMetrics(y_test, y_pred, y_proba)
            ref_accuracy = ref_metrics.accuracy()
            ref_macro = ref_metrics.macro_averaged_metrics()
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Accuracy: {ref_accuracy:.4f}")
            print(f"  Macro-averaged F1: {ref_macro['f1']:.4f}")
            
            # Compare metrics
            if 'student_accuracy' in locals():
                accuracy_diff = abs(student_accuracy - ref_accuracy)
                if accuracy_diff < 0.01:
                    print(f"{Colors.GREEN}✓ Accuracy matches reference (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Accuracy differs (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_3():
    """Test Exercise 3: Feature Engineering Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Feature Engineering Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        solution_module = load_module('solution_03.py', 'solution_03')
        
        # Create sample data
        data = pd.DataFrame({
            'age': [25, 30, None, 35, 40, None, 28, 45],
            'salary': [50000, 60000, 70000, None, 80000, 90000, 55000, 75000],
            'city': ['NYC', 'SF', None, 'NYC', 'SF', 'NYC', 'LA', 'NYC'],
            'department': ['Engineering', 'Sales', 'Engineering', None, 'Sales', 'Engineering', 'Marketing', 'Sales']
        })
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            student_pipeline = student_module.FeaturePipeline(
                numerical_features=['age', 'salary'],
                categorical_features=['city', 'department'],
                scaling=True
            )
            
            student_pipeline.fit(data)
            student_transformed = student_pipeline.transform(data)
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Transformed shape: {student_transformed.shape}")
            print(f"  Sample output:\n{student_transformed[:2]}")
            
            # Validate output
            if student_transformed.shape[0] == len(data):
                print(f"{Colors.GREEN}✓ Output shape is correct{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Output shape is incorrect{Colors.RESET}")
                failed += 1
            
            if not np.isnan(student_transformed).any():
                print(f"{Colors.GREEN}✓ No NaN values in output{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ NaN values found in output{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_pipeline = solution_module.FeaturePipeline(
                numerical_features=['age', 'salary'],
                categorical_features=['city', 'department'],
                scaling=True
            )
            
            ref_pipeline.fit(data)
            ref_transformed = ref_pipeline.transform(data)
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Transformed shape: {ref_transformed.shape}")
            
            # Compare shapes
            if 'student_transformed' in locals():
                if student_transformed.shape == ref_transformed.shape:
                    print(f"{Colors.GREEN}✓ Shape matches reference{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Shape differs (student: {student_transformed.shape}, ref: {ref_transformed.shape}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_4():
    """Test Exercise 4: End-to-End ML Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: End-to-End ML Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        solution_module = load_module('solution_04.py', 'solution_04')
        
        # Generate test data and save as CSV
        X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                                   n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        test_csv_path = SCRIPT_DIR / 'test_data.csv'
        df.to_csv(test_csv_path, index=False)
        
        passed = 0
        failed = 0
        
        config = {
            'model_type': 'random_forest',
            'model_params': {'n_estimators': 50, 'random_state': 42},
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Test student implementation
        try:
            student_pipeline = student_module.MLPipeline(config)
            
            # Test load_data
            student_df = student_pipeline.load_data(str(test_csv_path))
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Loaded data shape: {student_df.shape}")
            
            # Test split_data
            X_train, X_test, y_train, y_test = student_pipeline.split_data(student_df, 'target')
            print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            
            # Test train
            history = student_pipeline.train(X_train, y_train, X_test, y_test)
            print(f"  Training history: {history}")
            
            # Test evaluate
            metrics = student_pipeline.evaluate(X_test, y_test)
            print(f"  Test metrics: {metrics}")
            
            # Test predict
            predictions = student_pipeline.predict(X_test)
            print(f"  Predictions shape: {predictions.shape}")
            
            # Validate
            if student_df.shape[0] == len(df):
                print(f"{Colors.GREEN}✓ Data loading works{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Data loading failed{Colors.RESET}")
                failed += 1
            
            if student_pipeline.model is not None:
                print(f"{Colors.GREEN}✓ Model was trained{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Model was not trained{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_pipeline = solution_module.MLPipeline(config)
            ref_df = ref_pipeline.load_data(str(test_csv_path))
            X_train, X_test, y_train, y_test = ref_pipeline.split_data(ref_df, 'target')
            ref_history = ref_pipeline.train(X_train, y_train, X_test, y_test)
            ref_metrics = ref_pipeline.evaluate(X_test, y_test)
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Test accuracy: {ref_metrics.get('accuracy', 'N/A'):.4f if ref_metrics else 'N/A'}")
            
            # Compare metrics
            if 'metrics' in locals() and 'accuracy' in metrics:
                accuracy_diff = abs(metrics['accuracy'] - ref_metrics.get('accuracy', 0))
                if accuracy_diff < 0.1:
                    print(f"{Colors.GREEN}✓ Accuracy is close to reference (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Accuracy differs (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        # Cleanup
        if test_csv_path.exists():
            test_csv_path.unlink()
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_5():
    """Test Exercise 5: Data Cleaning and Preprocessing with Pandas"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 5: Data Cleaning and Preprocessing with Pandas")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_05.py', 'starter_05')
        solution_module = load_module('solution_05.py', 'solution_05')
        
        # Create messy sample data
        messy_data = pd.DataFrame({
            'id': [1, 2, 2, 3, 4, None, 5],
            'name': ['Alice', 'Bob', 'bob', 'Charlie', 'DAVE', '', 'Eve'],
            'age': [25, 30, 30, None, 150, 28, 35],
            'salary': [50000, '60000', 60000, None, 80000, 55000, 'invalid'],
            'date_joined': ['2020-01-15', '2020/02/20', 'invalid', '2020-03-10', None, '2020-04-01', '2020-05-15']
        })
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            cleaner = student_module.DataCleaner(handle_missing='mean', handle_duplicates=True)
            cleaned = cleaner.clean(messy_data, 
                                    date_columns=['date_joined'],
                                    numeric_columns=['age', 'salary'])
            
            report = cleaner.get_report()
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Cleaned shape: {cleaned.shape}")
            print(f"  Report keys: {list(report.keys())}")
            
            # Validate output
            if cleaned.shape[0] <= messy_data.shape[0]:
                print(f"{Colors.GREEN}✓ Shape is valid{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Shape is invalid{Colors.RESET}")
                failed += 1
            
            if not cleaned.isnull().all().any():
                print(f"{Colors.GREEN}✓ No all-NaN columns{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ All-NaN columns found{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_cleaner = solution_module.DataCleaner(handle_missing='mean', handle_duplicates=True)
            ref_cleaned = ref_cleaner.clean(messy_data, 
                                            date_columns=['date_joined'],
                                            numeric_columns=['age', 'salary'])
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Cleaned shape: {ref_cleaned.shape}")
            
            # Compare shapes
            if 'cleaned' in locals():
                if cleaned.shape == ref_cleaned.shape:
                    print(f"{Colors.GREEN}✓ Shape matches reference{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Shape differs (student: {cleaned.shape}, ref: {ref_cleaned.shape}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_6():
    """Test Exercise 6: Advanced Feature Engineering with Pandas and Scikit-learn"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 6: Advanced Feature Engineering with Pandas and Scikit-learn")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_06.py', 'starter_06')
        solution_module = load_module('solution_06.py', 'solution_06')
        
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
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            engineer = student_module.FeatureEngineer(
                date_columns=['transaction_date'],
                groupby_columns=['customer_id'],
                aggregate_columns=['amount'],
                create_interactions=True
            )
            
            features = engineer.fit_transform(data, target)
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Original shape: {data.shape}")
            print(f"  Engineered shape: {features.shape}")
            
            # Validate output
            if features.shape[1] >= data.shape[1]:
                print(f"{Colors.GREEN}✓ Features were created{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Features were not created{Colors.RESET}")
                failed += 1
            
            if features.shape[0] == data.shape[0]:
                print(f"{Colors.GREEN}✓ Row count preserved{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Row count changed{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_engineer = solution_module.FeatureEngineer(
                date_columns=['transaction_date'],
                groupby_columns=['customer_id'],
                aggregate_columns=['amount'],
                create_interactions=True
            )
            
            ref_features = ref_engineer.fit_transform(data, target)
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Engineered shape: {ref_features.shape}")
            
            # Compare feature counts
            if 'features' in locals():
                if features.shape[1] >= ref_features.shape[1] * 0.8:  # Allow some variance
                    print(f"{Colors.GREEN}✓ Feature count is reasonable{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Feature count differs significantly{Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_7():
    """Test Exercise 7: Model Training Pipeline with Scikit-learn"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 7: Model Training Pipeline with Scikit-learn")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_07.py', 'starter_07')
        solution_module = load_module('solution_07.py', 'solution_07')
        
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=500, n_features=10, 
                                   n_informative=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            trainer = student_module.ModelTrainer(
                task_type='classification',
                model_type='random_forest',
                model_params={'n_estimators': 50, 'random_state': 42}
            )
            
            results = trainer.train(X, pd.Series(y))
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Results keys: {list(results.keys()) if results else 'None'}")
            
            # Validate
            if results and 'test_metrics' in results:
                test_metrics = results['test_metrics']
                if 'accuracy' in test_metrics and 0 <= test_metrics['accuracy'] <= 1:
                    print(f"{Colors.GREEN}✓ Test metrics are valid{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.RED}✗ Test metrics are invalid{Colors.RESET}")
                    failed += 1
            else:
                print(f"{Colors.RED}✗ Results structure is incorrect{Colors.RESET}")
                failed += 1
            
            if trainer.pipeline is not None:
                print(f"{Colors.GREEN}✓ Pipeline was created{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Pipeline was not created{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 2
        
        # Test reference solution
        try:
            ref_trainer = solution_module.ModelTrainer(
                task_type='classification',
                model_type='random_forest',
                model_params={'n_estimators': 50, 'random_state': 42}
            )
            
            ref_results = ref_trainer.train(X, pd.Series(y))
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Test accuracy: {ref_results['test_metrics']['accuracy']:.4f}")
            
            # Compare accuracy if available
            if 'results' in locals() and 'test_metrics' in results:
                accuracy_diff = abs(results['test_metrics']['accuracy'] - ref_results['test_metrics']['accuracy'])
                if accuracy_diff < 0.15:  # Allow some variance
                    print(f"{Colors.GREEN}✓ Accuracy is close to reference (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ Accuracy differs significantly (diff: {accuracy_diff:.4f}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_8():
    """Test Exercise 8: Cross-validation and Hyperparameter Tuning"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 8: Cross-validation and Hyperparameter Tuning")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_08.py', 'starter_08')
        solution_module = load_module('solution_08.py', 'solution_08')
        
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=200, n_features=10, 
                                   n_informative=5, random_state=42)
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            tuner = student_module.HyperparameterTuner(
                task_type='classification',
                cv_folds=3,  # Reduced for speed
                random_state=42
            )
            
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5]
            }
            
            results = tuner.grid_search(X, pd.Series(y),
                                       RandomForestClassifier(random_state=42),
                                       param_grid)
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Results keys: {list(results.keys()) if results else 'None'}")
            
            # Validate
            if results and 'best_params' in results:
                print(f"{Colors.GREEN}✓ Grid search completed{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Grid search failed{Colors.RESET}")
                failed += 1
            
            if results and 'best_score' in results:
                if 0 <= results['best_score'] <= 1:
                    print(f"{Colors.GREEN}✓ Best score is valid{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.RED}✗ Best score is invalid{Colors.RESET}")
                    failed += 1
            else:
                print(f"{Colors.RED}✗ Best score not found{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 3
        
        # Test reference solution
        try:
            ref_tuner = solution_module.HyperparameterTuner(
                task_type='classification',
                cv_folds=3,
                random_state=42
            )
            
            ref_results = ref_tuner.grid_search(X, pd.Series(y),
                                               RandomForestClassifier(random_state=42),
                                               param_grid)
            
            print(f"\n{Colors.BLUE}Reference Solution:{Colors.RESET}")
            print(f"  Best CV score: {ref_results['best_score']:.4f}")
            
            # Compare scores if available
            if 'results' in locals() and 'best_score' in results:
                score_diff = abs(results['best_score'] - ref_results['best_score'])
                if score_diff < 0.15:  # Allow some variance
                    print(f"{Colors.GREEN}✓ CV score is close to reference (diff: {score_diff:.4f}){Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.YELLOW}⚠ CV score differs (diff: {score_diff:.4f}){Colors.RESET}")
                    failed += 1
        
        except Exception as e:
            print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*60)
    print("Easy Data/ML Coding - Automated Test Suite")
    print("="*60)
    print(f"{Colors.RESET}")
    
    # Check command line arguments
    exercise_num = None
    if len(sys.argv) > 1:
        if '--exercise' in sys.argv or '-e' in sys.argv:
            try:
                idx = sys.argv.index('--exercise') if '--exercise' in sys.argv else sys.argv.index('-e')
                exercise_num = int(sys.argv[idx + 1])
            except (IndexError, ValueError):
                print(f"{Colors.RED}Invalid exercise number. Use: python test_all.py --exercise <1-8>{Colors.RESET}")
                return
    
    # Run tests
    total_passed = 0
    total_failed = 0
    
    tests = [
        (1, test_exercise_1),
        (2, test_exercise_2),
        (3, test_exercise_3),
        (4, test_exercise_4),
        (5, test_exercise_5),
        (6, test_exercise_6),
        (7, test_exercise_7),
        (8, test_exercise_8),
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
    
    # Final summary
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

