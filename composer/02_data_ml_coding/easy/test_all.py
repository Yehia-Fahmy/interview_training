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
                print(f"{Colors.RED}Invalid exercise number. Use: python test_all.py --exercise <1-4>{Colors.RESET}")
                return
    
    # Run tests
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

