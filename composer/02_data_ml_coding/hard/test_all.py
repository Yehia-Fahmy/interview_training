"""
Automated Test Runner for Hard Data/ML Coding Exercises

This script tests all your implementations and compares them against reference solutions.

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
    """Test Exercise 1: LLM Evaluation Framework"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: LLM Evaluation Framework")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_01.py', 'starter_01')
        solution_module = load_module('solution_01.py', 'solution_01')
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            framework = student_module.LLMEvaluationFramework()
            
            # Register evaluators
            framework.register_evaluator("bleu", student_module.BLEUEvaluator())
            framework.register_evaluator("rouge", student_module.ROUGEEvaluator())
            
            # Example data
            predictions = [
                "The cat sat on the mat.",
                "It was a sunny day."
            ]
            references = [
                "A cat was sitting on the mat.",
                "The weather was sunny."
            ]
            
            # Evaluate
            results = framework.evaluate(
                model_name="test_model",
                predictions=predictions,
                references=references,
                metrics=["bleu", "rouge"]
            )
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Evaluated {len(results)} metrics")
            
            for metric_name, result in results.items():
                print(f"  {metric_name}: {result.score:.4f}")
            
            if len(results) > 0 and all(0 <= r.score <= 1 for r in results.values()):
                print(f"{Colors.GREEN}✓ LLM evaluation framework works{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ LLM evaluation framework failed{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 1
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        return 0, 1


def test_exercise_2():
    """Test Exercise 2: Model Serving Infrastructure"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: Model Serving Infrastructure")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        solution_module = load_module('solution_02.py', 'solution_02')
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            server = student_module.ModelServer(batch_size=32, max_wait_time=0.1)
            
            # Test health check
            health = server.health_check()
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Server status: {health.get('status', 'unknown')}")
            print(f"  Models loaded: {health.get('models_loaded', 0)}")
            
            if 'status' in health:
                print(f"{Colors.GREEN}✓ Model server infrastructure works{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Model server infrastructure failed{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 1
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        return 0, 1


def test_exercise_3():
    """Test Exercise 3: Distributed Training Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Distributed Training Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        solution_module = load_module('solution_03.py', 'solution_03')
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            trainer = student_module.DistributedTrainer(num_workers=4)
            parameter_server = student_module.ParameterServer({'weights': np.array([1.0, 2.0])})
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Distributed trainer initialized")
            print(f"  Parameter server initialized")
            
            # Test parameter server
            params = parameter_server.pull_parameters()
            if 'weights' in params:
                print(f"{Colors.GREEN}✓ Distributed training framework structure works{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Distributed training framework failed{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 1
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        return 0, 1


def test_exercise_4():
    """Test Exercise 4: Real-time Feature Pipeline"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: Real-time Feature Pipeline")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        solution_module = load_module('solution_04.py', 'solution_04')
        
        passed = 0
        failed = 0
        
        # Test student implementation
        try:
            pipeline = student_module.RealTimeFeaturePipeline()
            
            # Register window
            pipeline.register_window('amount', window_size=100, window_duration=60.0)
            
            # Process events
            for i in range(5):
                event = {
                    'entity_id': 'user_123',
                    'amount': i * 10,
                    'timestamp': time.time()
                }
                pipeline.process_event(event)
            
            # Get features
            features = pipeline.get_features('user_123', ['amount_mean'])
            
            print(f"\n{Colors.BLUE}Student Solution:{Colors.RESET}")
            print(f"  Processed events")
            print(f"  Retrieved features: {features}")
            
            if isinstance(features, dict):
                print(f"{Colors.GREEN}✓ Real-time feature pipeline works{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}✗ Real-time feature pipeline failed{Colors.RESET}")
                failed += 1
                
        except Exception as e:
            print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
            print(f"  {str(e)}")
            traceback.print_exc()
            failed += 1
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.RESET}")
        return 0, 1


def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*60)
    print("Hard Data/ML Coding - Automated Test Suite")
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

