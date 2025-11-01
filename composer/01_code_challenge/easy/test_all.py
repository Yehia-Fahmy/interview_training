"""
Automated Test Runner for Easy Code Challenge Exercises

This script tests all your implementations and compares them against:
1. Expected/ideal outputs (from reference solutions)
2. Inefficient "current implementations" mentioned in the problem descriptions

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
    # Resolve path relative to script directory
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
    """Test Exercise 1: Memory-Efficient List Operations"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: Memory-Efficient List Operations")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        # Load student implementation
        student_module = load_module('starter_01.py', 'starter_01')
        # Load reference solution
        solution_module = load_module('solution_01.py', 'solution_01')
        
        # Test cases
        test_cases = [
            ([1, 2, 2, 3, 4, 4, 5, 6, 7, 8], [4, 16, 36, 64]),
            ([1, 3, 5, 7], []),  # No even numbers
            ([2, 2, 2], [4]),  # All same even
            ([10, 20, 10, 30, 20, 40], [100, 400, 900, 1600]),
            (list(range(1, 11)), [4, 16, 36, 64, 100]),
        ]
        
        passed = 0
        failed = 0
        
        for i, (input_data, expected) in enumerate(test_cases, 1):
            print(f"\n{Colors.BLUE}Test Case {i}:{Colors.RESET} input={input_data}")
            
            # Test student implementation
            try:
                student_result = student_module.process_numbers_optimized(input_data)
                student_sorted = sorted(student_result)
                expected_sorted = sorted(expected)
                
                if student_sorted == expected_sorted:
                    print(f"{Colors.GREEN}✓ Student solution: PASSED{Colors.RESET}")
                    print(f"  Result: {student_result}")
                    passed += 1
                else:
                    print(f"{Colors.RED}✗ Student solution: FAILED{Colors.RESET}")
                    print(f"  Expected: {expected}")
                    print(f"  Got: {student_result}")
                    failed += 1
            except Exception as e:
                print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
                print(f"  {str(e)}")
                failed += 1
                continue
            
            # Test inefficient implementation for comparison
            try:
                inefficient_result = student_module.process_numbers(input_data)
                inefficient_sorted = sorted(inefficient_result)
                
                if inefficient_sorted == expected_sorted:
                    print(f"{Colors.YELLOW}  Inefficient implementation: Correct result, but inefficient{Colors.RESET}")
                else:
                    print(f"{Colors.RED}  Inefficient implementation: Unexpected result{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}  Inefficient implementation: Error (expected){Colors.RESET}")
            
            # Test reference solution
            try:
                ref_result = solution_module.process_numbers_optimized(input_data)
                ref_sorted = sorted(ref_result)
                
                if ref_sorted == expected_sorted:
                    print(f"{Colors.GREEN}  Reference solution: PASSED{Colors.RESET}")
                else:
                    print(f"{Colors.RED}  Reference solution: UNEXPECTED RESULT{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        # Performance comparison with larger data
        print(f"\n{Colors.BLUE}Performance Test (Large Dataset):{Colors.RESET}")
        large_data = list(range(100000)) + [99999, 99998]  # Some duplicates
        
        try:
            start = time.time()
            student_large = student_module.process_numbers_optimized(large_data)
            student_time = time.time() - start
            print(f"{Colors.CYAN}  Student solution: {student_time:.4f}s{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  Student solution: ERROR - {str(e)}{Colors.RESET}")
            student_time = None
        
        try:
            start = time.time()
            inefficient_large = student_module.process_numbers(large_data)
            inefficient_time = time.time() - start
            print(f"{Colors.YELLOW}  Inefficient solution: {inefficient_time:.4f}s{Colors.RESET}")
            
            if student_time is not None:
                speedup = inefficient_time / student_time if student_time > 0 else 0
                print(f"{Colors.CYAN}  Speedup: {speedup:.2f}x{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}  Inefficient solution: ERROR (expected for large data){Colors.RESET}")
        
        # Summary
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_2():
    """Test Exercise 2: String Builder Optimization"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: String Builder Optimization")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        solution_module = load_module('solution_02.py', 'solution_02')
        
        test_cases = [
            (["Hello", " ", "World", "!"], "Hello World!"),
            (["a", "b", "c"], "abc"),
            (["test"], "test"),
            ([], ""),
            (["", "", "a"], "a"),
        ]
        
        passed = 0
        failed = 0
        
        for i, (input_data, expected) in enumerate(test_cases, 1):
            print(f"\n{Colors.BLUE}Test Case {i}:{Colors.RESET} input={input_data[:3]}..." if len(input_data) > 3 else f"{Colors.BLUE}Test Case {i}:{Colors.RESET} input={input_data}")
            
            # Test student implementation
            try:
                student_result = student_module.build_string_optimized(input_data)
                
                if student_result == expected:
                    print(f"{Colors.GREEN}✓ Student solution: PASSED{Colors.RESET}")
                    print(f"  Result: '{student_result}'")
                    passed += 1
                else:
                    print(f"{Colors.RED}✗ Student solution: FAILED{Colors.RESET}")
                    print(f"  Expected: '{expected}'")
                    print(f"  Got: '{student_result}'")
                    failed += 1
            except Exception as e:
                print(f"{Colors.RED}✗ Student solution: ERROR{Colors.RESET}")
                print(f"  {str(e)}")
                failed += 1
                continue
            
            # Test inefficient implementation
            try:
                inefficient_result = student_module.build_string(input_data)
                
                if inefficient_result == expected:
                    print(f"{Colors.YELLOW}  Inefficient implementation: Correct result, but slow{Colors.RESET}")
                else:
                    print(f"{Colors.RED}  Inefficient implementation: Unexpected result{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}  Inefficient implementation: Error{Colors.RESET}")
            
            # Test reference solution
            try:
                ref_result = solution_module.build_string_optimized(input_data)
                if ref_result == expected:
                    print(f"{Colors.GREEN}  Reference solution: PASSED{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.RED}  Reference solution: ERROR - {str(e)}{Colors.RESET}")
        
        # Performance comparison
        print(f"\n{Colors.BLUE}Performance Test (10,000 strings):{Colors.RESET}")
        large_data = ["part"] * 10000
        
        try:
            start = time.time()
            student_large = student_module.build_string_optimized(large_data)
            student_time = time.time() - start
            print(f"{Colors.CYAN}  Student solution: {student_time:.4f}s{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  Student solution: ERROR - {str(e)}{Colors.RESET}")
            student_time = None
        
        try:
            start = time.time()
            inefficient_large = student_module.build_string(large_data)
            inefficient_time = time.time() - start
            print(f"{Colors.YELLOW}  Inefficient solution: {inefficient_time:.4f}s{Colors.RESET}")
            
            if student_time is not None and student_time > 0:
                speedup = inefficient_time / student_time
                print(f"{Colors.CYAN}  Speedup: {speedup:.2f}x{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.YELLOW}  Inefficient solution: ERROR (expected for large data){Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        return passed, failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_3():
    """Test Exercise 3: Choosing the Right Data Structure"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Choosing the Right Data Structure")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        solution_module = load_module('solution_03.py', 'solution_03')
        
        test_cases = [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9], 10, [(1, 9), (2, 8), (3, 7), (4, 6)]),
            ([1, 2, 3, 4], 5, [(1, 4), (2, 3)]),
            ([1, 2, 3], 10, []),
            ([1, 1, 2, 2], 3, [(1, 2)]),
            ([], 5, []),
        ]
        
        passed_brute = 0
        failed_brute = 0
        passed_opt = 0
        failed_opt = 0
        
        for i, (numbers, target, expected) in enumerate(test_cases, 1):
            print(f"\n{Colors.BLUE}Test Case {i}:{Colors.RESET} numbers={numbers}, target={target}")
            
            # Normalize expected results (sort pairs and list)
            expected_normalized = sorted([tuple(sorted(p)) for p in expected])
            
            # Test brute force
            try:
                brute_result = student_module.find_pairs_brute_force(numbers, target)
                brute_normalized = sorted([tuple(sorted(p)) for p in brute_result])
                
                if brute_normalized == expected_normalized:
                    print(f"{Colors.GREEN}✓ Brute force: PASSED{Colors.RESET}")
                    passed_brute += 1
                else:
                    print(f"{Colors.RED}✗ Brute force: FAILED{Colors.RESET}")
                    print(f"  Expected: {expected_normalized}")
                    print(f"  Got: {brute_normalized}")
                    failed_brute += 1
            except Exception as e:
                print(f"{Colors.RED}✗ Brute force: ERROR{Colors.RESET}")
                print(f"  {str(e)}")
                failed_brute += 1
            
            # Test optimized
            try:
                opt_result = student_module.find_pairs_optimized(numbers, target)
                opt_normalized = sorted([tuple(sorted(p)) for p in opt_result])
                
                if opt_normalized == expected_normalized:
                    print(f"{Colors.GREEN}✓ Optimized: PASSED{Colors.RESET}")
                    passed_opt += 1
                else:
                    print(f"{Colors.RED}✗ Optimized: FAILED{Colors.RESET}")
                    print(f"  Expected: {expected_normalized}")
                    print(f"  Got: {opt_normalized}")
                    failed_opt += 1
            except Exception as e:
                print(f"{Colors.RED}✗ Optimized: ERROR{Colors.RESET}")
                print(f"  {str(e)}")
                failed_opt += 1
        
        # Performance comparison
        print(f"\n{Colors.BLUE}Performance Test (1,000 elements):{Colors.RESET}")
        large_data = list(range(1, 1001))
        large_target = 500
        
        try:
            start = time.time()
            brute_large = student_module.find_pairs_brute_force(large_data, large_target)
            brute_time = time.time() - start
            print(f"{Colors.YELLOW}  Brute force: {brute_time:.4f}s{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  Brute force: ERROR - {str(e)}{Colors.RESET}")
            brute_time = None
        
        try:
            start = time.time()
            opt_large = student_module.find_pairs_optimized(large_data, large_target)
            opt_time = time.time() - start
            print(f"{Colors.CYAN}  Optimized: {opt_time:.4f}s{Colors.RESET}")
            
            if brute_time is not None and opt_time > 0:
                speedup = brute_time / opt_time
                print(f"{Colors.CYAN}  Speedup: {speedup:.2f}x{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  Optimized: ERROR - {str(e)}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Brute force - Passed: {passed_brute}, Failed: {failed_brute}{Colors.RESET}")
        print(f"{Colors.GREEN}Optimized - Passed: {passed_opt}, Failed: {failed_opt}{Colors.RESET}")
        return passed_brute + passed_opt, failed_brute + failed_opt
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def test_exercise_4():
    """Test Exercise 4: Understanding Algorithm Complexity"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: Understanding Algorithm Complexity")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        solution_module = load_module('solution_04.py', 'solution_04')
        
        # Test find_max (should work as-is)
        print(f"\n{Colors.BLUE}Testing find_max:{Colors.RESET}")
        try:
            test1 = [3, 1, 4, 1, 5, 9, 2, 6]
            result = student_module.find_max(test1)
            if result == 9:
                print(f"{Colors.GREEN}✓ find_max: PASSED{Colors.RESET}")
            else:
                print(f"{Colors.RED}✗ find_max: FAILED (expected 9, got {result}){Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ find_max: ERROR - {str(e)}{Colors.RESET}")
        
        # Test has_duplicate vs has_duplicate_optimized
        print(f"\n{Colors.BLUE}Testing has_duplicate:{Colors.RESET}")
        duplicate_tests = [
            ([1, 2, 3, 4, 5], False),
            ([1, 2, 3, 2, 5], True),
            ([], False),
            ([1, 1], True),
        ]
        
        passed_dup = 0
        failed_dup = 0
        passed_dup_opt = 0
        failed_dup_opt = 0
        
        for numbers, expected in duplicate_tests:
            # Test original
            try:
                result = student_module.has_duplicate(numbers)
                if result == expected:
                    passed_dup += 1
                else:
                    failed_dup += 1
                    print(f"{Colors.RED}  has_duplicate({numbers}): FAILED (expected {expected}, got {result}){Colors.RESET}")
            except Exception as e:
                failed_dup += 1
                print(f"{Colors.RED}  has_duplicate({numbers}): ERROR - {str(e)}{Colors.RESET}")
            
            # Test optimized
            try:
                result = student_module.has_duplicate_optimized(numbers)
                if result == expected:
                    passed_dup_opt += 1
                else:
                    failed_dup_opt += 1
                    print(f"{Colors.RED}  has_duplicate_optimized({numbers}): FAILED (expected {expected}, got {result}){Colors.RESET}")
            except Exception as e:
                failed_dup_opt += 1
                print(f"{Colors.RED}  has_duplicate_optimized({numbers}): ERROR - {str(e)}{Colors.RESET}")
        
        if passed_dup == len(duplicate_tests):
            print(f"{Colors.GREEN}✓ has_duplicate: All tests passed{Colors.RESET}")
        if passed_dup_opt == len(duplicate_tests):
            print(f"{Colors.GREEN}✓ has_duplicate_optimized: All tests passed{Colors.RESET}")
        
        # Test reverse_list vs reverse_list_optimized
        print(f"\n{Colors.BLUE}Testing reverse_list:{Colors.RESET}")
        reverse_tests = [
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),
            ([1], [1]),
            ([], []),
            ([1, 2], [2, 1]),
        ]
        
        passed_rev = 0
        failed_rev = 0
        passed_rev_opt = 0
        failed_rev_opt = 0
        
        for input_list, expected in reverse_tests:
            # Test original (make a copy since it might modify)
            try:
                input_copy = input_list.copy()
                result = student_module.reverse_list(input_copy)
                if result == expected:
                    passed_rev += 1
                else:
                    failed_rev += 1
                    print(f"{Colors.RED}  reverse_list({input_list}): FAILED (expected {expected}, got {result}){Colors.RESET}")
            except Exception as e:
                failed_rev += 1
                print(f"{Colors.RED}  reverse_list({input_list}): ERROR - {str(e)}{Colors.RESET}")
            
            # Test optimized
            try:
                input_copy = input_list.copy()
                result = student_module.reverse_list_optimized(input_copy)
                # Note: optimized might modify in-place, so check both result and modified list
                if result == expected or (result is None and input_copy == expected):
                    passed_rev_opt += 1
                else:
                    failed_rev_opt += 1
                    print(f"{Colors.RED}  reverse_list_optimized({input_list}): FAILED (expected {expected}, got {result}){Colors.RESET}")
            except Exception as e:
                failed_rev_opt += 1
                print(f"{Colors.RED}  reverse_list_optimized({input_list}): ERROR - {str(e)}{Colors.RESET}")
        
        if passed_rev == len(reverse_tests):
            print(f"{Colors.GREEN}✓ reverse_list: All tests passed{Colors.RESET}")
        if passed_rev_opt == len(reverse_tests):
            print(f"{Colors.GREEN}✓ reverse_list_optimized: All tests passed{Colors.RESET}")
        
        # Performance comparison for has_duplicate
        print(f"\n{Colors.BLUE}Performance Test (has_duplicate - 10,000 elements):{Colors.RESET}")
        large_data = list(range(10000)) + [9999]  # Has duplicate at end
        
        try:
            start = time.time()
            result = student_module.has_duplicate(large_data)
            inefficient_time = time.time() - start
            print(f"{Colors.YELLOW}  has_duplicate (inefficient): {inefficient_time:.4f}s{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  has_duplicate: ERROR - {str(e)}{Colors.RESET}")
            inefficient_time = None
        
        try:
            start = time.time()
            result = student_module.has_duplicate_optimized(large_data)
            optimized_time = time.time() - start
            print(f"{Colors.CYAN}  has_duplicate_optimized: {optimized_time:.4f}s{Colors.RESET}")
            
            if inefficient_time is not None and optimized_time > 0:
                speedup = inefficient_time / optimized_time
                print(f"{Colors.CYAN}  Speedup: {speedup:.2f}x{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  has_duplicate_optimized: ERROR - {str(e)}{Colors.RESET}")
        
        total_passed = passed_dup + passed_dup_opt + passed_rev + passed_rev_opt
        total_failed = failed_dup + failed_dup_opt + failed_rev + failed_rev_opt
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {total_passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {total_failed}{Colors.RESET}")
        return total_passed, total_failed
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1


def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*60)
    print("Easy Code Challenge - Automated Test Suite")
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
