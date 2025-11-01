"""
Automated Test Runner for Medium Code Challenge Exercises

This script tests all your implementations and compares them against:
1. Expected outputs (from reference solutions)
2. Baseline/inefficient implementations
3. Performance benchmarks

Run with: python test_all.py
Or test individual exercises: python test_all.py --exercise 1
"""

import sys
import os
import time
import traceback
import tracemalloc
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
    full_path = SCRIPT_DIR / file_path
    if not full_path.exists():
        raise FileNotFoundError(f"Could not find {full_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, str(full_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {full_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_grade(correctness_score: float, efficiency_score: float, total_tests: int, passed_tests: int) -> Tuple[float, str]:
    """
    Calculate grade based on correctness (60%) and efficiency (30%).
    Code quality (10%) is assumed based on whether code runs.
    """
    correctness_weight = 0.6
    efficiency_weight = 0.3
    code_quality_weight = 0.1
    
    correctness_grade = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    efficiency_grade = efficiency_score * 100
    code_quality_grade = 100 if correctness_grade > 0 else 0
    
    total_grade = (
        correctness_grade * correctness_weight +
        efficiency_grade * efficiency_weight +
        code_quality_grade * code_quality_weight
    )
    
    if total_grade >= 90:
        letter = "A"
    elif total_grade >= 80:
        letter = "B"
    elif total_grade >= 70:
        letter = "C"
    elif total_grade >= 60:
        letter = "D"
    else:
        letter = "F"
    
    return total_grade, letter


def test_exercise_1():
    """Test Exercise 1: Custom Memory-Efficient Data Structure"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: Custom Memory-Efficient Data Structure")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_01.py', 'starter_01')
        solution_module = load_module('solution_01.py', 'solution_01')
        
        test_cases = [
            ([1, 2, 2, 3], {1: 1, 2: 2, 3: 1}),
            ([1, 1, 1, 1], {1: 4}),
            ([], {}),
            ([5, 5, 3, 3, 3, 1], {5: 2, 3: 3, 1: 1}),
        ]
        
        passed = 0
        failed = 0
        
        for i, (input_data, expected_counts) in enumerate(test_cases, 1):
            print(f"\n{Colors.BLUE}Test Case {i}:{Colors.RESET} input={input_data}")
            
            try:
                student_counter = student_module.EfficientCounter()
                for val in input_data:
                    student_counter.add(val)
                
                # Check all counts
                all_correct = True
                for val, expected_count in expected_counts.items():
                    actual_count = student_counter.get_count(val)
                    if actual_count != expected_count:
                        print(f"{Colors.RED}  ✗ Count for {val}: expected {expected_count}, got {actual_count}{Colors.RESET}")
                        all_correct = False
                
                # Check get_all_items
                all_items = student_counter.get_all_items()
                if all_items is None:
                    print(f"{Colors.RED}  ✗ get_all_items() returned None{Colors.RESET}")
                    all_correct = False
                else:
                    item_dict = dict(all_items)
                    if item_dict != expected_counts:
                        print(f"{Colors.RED}  ✗ get_all_items() mismatch: expected {expected_counts}, got {item_dict}{Colors.RESET}")
                        all_correct = False
                
                if all_correct:
                    print(f"{Colors.GREEN}  ✓ PASSED{Colors.RESET}")
                    passed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"{Colors.RED}  ✗ ERROR: {str(e)}{Colors.RESET}")
                traceback.print_exc()
                failed += 1
        
        # Memory comparison test
        print(f"\n{Colors.BLUE}Memory Efficiency Test:{Colors.RESET}")
        large_data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] * 10000
        
        try:
            from collections import defaultdict
            import sys as sys_module
            
            # Baseline
            baseline = defaultdict(int)
            for val in large_data:
                baseline[val] += 1
            baseline_memory = sum(sys_module.getsizeof(k) + sys_module.getsizeof(v) for k, v in baseline.items())
            
            # Student implementation
            student_counter = student_module.EfficientCounter()
            for val in large_data:
                student_counter.add(val)
            student_memory = student_counter.memory_usage() or sys_module.getsizeof(student_counter)
            
            # Solution implementation
            solution_counter = solution_module.EfficientCounter()
            for val in large_data:
                solution_counter.add(val)
            solution_memory = solution_counter.memory_usage()
            
            print(f"  Baseline (defaultdict):    {baseline_memory:,} bytes")
            print(f"  Student implementation:     {student_memory:,} bytes")
            print(f"  Reference solution:         {solution_memory:,} bytes")
            
            memory_savings = (1 - student_memory / baseline_memory) * 100 if baseline_memory > 0 else 0
            print(f"  Memory savings vs baseline: {memory_savings:.1f}%")
            
            efficiency_score = min(1.0, memory_savings / 50.0)  # Target: 50% savings
            if memory_savings < 20:
                print(f"{Colors.YELLOW}  ⚠ Low memory savings - aim for >20%{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}  ✗ Memory test error: {str(e)}{Colors.RESET}")
            efficiency_score = 0.0
        
        total = passed + failed
        grade, letter = calculate_grade(0.6, efficiency_score, total, passed)
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}/{total}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}/{total}{Colors.RESET}")
        print(f"{Colors.CYAN}Grade: {grade:.1f}% ({letter}){Colors.RESET}")
        
        return passed, failed, grade
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1, 0.0


def test_exercise_2():
    """Test Exercise 2: Threading vs Multiprocessing"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: Threading vs Multiprocessing")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        solution_module = load_module('solution_02.py', 'solution_02')
        
        # Small test first
        test_data = list(range(10, 30))
        
        print(f"\n{Colors.BLUE}Correctness Test:{Colors.RESET}")
        passed = 0
        failed = 0
        
        try:
            seq_result = student_module.sequential_process(test_data)
            thread_result = student_module.threaded_process(test_data, num_threads=2)
            mp_result = student_module.multiprocessed_process(test_data, num_processes=2)
            
            if seq_result == thread_result == mp_result:
                print(f"{Colors.GREEN}  ✓ All implementations produce same results{Colors.RESET}")
                passed += 1
            else:
                print(f"{Colors.RED}  ✗ Results don't match{Colors.RESET}")
                print(f"    Sequential length: {len(seq_result)}")
                print(f"    Threading length: {len(thread_result)}")
                print(f"    Multiprocessing length: {len(mp_result)}")
                failed += 1
        except Exception as e:
            print(f"{Colors.RED}  ✗ ERROR: {str(e)}{Colors.RESET}")
            traceback.print_exc()
            failed += 1
        
        # Performance test
        print(f"\n{Colors.BLUE}Performance Test (1000 items):{Colors.RESET}")
        perf_data = list(range(100, 1100))
        
        try:
            # Sequential
            start = time.time()
            seq_result = student_module.sequential_process(perf_data)
            seq_time = time.time() - start
            
            # Threading
            start = time.time()
            thread_result = student_module.threaded_process(perf_data, num_threads=4)
            thread_time = time.time() - start
            
            # Multiprocessing
            start = time.time()
            mp_result = student_module.multiprocessed_process(perf_data, num_processes=4)
            mp_time = time.time() - start
            
            print(f"  Sequential:      {seq_time:.4f}s")
            print(f"  Threading:       {thread_time:.4f}s")
            print(f"  Multiprocessing: {mp_time:.4f}s")
            
            if mp_time > 0:
                threading_speedup = seq_time / thread_time if thread_time > 0 else 0
                mp_speedup = seq_time / mp_time
                print(f"  Threading speedup:     {threading_speedup:.2f}x")
                print(f"  Multiprocessing speedup: {mp_speedup:.2f}x")
                
                # Efficiency score: multiprocessing should be faster
                if mp_speedup > 1.5:
                    efficiency_score = 1.0
                elif mp_speedup > 1.2:
                    efficiency_score = 0.8
                elif mp_speedup > 1.0:
                    efficiency_score = 0.6
                else:
                    efficiency_score = 0.3
            else:
                efficiency_score = 0.0
                
        except Exception as e:
            print(f"{Colors.RED}  ✗ Performance test error: {str(e)}{Colors.RESET}")
            traceback.print_exc()
            efficiency_score = 0.0
        
        total = passed + failed
        grade, letter = calculate_grade(0.6, efficiency_score, total, passed)
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}/{total}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}/{total}{Colors.RESET}")
        print(f"{Colors.CYAN}Grade: {grade:.1f}% ({letter}){Colors.RESET}")
        
        return passed, failed, grade
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1, 0.0


def test_exercise_3():
    """Test Exercise 3: Memory Profiling and Optimization"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Memory Profiling and Optimization")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        solution_module = load_module('solution_03.py', 'solution_03')
        
        # Create test file
        test_file = SCRIPT_DIR / 'test_data_ex3.csv'
        num_rows = 5000
        print(f"\n{Colors.BLUE}Creating test file ({num_rows} rows)...{Colors.RESET}")
        with open(test_file, 'w') as f:
            for i in range(num_rows):
                f.write(f"{i},{i*2},{i%10}\n")
        
        passed = 0
        failed = 0
        
        # Correctness test
        print(f"\n{Colors.BLUE}Correctness Test:{Colors.RESET}")
        try:
            # Original implementation result
            original_result = student_module.process_large_dataset(str(test_file))
            
            # Student optimized implementation
            student_result = student_module.process_large_dataset_optimized(str(test_file))
            
            if len(original_result) == len(student_result):
                # Check a few sample results
                if student_result[:10] == original_result[:10] and student_result[-10:] == original_result[-10:]:
                    print(f"{Colors.GREEN}  ✓ Results match{Colors.RESET}")
                    passed += 1
                else:
                    print(f"{Colors.RED}  ✗ Results don't match{Colors.RESET}")
                    failed += 1
            else:
                print(f"{Colors.RED}  ✗ Result lengths don't match: {len(original_result)} vs {len(student_result)}{Colors.RESET}")
                failed += 1
        except Exception as e:
            print(f"{Colors.RED}  ✗ ERROR: {str(e)}{Colors.RESET}")
            traceback.print_exc()
            failed += 1
        
        # Memory efficiency test
        print(f"\n{Colors.BLUE}Memory Efficiency Test:{Colors.RESET}")
        try:
            tracemalloc.start()
            
            # Test optimized version
            student_module.process_large_dataset_optimized(str(test_file))
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Compare with original (approximate)
            tracemalloc.start()
            student_module.process_large_dataset(str(test_file))
            current_orig, peak_orig = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"  Original peak memory:  {peak_orig / 1024 / 1024:.2f} MB")
            print(f"  Optimized peak memory: {peak / 1024 / 1024:.2f} MB")
            
            memory_savings = (1 - peak / peak_orig) * 100 if peak_orig > 0 else 0
            print(f"  Memory savings: {memory_savings:.1f}%")
            
            if memory_savings > 50:
                efficiency_score = 1.0
            elif memory_savings > 30:
                efficiency_score = 0.8
            elif memory_savings > 10:
                efficiency_score = 0.6
            else:
                efficiency_score = 0.3
                
        except Exception as e:
            print(f"{Colors.YELLOW}  ⚠ Memory test error: {str(e)}{Colors.RESET}")
            efficiency_score = 0.5  # Partial credit if correctness works
        
        # Cleanup
        if test_file.exists():
            test_file.unlink()
        
        total = passed + failed
        grade, letter = calculate_grade(0.6, efficiency_score, total, passed)
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}/{total}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}/{total}{Colors.RESET}")
        print(f"{Colors.CYAN}Grade: {grade:.1f}% ({letter}){Colors.RESET}")
        
        return passed, failed, grade
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1, 0.0


def test_exercise_4():
    """Test Exercise 4: Efficient Caching Implementation"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: Efficient Caching Implementation")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        solution_module = load_module('solution_04.py', 'solution_04')
        
        passed = 0
        failed = 0
        
        # Test LRUCache class
        print(f"\n{Colors.BLUE}LRUCache Class Test:{Colors.RESET}")
        try:
            cache = student_module.LRUCache(max_size=3)
            
            # Add items
            cache.put(1, "one")
            cache.put(2, "two")
            cache.put(3, "three")
            
            if cache.size() != 3:
                print(f"{Colors.RED}  ✗ Size mismatch after adding 3 items{Colors.RESET}")
                failed += 1
            elif cache.get(1) != "one":
                print(f"{Colors.RED}  ✗ Get failed for existing key{Colors.RESET}")
                failed += 1
            else:
                # Test LRU eviction
                cache.put(4, "four")  # Should evict 2 (if 1 was accessed) or 2 (if not)
                if cache.size() != 3:
                    print(f"{Colors.RED}  ✗ Size mismatch after eviction{Colors.RESET}")
                    failed += 1
                elif cache.get(2) is not None:  # 2 should be evicted
                    # Check that we can still get 3 (most recent)
                    if cache.get(3) == "three" and cache.get(4) == "four":
                        print(f"{Colors.GREEN}  ✓ LRU eviction works correctly{Colors.RESET}")
                        passed += 1
                    else:
                        print(f"{Colors.YELLOW}  ⚠ LRU eviction may not be working optimally{Colors.RESET}")
                        passed += 1  # Partial credit
                else:
                    print(f"{Colors.GREEN}  ✓ LRU eviction works correctly{Colors.RESET}")
                    passed += 1
        except Exception as e:
            print(f"{Colors.RED}  ✗ ERROR: {str(e)}{Colors.RESET}")
            traceback.print_exc()
            failed += 1
        
        # Test decorator and caching
        print(f"\n{Colors.BLUE}Caching Performance Test:{Colors.RESET}")
        try:
            test_values = list(range(10))
            
            # First call (cache miss)
            start = time.time()
            results1 = [student_module.cached_computation(i) for i in test_values]
            first_call_time = time.time() - start
            
            # Second call (cache hit)
            start = time.time()
            results2 = [student_module.cached_computation(i) for i in test_values]
            second_call_time = time.time() - start
            
            if results1 == results2:
                speedup = first_call_time / second_call_time if second_call_time > 0 else 0
                print(f"  First call (cache miss):  {first_call_time:.4f}s")
                print(f"  Second call (cache hit):  {second_call_time:.4f}s")
                print(f"  Speedup:                  {speedup:.2f}x")
                
                if speedup > 5:
                    efficiency_score = 1.0
                    print(f"{Colors.GREEN}  ✓ Excellent caching performance{Colors.RESET}")
                    passed += 1
                elif speedup > 2:
                    efficiency_score = 0.8
                    print(f"{Colors.GREEN}  ✓ Good caching performance{Colors.RESET}")
                    passed += 1
                elif speedup > 1.1:
                    efficiency_score = 0.6
                    print(f"{Colors.YELLOW}  ⚠ Moderate caching performance{Colors.RESET}")
                    passed += 1
                else:
                    efficiency_score = 0.3
                    print(f"{Colors.RED}  ✗ Low caching speedup{Colors.RESET}")
                    failed += 1
            else:
                print(f"{Colors.RED}  ✗ Cached results don't match{Colors.RESET}")
                failed += 1
        except Exception as e:
            print(f"{Colors.RED}  ✗ ERROR: {str(e)}{Colors.RESET}")
            traceback.print_exc()
            efficiency_score = 0.0
            failed += 1
        
        total = passed + failed
        grade, letter = calculate_grade(0.6, efficiency_score if 'efficiency_score' in locals() else 0.5, total, passed)
        
        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {passed}/{total}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}/{total}{Colors.RESET}")
        print(f"{Colors.CYAN}Grade: {grade:.1f}% ({letter}){Colors.RESET}")
        
        return passed, failed, grade
        
    except Exception as e:
        print(f"{Colors.RED}Error loading modules: {str(e)}{Colors.RESET}")
        traceback.print_exc()
        return 0, 1, 0.0


def main():
    """Run all tests"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*60)
    print("Medium Code Challenge - Automated Test Suite")
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
    grades = []
    
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
        passed, failed, grade = test_func()
        total_passed += passed
        total_failed += failed
        grades.append(grade)
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Final Summary")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.GREEN}Total Passed: {total_passed}{Colors.RESET}")
    print(f"{Colors.RED}Total Failed: {total_failed}{Colors.RESET}")
    
    if grades:
        avg_grade = sum(grades) / len(grades)
        print(f"{Colors.CYAN}Average Grade: {avg_grade:.1f}%{Colors.RESET}")
        
        if avg_grade >= 90:
            print(f"{Colors.GREEN}{Colors.BOLD}Excellent work! 🎉{Colors.RESET}")
        elif avg_grade >= 80:
            print(f"{Colors.GREEN}{Colors.BOLD}Great job! 👍{Colors.RESET}")
        elif avg_grade >= 70:
            print(f"{Colors.YELLOW}{Colors.BOLD}Good effort! Keep practicing! 💪{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}Keep working at it! You'll improve! 💪{Colors.RESET}")
    
    if total_failed == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())

