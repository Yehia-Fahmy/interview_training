"""
Automated Test Runner for Hard Code Challenge Exercises

This script tests all your implementations for the hard exercises.

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
    """Test Exercise 1: Memory-Mapped File Processing"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 1: Memory-Mapped File Processing")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_01.py', 'starter_01')
        
        # Create a temporary test file
        import tempfile
        test_file = os.path.join(tempfile.gettempdir(), 'test_mmap.bin')
        
        # Generate test file
        print(f"{Colors.BLUE}Generating test file...{Colors.RESET}")
        student_module.generate_test_file(test_file, 1000)
        
        if os.path.exists(test_file):
            # Test mmap processing
            print(f"{Colors.BLUE}Testing memory-mapped processing...{Colors.RESET}")
            mmap_stats = student_module.process_with_mmap(test_file)
            
            # Test standard I/O processing
            print(f"{Colors.BLUE}Testing standard I/O processing...{Colors.RESET}")
            std_stats = student_module.process_standard_io(test_file)
            
            print(f"{Colors.GREEN}✓ File processing completed{Colors.RESET}")
            print(f"  MMAP stats: {mmap_stats}")
            print(f"  Standard I/O stats: {std_stats}")
            
            # Cleanup
            os.remove(test_file)
        else:
            print(f"{Colors.RED}✗ Could not generate test file{Colors.RESET}")
            
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {str(e)}{Colors.RESET}")
        traceback.print_exc()


def test_exercise_2():
    """Test Exercise 2: Garbage Collection Optimization"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 2: Garbage Collection Optimization")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_02.py', 'starter_02')
        
        print(f"{Colors.BLUE}Testing original implementation...{Colors.RESET}")
        # Test original (should work)
        
        print(f"{Colors.BLUE}Testing optimized implementation...{Colors.RESET}")
        # Test optimized
        student_module.process_data_optimized()
        
        print(f"{Colors.GREEN}✓ GC optimization test completed{Colors.RESET}")
        print(f"{Colors.YELLOW}Note: Check output for GC statistics{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {str(e)}{Colors.RESET}")
        traceback.print_exc()


def test_exercise_3():
    """Test Exercise 3: Optimized Graph Algorithm"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 3: Optimized Graph Algorithm")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_03.py', 'starter_03')
        
        g = student_module.EfficientGraph()
        
        # Build test graph
        edges = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
        for u, v in edges:
            g.add_edge(u, v)
        
        # Test shortest path
        print(f"{Colors.BLUE}Testing shortest path...{Colors.RESET}")
        path = g.shortest_path(1, 4)
        expected_path = [1, 2, 3, 4]
        
        if path == expected_path:
            print(f"{Colors.GREEN}✓ Shortest path: PASSED{Colors.RESET}")
            print(f"  Path: {path}")
        else:
            print(f"{Colors.RED}✗ Shortest path: FAILED{Colors.RESET}")
            print(f"  Expected: {expected_path}")
            print(f"  Got: {path}")
        
        # Test connected components
        print(f"{Colors.BLUE}Testing connected components...{Colors.RESET}")
        components = g.connected_components()
        print(f"{Colors.GREEN}✓ Components: {components}{Colors.RESET}")
        
        # Test memory usage
        memory = g.memory_usage()
        print(f"{Colors.GREEN}✓ Memory usage: {memory} bytes{Colors.RESET}")
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {str(e)}{Colors.RESET}")
        traceback.print_exc()


def test_exercise_4():
    """Test Exercise 4: Concurrent Producer-Consumer Pattern"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("Exercise 4: Concurrent Producer-Consumer Pattern")
    print(f"{'='*60}{Colors.RESET}")
    
    try:
        student_module = load_module('starter_04.py', 'starter_04')
        
        system = student_module.ProducerConsumerSystem(
            num_producers=2,  # Reduced for testing
            num_consumers=1,
            queue_size=100
        )
        
        print(f"{Colors.BLUE}Starting system...{Colors.RESET}")
        start_time = time.time()
        system.start()
        
        # Run for 2 seconds (reduced for testing)
        time.sleep(2)
        
        print(f"{Colors.BLUE}Stopping system...{Colors.RESET}")
        system.stop()
        elapsed = time.time() - start_time
        
        print(f"{Colors.GREEN}✓ System test completed{Colors.RESET}")
        print(f"  Total time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"{Colors.RED}✗ ERROR: {str(e)}{Colors.RESET}")
        traceback.print_exc()


def main():
    """Run all tests or specific exercise."""
    exercises = {
        '1': test_exercise_1,
        '2': test_exercise_2,
        '3': test_exercise_3,
        '4': test_exercise_4,
    }
    
    # Check for command line argument
    if len(sys.argv) > 1 and '--exercise' in sys.argv:
        idx = sys.argv.index('--exercise')
        if idx + 1 < len(sys.argv):
            ex_num = sys.argv[idx + 1]
            if ex_num in exercises:
                exercises[ex_num]()
            else:
                print(f"{Colors.RED}Invalid exercise number: {ex_num}{Colors.RESET}")
                print(f"Available exercises: {list(exercises.keys())}")
        else:
            print(f"{Colors.RED}Usage: python test_all.py --exercise <1-4>{Colors.RESET}")
    else:
        # Run all tests
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("="*60)
        print("Running All Hard Exercise Tests")
        print("="*60)
        print(f"{Colors.RESET}")
        
        for ex_num, test_func in exercises.items():
            try:
                test_func()
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}Critical error in exercise {ex_num}: {e}{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
        print("All Tests Completed")
        print(f"{'='*60}{Colors.RESET}")


if __name__ == "__main__":
    main()

