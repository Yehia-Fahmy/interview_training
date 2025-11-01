"""
Exercise 2: Threading vs Multiprocessing

Process a large dataset using CPU-intensive operations. Implement and
compare sequential, threading, and multiprocessing approaches.

Requirements:
- Implement all three approaches
- Measure performance for each
- Handle proper resource cleanup
- Explain why one approach performs better

Note: Threading may not help with CPU-bound tasks due to Python's GIL.
"""

import time
import math
from threading import Thread
from multiprocessing import Process, Pool


def cpu_intensive_task(n):
    """
    A CPU-intensive operation.
    Returns True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def sequential_process(data):
    """
    Sequential processing - baseline implementation.
    Process each item one by one.
    """
    results = []
    for item in data:
        results.append(cpu_intensive_task(item))
    return results


def threaded_process(data, num_threads=4):
    """
    Threading approach.
    Hint: Split data into chunks and process in parallel threads.
    """
    pass


def multiprocessed_process(data, num_processes=4):
    """
    Multiprocessing approach.
    Hint: Use multiprocessing.Pool for parallel processing.
    """
    pass


def compare_approaches():
    """Compare all three approaches"""
    data = list(range(1000, 11000))  # 10,000 numbers
    
    # Sequential
    print("Running sequential approach...")
    start = time.time()
    seq_results = sequential_process(data)
    seq_time = time.time() - start
    
    # Threading
    print("Running threading approach...")
    start = time.time()
    thread_results = threaded_process(data)
    thread_time = time.time() - start
    
    # Multiprocessing
    print("Running multiprocessing approach...")
    start = time.time()
    mp_results = multiprocessed_process(data)
    mp_time = time.time() - start
    
    # Verify all produce same results
    assert seq_results == thread_results == mp_results, "Results don't match!"
    
    print(f"\nSequential:    {seq_time:.2f}s")
    print(f"Threading:     {thread_time:.2f}s")
    print(f"Multiprocessing: {mp_time:.2f}s")
    
    print(f"\nThreading speedup: {seq_time/thread_time:.2f}x")
    print(f"Multiprocessing speedup: {seq_time/mp_time:.2f}x")


if __name__ == "__main__":
    # Quick test with small data
    test_data = list(range(10, 20))
    print("Testing sequential...")
    result = sequential_process(test_data)
    print(f"Processed {len(result)} items")
    print(f"First few results: {result[:5]}")

