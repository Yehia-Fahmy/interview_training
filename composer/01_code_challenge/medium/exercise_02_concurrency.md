# Exercise 2: Threading vs Multiprocessing Optimization

**Difficulty:** Medium  
**Time Limit:** 45 minutes  
**Focus:** Understanding Python's GIL, threading, multiprocessing

## Problem

You need to process a large dataset by applying a CPU-intensive function to each element. Compare and implement solutions using:

1. Sequential processing
2. Threading
3. Multiprocessing

**Task:**
Process 10,000 numbers, applying a computationally expensive operation (e.g., checking if number is prime, or computing factorial) to each.

## Requirements

1. **Implement** all three approaches
2. **Measure** performance
3. **Explain** why one approach performs better than others
4. **Handle** proper resource cleanup

## Solution Template

```python
import time
from threading import Thread
from multiprocessing import Process, Pool
import math

def cpu_intensive_task(n):
    """A CPU-intensive operation"""
    # Example: Check if number is prime (simplified)
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def sequential_process(data):
    """Sequential processing"""
    results = []
    for item in data:
        results.append(cpu_intensive_task(item))
    return results

def threaded_process(data, num_threads=4):
    """Threading approach"""
    # Your implementation
    pass

def multiprocessed_process(data, num_processes=4):
    """Multiprocessing approach"""
    # Your implementation
    pass

def compare_approaches():
    data = list(range(1000, 11000))
    
    # Sequential
    start = time.time()
    seq_results = sequential_process(data)
    seq_time = time.time() - start
    
    # Threading
    start = time.time()
    thread_results = threaded_process(data)
    thread_time = time.time() - start
    
    # Multiprocessing
    start = time.time()
    mp_results = multiprocessed_process(data)
    mp_time = time.time() - start
    
    # Verify all produce same results
    assert seq_results == thread_results == mp_results
    
    print(f"Sequential: {seq_time:.2f}s")
    print(f"Threading: {thread_time:.2f}s")
    print(f"Multiprocessing: {mp_time:.2f}s")

if __name__ == "__main__":
    compare_approaches()
```

## Key Learning Points

1. **GIL (Global Interpreter Lock):** Why threading may not help with CPU-bound tasks
2. **Multiprocessing:** When to use separate processes
3. **I/O-bound vs CPU-bound:** Different strategies for each

## Expected Findings

- Sequential: Baseline performance
- Threading: May not improve CPU-bound tasks (due to GIL)
- Multiprocessing: Should show significant improvement for CPU-bound tasks

