# Code Challenge - Medium Exercises

These exercises focus on intermediate Python concepts, concurrency, memory profiling, and optimization techniques.

---

## Exercise 1: Custom Memory-Efficient Data Structure

**Difficulty:** Medium  
**Time Limit:** 45 minutes  
**Focus:** Implementing custom data structures, memory optimization

### Problem

You need to implement a data structure that efficiently stores a large collection of integers with duplicate counts, but with minimal memory overhead.

**Requirements:**
- Store integer-value pairs where many integers may repeat
- Support operations: `add(value)`, `get_count(value)`, `get_all_items()`
- Optimize for memory when there are many duplicates
- Should handle millions of entries efficiently

### Tasks

1. **Implement** a memory-efficient structure that:
   - Uses less memory when many duplicates exist
   - Still provides O(1) average-case access

2. **Compare** memory usage vs a standard dictionary approach

3. **Measure** and report memory savings

### Design Considerations

- Consider using `collections.Counter` vs `defaultdict(int)` vs custom implementation
- Think about sparse vs dense representations
- Consider compression techniques for large datasets

### Solution Template

```python
from collections import Counter, defaultdict

class EfficientCounter:
    """Your custom implementation"""
    def __init__(self):
        pass
    
    def add(self, value):
        """Add or increment count for value"""
        pass
    
    def get_count(self, value):
        """Return count for value, 0 if not present"""
        pass
    
    def get_all_items(self):
        """Return list of (value, count) tuples"""
        pass
    
    def memory_usage(self):
        """Return approximate memory usage in bytes"""
        pass

# Comparison
def compare_implementations():
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] * 100000
    
    # Standard dictionary
    standard = defaultdict(int)
    for val in data:
        standard[val] += 1
    
    # Your implementation
    efficient = EfficientCounter()
    for val in data:
        efficient.add(val)
    
    # Compare memory
    print(f"Standard: {get_memory_usage(standard)} bytes")
    print(f"Efficient: {efficient.memory_usage()} bytes")
```

### Key Learning Points

1. **Memory Optimization:** Understanding trade-offs between different representations
2. **Data Structure Choice:** When to use built-in vs custom structures
3. **Memory Profiling:** Measuring actual memory usage

### Hints

- Consider using `sys.getsizeof()` for memory measurement
- Think about using `__slots__` for custom classes
- Explore `array.array` for numeric data

---

## Exercise 2: Threading vs Multiprocessing Optimization

**Difficulty:** Medium  
**Time Limit:** 45 minutes  
**Focus:** Understanding Python's GIL, threading, multiprocessing

### Problem

You need to process a large dataset by applying a CPU-intensive function to each element. Compare and implement solutions using:

1. Sequential processing
2. Threading
3. Multiprocessing

**Task:**
Process 10,000 numbers, applying a computationally expensive operation (e.g., checking if number is prime, or computing factorial) to each.

### Requirements

1. **Implement** all three approaches
2. **Measure** performance
3. **Explain** why one approach performs better than others
4. **Handle** proper resource cleanup

### Solution Template

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

### Key Learning Points

1. **GIL (Global Interpreter Lock):** Why threading may not help with CPU-bound tasks
2. **Multiprocessing:** When to use separate processes
3. **I/O-bound vs CPU-bound:** Different strategies for each

### Expected Findings

- Sequential: Baseline performance
- Threading: May not improve CPU-bound tasks (due to GIL)
- Multiprocessing: Should show significant improvement for CPU-bound tasks

---

## Exercise 3: Memory Profiling and Optimization

**Difficulty:** Medium  
**Time Limit:** 40 minutes  
**Focus:** Memory profiling tools, identifying memory bottlenecks

### Problem

You have a function that processes a large CSV-like dataset and you suspect it has memory issues. Your task is to:

1. Profile the memory usage
2. Identify memory bottlenecks
3. Optimize the code

**Given Function:**
```python
def process_large_dataset(file_path):
    """Process a large CSV file"""
    # Read entire file into memory
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    processed = []
    for line in lines:
        parts = line.strip().split(',')
        # Create dictionary for each row
        row_dict = {}
        for i, part in enumerate(parts):
            row_dict[f'col_{i}'] = part
        processed.append(row_dict)
    
    # Filter rows
    filtered = []
    for row in processed:
        if int(row['col_0']) > 100:
            filtered.append(row)
    
    # Transform data
    result = []
    for row in filtered:
        transformed = {
            'id': row['col_0'],
            'value': row['col_1'],
            'category': row['col_2']
        }
        result.append(transformed)
    
    return result
```

### Tasks

1. **Profile** the memory usage using `memory_profiler` or similar tools
2. **Identify** the memory bottlenecks
3. **Optimize** the function to use less memory
4. **Measure** the improvement

### Solution Template

```python
# Install: pip install memory-profiler
# Usage: python -m memory_profiler your_script.py

from memory_profiler import profile
import sys

@profile
def process_large_dataset_optimized(file_path):
    """Your optimized version"""
    # Use generators, process line by line
    pass

def compare_memory_usage():
    # Create test file
    test_file = 'test_data.csv'
    with open(test_file, 'w') as f:
        for i in range(100000):
            f.write(f"{i},{i*2},{i%10}\n")
    
    # Profile original (if you implement it)
    # Profile optimized
    result = process_large_dataset_optimized(test_file)
    
    print(f"Processed {len(result)} rows")

if __name__ == "__main__":
    compare_memory_usage()
```

### Key Learning Points

1. **Memory Profiling Tools:** `memory_profiler`, `tracemalloc`, `pympler`
2. **Generator Patterns:** Processing data incrementally
3. **Memory-Efficient Parsing:** Reading files line-by-line

### Optimization Strategies

- Use generators instead of lists
- Process data incrementally
- Avoid creating intermediate data structures
- Use appropriate data types (e.g., `array.array` for numeric data)

---

## Exercise 4: Implementing Efficient Caching

**Difficulty:** Medium  
**Time Limit:** 40 minutes  
**Focus:** Caching strategies, LRU cache implementation

### Problem

Implement a caching mechanism for a computationally expensive function. You need to handle:

1. **LRU (Least Recently Used) eviction** when cache is full
2. **Memory limits** - cache should not exceed a certain memory size
3. **Thread safety** (optional but preferred)

**Given Function:**
```python
def expensive_computation(n):
    """A function that takes time to compute"""
    import time
    time.sleep(0.1)  # Simulate expensive operation
    return n * n
```

### Tasks

1. **Implement** an LRU cache decorator
2. **Add** memory-based eviction (estimate memory usage of cached values)
3. **Compare** performance with and without caching
4. **Handle** edge cases (None values, unhashable types, etc.)

### Solution Template

```python
from collections import OrderedDict
import functools

class LRUCache:
    """LRU Cache implementation"""
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get value from cache"""
        pass
    
    def put(self, key, value):
        """Add value to cache"""
        pass
    
    def clear(self):
        """Clear cache"""
        pass

def lru_cache_decorator(max_size=128):
    """LRU cache decorator"""
    # Your implementation
    pass

# Usage
@lru_cache_decorator(max_size=100)
def expensive_computation(n):
    import time
    time.sleep(0.1)
    return n * n

# Test
if __name__ == "__main__":
    import time
    
    # Without cache
    start = time.time()
    for i in range(10):
        expensive_computation(i)
    no_cache_time = time.time() - start
    
    # With cache (call same values)
    start = time.time()
    for i in range(10):
        expensive_computation(i)
    cached_time = time.time() - start
    
    print(f"No cache: {no_cache_time:.2f}s")
    print(f"With cache: {cached_time:.2f}s")
    print(f"Speedup: {no_cache_time/cached_time:.2f}x")
```

### Key Learning Points

1. **LRU Algorithm:** OrderedDict is perfect for LRU implementation
2. **Decorator Pattern:** Creating reusable caching decorators
3. **Memory Management:** Balancing cache size vs performance

### Advanced Considerations

- Consider using `functools.lru_cache` from standard library for comparison
- Implement memory-based eviction by estimating object sizes
- Add thread safety using locks if needed

