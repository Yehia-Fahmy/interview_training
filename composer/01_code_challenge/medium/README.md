# Code Challenge - Medium Exercises

These exercises focus on intermediate Python concepts, concurrency, and performance optimization. Each exercise tests your ability to write efficient, production-ready code that handles real-world constraints.

## Getting Started

1. **Read the exercise description** in the corresponding markdown file
2. **Implement your solution** in the `starter_XX.py` file
3. **Run the test suite** with: `python test_all.py`
4. **Compare your solution** with the reference solution in `solution_XX.py` after attempting

## Running Tests

Test all exercises:
```bash
python test_all.py
```

Test a specific exercise:
```bash
python test_all.py --exercise 1
```

The test suite will:
- Verify correctness of your implementation
- Measure performance metrics (time, memory)
- Compare against baseline implementations
- Provide a grade based on correctness and efficiency

## Exercises

### 1. Custom Memory-Efficient Data Structure
**Focus:** Data structure design, memory optimization  
**Time Limit:** 45 minutes  
**Challenge:** Implement a counter-like data structure that efficiently handles millions of entries with many duplicates. Your solution must use significantly less memory than a standard dictionary while maintaining O(1) average-case access.

**Key Skills Tested:**
- Understanding memory overhead of different data structures
- Choosing appropriate data structures for specific use cases
- Memory profiling and measurement

### 2. Threading vs Multiprocessing
**Focus:** Python GIL, concurrency, parallel processing  
**Time Limit:** 45 minutes  
**Challenge:** Process 10,000+ numbers using CPU-intensive operations. Implement and compare sequential, threading, and multiprocessing approaches. Explain the performance differences and demonstrate understanding of Python's Global Interpreter Lock (GIL).

**Key Skills Tested:**
- Understanding when to use threading vs multiprocessing
- Practical application of concurrency primitives
- Performance measurement and analysis
- Resource cleanup and best practices

### 3. Memory Profiling and Optimization
**Focus:** Memory profiling, optimization techniques  
**Time Limit:** 40 minutes  
**Challenge:** Profile a memory-inefficient CSV processing function and optimize it to handle large datasets without running out of memory. Use profiling tools to identify bottlenecks and implement a memory-efficient solution.

**Key Skills Tested:**
- Using memory profiling tools (memory_profiler, tracemalloc)
- Identifying memory bottlenecks
- Implementing generator-based patterns
- Optimizing data processing pipelines

### 4. Efficient Caching Implementation
**Focus:** Caching algorithms, decorators, memory management  
**Time Limit:** 40 minutes  
**Challenge:** Implement an LRU (Least Recently Used) cache decorator with memory limits. Your cache must evict entries based on both recency and memory usage, and demonstrate significant performance improvements for expensive computations.

**Key Skills Tested:**
- Understanding and implementing LRU cache algorithm
- Creating reusable decorators
- Memory estimation and management
- Handling edge cases and thread safety (optional)

## Assessment Criteria

Each exercise is graded on:
1. **Correctness (60%)** - Does it produce the correct results?
2. **Efficiency (30%)** - Performance compared to baseline/reference
3. **Code Quality (10%)** - Clean, readable, well-documented code

## Tips

- Start with correctness, then optimize
- Use appropriate tools for profiling and measurement
- Read the exercise descriptions carefully for requirements
- Test with both small and large datasets
- Consider edge cases (empty inputs, large inputs, etc.)
