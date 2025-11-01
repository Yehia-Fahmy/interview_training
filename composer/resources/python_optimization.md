# Python Optimization Guide

Quick reference for Python performance optimization concepts relevant to the Code Challenge interview.

## Memory Optimization

- **Generators**: Use generators instead of lists for large datasets
- **`__slots__`**: Reduce memory overhead in classes
- **`array.array`**: More memory-efficient than lists for numeric data
- **Memory Mapping**: `mmap` for large files

## Performance Profiling

- **`cProfile`**: CPU profiling
- **`memory_profiler`**: Memory profiling
- **`line_profiler`**: Line-by-line profiling
- **`timeit`**: Micro-benchmarking

## Concurrency

- **Threading**: I/O-bound tasks (GIL limitations)
- **Multiprocessing**: CPU-bound tasks
- **`concurrent.futures`**: High-level concurrency
- **Async/await**: Asynchronous I/O

## Data Structures

- **Sets**: O(1) membership testing
- **Deque**: Fast append/pop from both ends
- **Counter**: Efficient counting
- **defaultdict**: Avoid key checks

## Common Patterns

- **String Building**: Use `join()` not `+=`
- **List Comprehensions**: Generally faster than loops
- **Caching**: `functools.lru_cache` for expensive functions
- **Vectorization**: NumPy for array operations

