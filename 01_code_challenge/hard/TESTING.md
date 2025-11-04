# Testing Guide - Hard Exercises

This directory contains starter scripts and an automated test runner for the hard code challenge exercises.

## Files

- **`starter_01.py` through `starter_04.py`**: Your starter scripts with TODO comments
- **`test_all.py`**: Automated test runner for your implementations
- **`exercise_*.md`**: Problem descriptions and requirements

## Quick Start

### Run All Tests
```bash
python test_all.py
```

### Run a Specific Exercise
```bash
python test_all.py --exercise 1
python test_all.py --exercise 2
python test_all.py --exercise 3
python test_all.py --exercise 4
```

## Exercises Overview

1. **Exercise 1: Memory-Mapped File Processing**
   - Process large binary files without loading into RAM
   - Compare mmap vs standard I/O performance
   - Extension: Multiprocessing for parallel processing
   - Benchmark different approaches (single-threaded vs multiprocessing)

2. **Exercise 2: Garbage Collection Optimization**
   - Profile GC behavior
   - Optimize object lifecycle management

3. **Exercise 3: Optimized Graph Algorithm**
   - Implement efficient graph data structure
   - Shortest path and connected components

4. **Exercise 4: Concurrent Producer-Consumer Pattern**
   - Thread-safe producer-consumer system
   - Bounded queue and graceful shutdown

## Workflow

1. Read the exercise markdown file (e.g., `exercise_01_mmap.md`)
2. Open and work on the corresponding starter file (e.g., `starter_01.py`)
3. Implement your solution following the TODO comments
4. Run `python test_all.py --exercise 1` to test your work
5. Fix any issues and re-run tests

## Testing Notes

- **Exercise 1**: Creates temporary test files in system temp directory
- **Exercise 2**: Measures GC statistics - check console output
- **Exercise 3**: Tests graph algorithms with small test cases
- **Exercise 4**: Uses reduced thread counts for faster testing

## Performance Considerations

These exercises focus on:
- **Memory efficiency**: Using appropriate data structures
- **Performance optimization**: Reducing GC overhead, efficient algorithms
- **Scalability**: Handling large datasets efficiently
- **Concurrency**: Thread-safe operations and synchronization

## Troubleshooting

If you get import errors, make sure you're running from the `hard` directory:
```bash
cd composer/01_code_challenge/hard
python test_all.py
```

### Exercise-Specific Issues

**Exercise 1 (mmap)**:
- Make sure you have write permissions for temp directory
- Large test files may take time to generate
- **Multiprocessing Extension**:
  - The script includes a `compare_performance()` function that benchmarks all approaches
  - Run the script directly to see performance comparisons
  - Verify all approaches produce identical results
  - Check that multiprocessing shows speedup on multi-core systems
  - Test with different process counts (2, 4, 8) to find optimal performance

**Exercise 2 (GC)**:
- GC behavior varies by Python version
- Check console output for GC statistics

**Exercise 3 (Graph)**:
- Ensure graph is properly initialized before adding edges
- Verify path finding algorithms handle disconnected graphs

**Exercise 4 (Producer-Consumer)**:
- Threads may take time to start/stop
- Ensure graceful shutdown completes all items in queue

## Advanced Testing

For more thorough testing:

1. **Exercise 1**: 
   - Test with larger files (uncomment larger num_integers)
   - **Multiprocessing**: Run `compare_performance()` to benchmark all approaches
   - Test with different file sizes to see scaling behavior
   - Compare performance across different numbers of processes
2. **Exercise 2**: Compare GC stats before/after optimization
3. **Exercise 3**: Test with larger graphs (millions of nodes)
4. **Exercise 4**: Measure throughput with different producer/consumer ratios

