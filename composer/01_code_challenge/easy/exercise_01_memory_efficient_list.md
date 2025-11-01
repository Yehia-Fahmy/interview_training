# Exercise 1: Memory-Efficient List Operations

**Difficulty:** Easy  
**Time Limit:** 20 minutes  
**Focus:** Memory optimization, understanding Python internals

## Problem

You are given a function that processes a large list of integers (potentially millions of elements). The current implementation creates multiple intermediate lists, which causes high memory usage.

**Current Implementation:**
```python
def process_numbers(numbers):
    # Remove duplicates
    unique = []
    for num in numbers:
        if num not in unique:
            unique.append(num)
    
    # Filter even numbers
    evens = []
    for num in unique:
        if num % 2 == 0:
            evens.append(num)
    
    # Square each number
    squared = []
    for num in evens:
        squared.append(num * num)
    
    return squared
```

## Tasks

1. **Optimize the memory usage** of this function. Consider:
   - Using generators instead of lists where possible
   - Using set operations for deduplication
   - Combining operations in a single pass

2. **Explain** the memory difference between your solution and the original

3. **Measure** the memory usage of both approaches (you can use `memory_profiler` or `sys.getsizeof()`)

## Requirements

- Your solution should work with lists containing millions of elements
- Maintain the same functionality (deduplicate → filter evens → square)
- Reduce memory footprint significantly

## Solution Template

```python
def process_numbers_optimized(numbers):
    # Your optimized implementation here
    pass

# Test your solution
if __name__ == "__main__":
    # Test with small data first
    test_data = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]
    result = process_numbers_optimized(test_data)
    print(result)  # Expected: [4, 16, 36, 64]
    
    # Then test with larger data for memory comparison
    # large_data = list(range(1000000))
    # Compare memory usage
```

## Key Learning Points

1. **Generators vs Lists:**** Generators don't store all values in memory
2. **Set Operations:** Sets provide O(1) lookup vs O(n) for list membership
3. **Single Pass:** Combining operations reduces intermediate storage

## Expected Solution Concepts

- Use a set for deduplication (O(1) average lookup)
- Use generator expressions for filtering and transformation
- Combine operations to avoid multiple passes over data

