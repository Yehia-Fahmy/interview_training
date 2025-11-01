# Exercise 2: String Builder Optimization

**Difficulty:** Easy  
**Time Limit:** 15 minutes  
**Focus:** Understanding string immutability and concatenation overhead

## Problem

You need to build a large string by concatenating many smaller strings. The naive approach of using `+=` for string concatenation in Python is inefficient because strings are immutable.

**Current Implementation:**
```python
def build_string(parts):
    result = ""
    for part in parts:
        result += part
    return result
```

## Tasks

1. **Optimize** this function for better time complexity
2. **Explain** why the original approach is inefficient
3. **Compare** the performance of different approaches:
   - String concatenation with `+=`
   - Using `str.join()`
   - Using `io.StringIO`

## Requirements

- Handle lists with 10,000+ string parts
- Maintain functionality (concatenate all parts in order)

## Solution Template

```python
def build_string_optimized(parts):
    # Your optimized implementation here
    pass

# Performance comparison
import time

def compare_performance():
    test_data = ["part"] * 10000
    
    # Time original
    start = time.time()
    result1 = build_string(test_data)
    time1 = time.time() - start
    
    # Time optimized
    start = time.time()
    result2 = build_string_optimized(test_data)
    time2 = time.time() - start
    
    assert result1 == result2
    print(f"Original: {time1:.4f}s, Optimized: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")

if __name__ == "__main__":
    compare_performance()
```

## Key Learning Points

1. **String Immutability:** Each `+=` creates a new string object
2. **join() Method:** More efficient for multiple concatenations
3. **StringIO:** Useful for building strings incrementally

## Expected Solution Concepts

- Use `str.join()` which is O(n) instead of O(n²)
- Understand why list comprehensions + join is optimal
- Know when to use StringIO for incremental building

