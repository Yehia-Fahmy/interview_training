# Exercise 1: Custom Memory-Efficient Data Structure

**Difficulty:** Medium  
**Time Limit:** 45 minutes  
**Focus:** Implementing custom data structures, memory optimization

## Problem

You need to implement a data structure that efficiently stores a large collection of integers with duplicate counts, but with minimal memory overhead.

**Requirements:**
- Store integer-value pairs where many integers may repeat
- Support operations: `add(value)`, `get_count(value)`, `get_all_items()`
- Optimize for memory when there are many duplicates
- Should handle millions of entries efficiently

## Tasks

1. **Implement** a memory-efficient structure that:
   - Uses less memory when many duplicates exist
   - Still provides O(1) average-case access

2. **Compare** memory usage vs a standard dictionary approach

3. **Measure** and report memory savings

## Design Considerations

- Consider using `collections.Counter` vs `defaultdict(int)` vs custom implementation
- Think about sparse vs dense representations
- Consider compression techniques for large datasets

## Solution Template

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

## Key Learning Points

1. **Memory Optimization:** Understanding trade-offs between different representations
2. **Data Structure Choice:** When to use built-in vs custom structures
3. **Memory Profiling:** Measuring actual memory usage

## Hints

- Consider using `sys.getsizeof()` for memory measurement
- Think about using `__slots__` for custom classes
- Explore `array.array` for numeric data

