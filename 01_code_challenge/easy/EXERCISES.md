# Code Challenge - Easy Exercises

These exercises focus on fundamental Python concepts and basic optimizations.

---

## Exercise 1: Memory-Efficient List Operations

**Difficulty:** Easy  
**Time Limit:** 20 minutes  
**Focus:** Memory optimization, understanding Python internals

### Problem

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

### Tasks

1. **Optimize the memory usage** of this function. Consider:
   - Using generators instead of lists where possible
   - Using set operations for deduplication
   - Combining operations in a single pass

2. **Explain** the memory difference between your solution and the original

3. **Measure** the memory usage of both approaches (you can use `memory_profiler` or `sys.getsizeof()`)

### Requirements

- Your solution should work with lists containing millions of elements
- Maintain the same functionality (deduplicate → filter evens → square)
- Reduce memory footprint significantly

### Solution Template

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

### Key Learning Points

1. **Generators vs Lists:** Generators don't store all values in memory
2. **Set Operations:** Sets provide O(1) lookup vs O(n) for list membership
3. **Single Pass:** Combining operations reduces intermediate storage

### Expected Solution Concepts

- Use a set for deduplication (O(1) average lookup)
- Use generator expressions for filtering and transformation
- Combine operations to avoid multiple passes over data

---

## Exercise 2: String Builder Optimization

**Difficulty:** Easy  
**Time Limit:** 15 minutes  
**Focus:** Understanding string immutability and concatenation overhead

### Problem

You need to build a large string by concatenating many smaller strings. The naive approach of using `+=` for string concatenation in Python is inefficient because strings are immutable.

**Current Implementation:**
```python
def build_string(parts):
    result = ""
    for part in parts:
        result += part
    return result
```

### Tasks

1. **Optimize** this function for better time complexity
2. **Explain** why the original approach is inefficient
3. **Compare** the performance of different approaches:
   - String concatenation with `+=`
   - Using `str.join()`
   - Using `io.StringIO`

### Requirements

- Handle lists with 10,000+ string parts
- Maintain functionality (concatenate all parts in order)

### Solution Template

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

### Key Learning Points

1. **String Immutability:** Each `+=` creates a new string object
2. **join() Method:** More efficient for multiple concatenations
3. **StringIO:** Useful for building strings incrementally

### Expected Solution Concepts

- Use `str.join()` which is O(n) instead of O(n²)
- Understand why list comprehensions + join is optimal
- Know when to use StringIO for incremental building

---

## Exercise 3: Choosing the Right Data Structure

**Difficulty:** Easy  
**Time Limit:** 15 minutes  
**Focus:** Understanding time/space complexity of different data structures

### Problem

You need to implement a function that finds all unique pairs of numbers in a list that sum to a target value.

**Requirements:**
- Find all pairs (i, j) where i + j == target
- Avoid duplicates (if (1, 2) is found, don't include (2, 1))
- Return list of tuples

### Tasks

1. **Implement** the function using different approaches:
   - Nested loops (brute force)
   - Using a set for O(1) lookups

2. **Analyze** the time and space complexity of each approach

3. **Measure** the performance difference on larger inputs

### Solution Template

```python
def find_pairs_brute_force(numbers, target):
    """Brute force approach - O(n²) time, O(1) space"""
    pairs = []
    # Your implementation
    return pairs

def find_pairs_optimized(numbers, target):
    """Optimized approach - O(n) time, O(n) space"""
    pairs = []
    # Your implementation using a set
    return pairs

# Test
if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    
    result1 = find_pairs_brute_force(numbers, target)
    result2 = find_pairs_optimized(numbers, target)
    
    print("Brute force:", result1)
    print("Optimized:", result2)
    # Expected: [(1, 9), (2, 8), (3, 7), (4, 6)] or similar
```

### Key Learning Points

1. **Trade-offs:** Space vs Time complexity
2. **Set Lookup:** O(1) average case vs O(n) for list
3. **Choosing Structures:** Understand when to use dict, set, list

### Expected Solution Concepts

- Brute force: O(n²) time, O(1) space
- Optimized: O(n) time, O(n) space using a set
- Consider edge cases (duplicates, negative numbers)

---

## Exercise 4: Understanding Algorithm Complexity

**Difficulty:** Easy  
**Time Limit:** 20 minutes  
**Focus:** Big O notation, analyzing code complexity

### Problem

Analyze the time and space complexity of the following code snippets and provide optimized versions where possible.

### Code Snippets to Analyze

#### Snippet 1
```python
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
```

#### Snippet 2
```python
def has_duplicate(numbers):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                return True
    return False
```

#### Snippet 3
```python
def reverse_list(lst):
    n = len(lst)
    result = []
    for i in range(n):
        result.append(lst[n - 1 - i])
    return result
```

### Tasks

1. **Analyze** the time and space complexity of each snippet
2. **Determine** if any can be optimized (improve complexity or reduce constants)
3. **Implement** optimized versions
4. **Explain** your reasoning

### Solution Template

```python
# Analysis
"""
Snippet 1:
- Time Complexity: O(?)
- Space Complexity: O(?)
- Optimization: ?

Snippet 2:
- Time Complexity: O(?)
- Space Complexity: O(?)
- Optimization: ?

Snippet 3:
- Time Complexity: O(?)
- Space Complexity: O(?)
- Optimization: ?
"""

# Optimized implementations
def has_duplicate_optimized(numbers):
    # Your optimized version
    pass

def reverse_list_optimized(lst):
    # Your optimized version
    pass
```

### Key Learning Points

1. **Big O Analysis:** Understanding worst-case, average-case, best-case
2. **Trade-offs:** When to optimize time vs space
3. **Python Built-ins:** Often optimized at C level

### Expected Analysis

- Snippet 1: O(n) time, O(1) space - already optimal
- Snippet 2: O(n²) time → can be O(n) with set
- Snippet 3: O(n) time, O(n) space → can be O(n) time, O(1) space in-place

