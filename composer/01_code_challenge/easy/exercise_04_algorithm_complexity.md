# Exercise 4: Understanding Algorithm Complexity

**Difficulty:** Easy  
**Time Limit:** 20 minutes  
**Focus:** Big O notation, analyzing code complexity

## Problem

Analyze the time and space complexity of the following code snippets and provide optimized versions where possible.

## Code Snippets to Analyze

### Snippet 1
```python
def find_max(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
```

### Snippet 2
```python
def has_duplicate(numbers):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                return True
    return False
```

### Snippet 3
```python
def reverse_list(lst):
    n = len(lst)
    result = []
    for i in range(n):
        result.append(lst[n - 1 - i])
    return result
```

## Tasks

1. **Analyze** the time and space complexity of each snippet
2. **Determine** if any can be optimized (improve complexity or reduce constants)
3. **Implement** optimized versions
4. **Explain** your reasoning

## Solution Template

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

## Key Learning Points

1. **Big O Analysis:** Understanding worst-case, average-case, best-case
2. **Trade-offs:** When to optimize time vs space
3. **Python Built-ins:** Often optimized at C level

## Expected Analysis

- Snippet 1: O(n) time, O(1) space - already optimal
- Snippet 2: O(n²) time → can be O(n) with set
- Snippet 3: O(n) time, O(n) space → can be O(n) time, O(1) space in-place

