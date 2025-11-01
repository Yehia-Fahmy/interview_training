# Exercise 3: Choosing the Right Data Structure

**Difficulty:** Easy  
**Time Limit:** 15 minutes  
**Focus:** Understanding time/space complexity of different data structures

## Problem

You need to implement a function that finds all unique pairs of numbers in a list that sum to a target value.

**Requirements:**
- Find all pairs (i, j) where i + j == target
- Avoid duplicates (if (1, 2) is found, don't include (2, 1))
- Return list of tuples

## Tasks

1. **Implement** the function using different approaches:
   - Nested loops (brute force)
   - Using a set for O(1) lookups

2. **Analyze** the time and space complexity of each approach

3. **Measure** the performance difference on larger inputs

## Solution Template

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

## Key Learning Points

1. **Trade-offs:** Space vs Time complexity
2. **Set Lookup:** O(1) average case vs O(n) for list
3. **Choosing Structures:** Understand when to use dict, set, list

## Expected Solution Concepts

- Brute force: O(n²) time, O(1) space
- Optimized: O(n) time, O(n) space using a set
- Consider edge cases (duplicates, negative numbers)

