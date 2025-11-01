"""
Solution for Exercise 1: Memory-Efficient List Operations

This file contains the reference solution. It's kept separate so you
can't see it while working on your implementation.
"""

def process_numbers_optimized(numbers):
    """
    Optimized implementation using set for deduplication and
    generator expressions to avoid intermediate lists.
    """
    # Use set for O(1) deduplication instead of O(n) list lookup
    seen = set()
    unique = (num for num in numbers if num not in seen and not seen.add(num))
    
    # Chain operations: filter evens and square in a single pass
    # This returns a generator, so we convert to list at the end
    result = [num * num for num in unique if num % 2 == 0]
    
    return result


# Alternative solution using a single pass
def process_numbers_optimized_v2(numbers):
    """
    Single-pass solution that combines all operations.
    Most memory-efficient approach.
    """
    seen = set()
    result = []
    for num in numbers:
        # Deduplicate and check if seen
        if num in seen:
            continue
        seen.add(num)
        
        # Filter evens and square
        if num % 2 == 0:
            result.append(num * num)
    
    return result
