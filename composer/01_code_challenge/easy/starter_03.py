"""
Exercise 3: Choosing the Right Data Structure

Implement a function that finds all unique pairs of numbers in a list
that sum to a target value. Consider the trade-offs between different
approaches and data structures.
"""

def find_pairs_brute_force(numbers, target):
    """
    Brute force approach - implement using nested loops.
    
    Requirements:
    - Find all pairs (i, j) where i + j == target
    - Avoid duplicates (if (1, 2) is found, don't include (2, 1))
    - Return list of tuples
    """
    pass


def find_pairs_optimized(numbers, target):
    """
    Optimized approach - use a set for O(1) lookups.
    
    Requirements:
    - Find all pairs (i, j) where i + j == target
    - Avoid duplicates (if (1, 2) is found, don't include (2, 1))
    - Return list of tuples
    - Use a set to achieve O(n) time complexity
    """
    pass


if __name__ == "__main__":
    # Basic test
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    target = 10
    
    result1 = find_pairs_brute_force(numbers, target)
    result2 = find_pairs_optimized(numbers, target)
    
    print(f"Brute force result: {result1}")
    print(f"Optimized result: {result2}")
    
    # Note: Order may differ, but should contain same pairs
    print(f"Expected pairs: [(1, 9), (2, 8), (3, 7), (4, 6)]")
