"""
Exercise 1: Memory-Efficient List Operations

Optimize the process_numbers function to reduce memory usage.
Your solution should handle large lists efficiently while maintaining
the same functionality: deduplicate → filter evens → square.
"""


def process_numbers(numbers):
    """
    Current inefficient implementation - creates multiple intermediate lists.
    DO NOT MODIFY THIS FUNCTION - it's used for comparison in tests.
    """
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


def process_numbers_optimized(numbers):
    """
    Your optimized implementation here.
    
    Requirements:
    - Maintain same functionality: deduplicate → filter evens → square
    - Reduce memory footprint significantly
    - Work efficiently with large lists (millions of elements)
    """
    # REVIEW: Good approach. Uses a set for O(1) dedup checks and a single
    # output list. This preserves the order of first occurrences that pass the
    # even filter, matching the spec pipeline. Time: O(n). Space: O(n) for the
    # set plus the result. Clear and memory-efficient.
    unique = set()
    squared = []
    for n in numbers:
        if not n in unique:
            unique.add(n)
            if n % 2 == 0:
                squared.append(n*n)
    return squared



if __name__ == "__main__":
    # Test with small data
    test_data = [1, 2, 2, 3, 4, 4, 5, 6, 7, 8]
    result = process_numbers_optimized(test_data)
    print(f"Result: {result}")
    print(f"Expected: [4, 16, 36, 64]")
