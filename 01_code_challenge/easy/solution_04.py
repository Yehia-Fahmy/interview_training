"""
Solution for Exercise 4: Understanding Algorithm Complexity

This file contains the reference solutions and analysis.
"""

def find_max(numbers):
    """
    Snippet 1: Already optimal
    Time Complexity: O(n) - must check every element
    Space Complexity: O(1) - only uses constant extra space
    """
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val


def has_duplicate_optimized(numbers):
    """
    Optimized version using set - O(n) time, O(n) space
    Original was O(n²) time, O(1) space
    """
    seen = set()
    for num in numbers:
        if num in seen:
            return True
        seen.add(num)
    return False


def reverse_list_optimized(lst):
    """
    Optimized version - in-place reversal O(n) time, O(1) space
    Original was O(n) time, O(n) space
    
    Option 1: In-place reversal (modifies original)
    """
    n = len(lst)
    for i in range(n // 2):
        lst[i], lst[n - 1 - i] = lst[n - 1 - i], lst[i]
    return lst


def reverse_list_optimized_v2(lst):
    """
    Option 2: Using slicing (but creates new list, so O(n) space)
    Most Pythonic way
    """
    return lst[::-1]


def reverse_list_optimized_v3(lst):
    """
    Option 3: Using built-in reversed (creates iterator, O(1) space for iterator)
    """
    return list(reversed(lst))
