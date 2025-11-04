"""
Solution for Exercise 3: Choosing the Right Data Structure

This file contains the reference solutions.
"""

def find_pairs_brute_force(numbers, target):
    """
    Brute force approach - O(n²) time, O(1) space
    """
    pairs = []
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if numbers[i] + numbers[j] == target:
                pairs.append((numbers[i], numbers[j]))
    return pairs


def find_pairs_optimized(numbers, target):
    """
    Optimized approach using set - O(n) time, O(n) space
    """
    pairs = []
    seen = set()
    
    for num in numbers:
        complement = target - num
        if complement in seen:
            # Avoid duplicates by only storing the smaller number first
            if num < complement:
                pairs.append((num, complement))
            else:
                pairs.append((complement, num))
        seen.add(num)
    
    return pairs


# Alternative that handles edge cases better
def find_pairs_optimized_v2(numbers, target):
    """
    Alternative that ensures no duplicate pairs and sorted order
    """
    pairs = []
    seen = set()
    result_set = set()  # Use set to avoid duplicate pairs
    
    for num in numbers:
        complement = target - num
        if complement in seen:
            # Create sorted tuple to ensure consistent ordering
            pair = tuple(sorted([num, complement]))
            if pair not in result_set:
                result_set.add(pair)
                pairs.append(pair)
        seen.add(num)
    
    return sorted(pairs)
