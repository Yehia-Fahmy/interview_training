"""
Solution: Two Sum

This file contains the solution with detailed explanations.
Try to solve it yourself first before looking at this!
"""

from typing import List


def two_sum_brute_force(nums: List[int], target: int) -> List[int]:
    """
    Brute force approach: Check all pairs
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []  # No solution found


def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Optimized approach using hash table
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(n) - hash table stores up to n elements
    
    Key Insight:
    For each number x, we need to find (target - x).
    Instead of searching the entire array each time (O(n)),
    we can use a hash table to check in O(1) time.
    
    Algorithm:
    1. Create a hash table to store {value: index}
    2. For each number in the array:
       - Calculate complement = target - current_number
       - Check if complement exists in hash table
       - If yes, return [complement_index, current_index]
       - If no, add current number to hash table
    """
    seen = {}  # {value: index}
    
    for i, num in enumerate(nums):
        complement = target - num
        
        # Check if we've seen the complement before
        if complement in seen:
            return [seen[complement], i]
        
        # Store current number and its index
        seen[num] = i
    
    return []  # No solution found (shouldn't happen per problem constraints)


# Alternative solution using enumerate
def two_sum_pythonic(nums: List[int], target: int) -> List[int]:
    """
    More Pythonic version with the same logic
    """
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
    return []


# Detailed walkthrough example
def two_sum_with_trace(nums: List[int], target: int) -> List[int]:
    """
    Version with detailed trace for learning
    """
    seen = {}
    print(f"Looking for two numbers that sum to {target}")
    print(f"Array: {nums}\n")
    
    for i, num in enumerate(nums):
        complement = target - num
        print(f"Step {i + 1}: num = {num}, complement = {complement}")
        print(f"  Seen so far: {seen}")
        
        if complement in seen:
            print(f"  ✓ Found! {complement} at index {seen[complement]}, {num} at index {i}")
            return [seen[complement], i]
        
        seen[num] = i
        print(f"  Added {num} to seen\n")
    
    return []


# Test cases
def test_two_sum():
    print("Running tests...\n")
    
    # Test case 1
    result = two_sum([2, 7, 11, 15], 9)
    assert sorted(result) == [0, 1], f"Test 1 failed: {result}"
    print("✓ Test 1 passed: [2,7,11,15], target=9 -> [0,1]")
    
    # Test case 2
    result = two_sum([3, 2, 4], 6)
    assert sorted(result) == [1, 2], f"Test 2 failed: {result}"
    print("✓ Test 2 passed: [3,2,4], target=6 -> [1,2]")
    
    # Test case 3
    result = two_sum([3, 3], 6)
    assert sorted(result) == [0, 1], f"Test 3 failed: {result}"
    print("✓ Test 3 passed: [3,3], target=6 -> [0,1]")
    
    # Edge case: negative numbers
    result = two_sum([-1, -2, -3, -4, -5], -8)
    assert sorted(result) in [[2, 4], [3, 4]], f"Test 4 failed: {result}"
    print("✓ Test 4 passed: negative numbers")
    
    # Edge case: zero
    result = two_sum([0, 4, 3, 0], 0)
    assert sorted(result) == [0, 3], f"Test 5 failed: {result}"
    print("✓ Test 5 passed: with zeros")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_two_sum()
    
    # Detailed trace example
    print("\n" + "="*50)
    print("DETAILED WALKTHROUGH")
    print("="*50 + "\n")
    two_sum_with_trace([2, 7, 11, 15], 9)
    
    # Compare approaches
    print("\n" + "="*50)
    print("COMPLEXITY COMPARISON")
    print("="*50)
    print("""
Brute Force:
- Time: O(n²) - nested loops
- Space: O(1) - no extra storage
- When to use: Very small arrays (n < 10)

Hash Table (Optimal):
- Time: O(n) - single pass
- Space: O(n) - hash table storage
- When to use: Always for this problem!

Key Takeaway:
Trading space for time is often worth it. O(n) space for O(n²) → O(n) time
is an excellent trade-off.
    """)

