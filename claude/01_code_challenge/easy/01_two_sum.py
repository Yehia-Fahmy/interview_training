"""
Problem: Two Sum

Difficulty: Easy
Time: 15-20 minutes

Description:
Given an array of integers `nums` and an integer `target`, return indices of the 
two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not 
use the same element twice.

You can return the answer in any order.

Constraints:
- 2 <= nums.length <= 10^4
- -10^9 <= nums[i] <= 10^9
- -10^9 <= target <= 10^9
- Only one valid answer exists

Examples:
    Input: nums = [2,7,11,15], target = 9
    Output: [0,1]
    Explanation: nums[0] + nums[1] == 9, so we return [0, 1]

    Input: nums = [3,2,4], target = 6
    Output: [1,2]

    Input: nums = [3,3], target = 6
    Output: [0,1]

Learning Objectives:
- Hash table for O(1) lookups
- Single-pass optimization
- Space-time trade-off

Hints:
1. Brute force is O(n²) - can you do better?
2. What if you stored what you've seen so far?
3. For each number, what are you looking for?
"""

from typing import List


def two_sum_brute_force(nums: List[int], target: int) -> List[int]:
    """
    Brute force approach: Check all pairs
    Time Complexity: O(n²)
    Space Complexity: O(1)
    
    This is NOT optimal but good to start with.
    """
    # TODO: Implement brute force solution
    pass


def two_sum(nums: List[int], target: int) -> List[int]:
    """
    Optimized approach using hash table
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    This is the optimal solution.
    """
    # TODO: Implement optimal solution
    pass


# Test cases
def test_two_sum():
    # Test case 1
    assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]
    
    # Test case 2
    assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]
    
    # Test case 3
    assert sorted(two_sum([3, 3], 6)) == [0, 1]
    
    # Edge case: negative numbers
    assert sorted(two_sum([-1, -2, -3, -4, -5], -8)) in [[2, 4], [3, 4]]
    
    # Edge case: zero
    assert sorted(two_sum([0, 4, 3, 0], 0)) == [0, 3]
    
    print("All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_two_sum()
    
    # Example usage
    nums = [2, 7, 11, 15]
    target = 9
    result = two_sum(nums, target)
    print(f"Input: nums = {nums}, target = {target}")
    print(f"Output: {result}")

