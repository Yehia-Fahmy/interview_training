"""
Problem: Median of Two Sorted Arrays

Difficulty: Hard
Time: 45-60 minutes

Description:
Given two sorted arrays nums1 and nums2 of size m and n respectively, return 
the median of the two sorted arrays.

The overall run time complexity should be O(log(min(m,n))).

Constraints:
- nums1.length == m
- nums2.length == n
- 0 <= m <= 1000
- 0 <= n <= 1000
- 1 <= m + n <= 2000
- -10^6 <= nums1[i], nums2[i] <= 10^6

Examples:
    Input: nums1 = [1,3], nums2 = [2]
    Output: 2.00000
    Explanation: merged = [1,2,3], median = 2

    Input: nums1 = [1,2], nums2 = [3,4]
    Output: 2.50000
    Explanation: merged = [1,2,3,4], median = (2 + 3) / 2 = 2.5

Learning Objectives:
- Binary search on answer space
- Partitioning technique
- Handling edge cases with empty arrays
- O(log n) complexity analysis
- Working with sorted arrays

Hints:
1. Merging arrays would be O(m+n) - not optimal
2. Can you use binary search? What are you searching for?
3. Think about partitioning both arrays such that:
   - Left half has same number of elements as right half
   - All elements in left half <= all elements in right half
4. The median is at the partition point!
"""

from typing import List


def find_median_sorted_arrays_merge(nums1: List[int], nums2: List[int]) -> float:
    """
    Approach 1: Merge and find median
    Time Complexity: O(m + n)
    Space Complexity: O(m + n)
    
    This works but doesn't meet the O(log(min(m,n))) requirement.
    Good starting point to understand the problem.
    """
    # TODO: Implement merge approach
    pass


def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Approach 2: Binary search on partition
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    
    This is the optimal solution that meets the requirement.
    """
    # TODO: Implement binary search approach
    pass


# Test cases
def test_find_median_sorted_arrays():
    # Test case 1
    assert find_median_sorted_arrays([1, 3], [2]) == 2.0
    
    # Test case 2
    assert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5
    
    # Test case 3: One empty array
    assert find_median_sorted_arrays([], [1]) == 1.0
    
    # Test case 4: Different sizes
    assert find_median_sorted_arrays([1, 2, 3], [4, 5, 6, 7, 8]) == 4.5
    
    # Test case 5: Duplicates
    assert find_median_sorted_arrays([1, 1], [1, 2]) == 1.0
    
    # Test case 6: Negative numbers
    assert find_median_sorted_arrays([-5, -3, -1], [0, 2, 4]) == -0.5
    
    print("All tests passed!")


if __name__ == "__main__":
    test_find_median_sorted_arrays()
    
    # Example usage
    test_cases = [
        ([1, 3], [2]),
        ([1, 2], [3, 4]),
        ([1, 2, 3, 4], [5, 6, 7, 8])
    ]
    
    for nums1, nums2 in test_cases:
        result = find_median_sorted_arrays(nums1, nums2)
        print(f"nums1={nums1}, nums2={nums2} -> median={result}")

