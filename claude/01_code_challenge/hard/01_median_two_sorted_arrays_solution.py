"""
Solution: Median of Two Sorted Arrays

This is a challenging problem that requires binary search thinking.
"""

from typing import List


def find_median_sorted_arrays_merge(nums1: List[int], nums2: List[int]) -> float:
    """
    Approach 1: Merge arrays and find median
    Time Complexity: O(m + n)
    Space Complexity: O(m + n)
    
    Simple but doesn't meet the O(log(min(m,n))) requirement.
    """
    # Merge two sorted arrays
    merged = []
    i, j = 0, 0
    
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1
    
    # Add remaining elements
    merged.extend(nums1[i:])
    merged.extend(nums2[j:])
    
    # Find median
    n = len(merged)
    if n % 2 == 0:
        return (merged[n // 2 - 1] + merged[n // 2]) / 2
    else:
        return float(merged[n // 2])


def find_median_sorted_arrays(nums1: List[int], nums2: List[int]) -> float:
    """
    Approach 2: Binary search on partition (Optimal)
    Time Complexity: O(log(min(m, n)))
    Space Complexity: O(1)
    
    Key Insight:
    The median divides the combined array into two equal halves.
    We can use binary search to find the correct partition point.
    
    Visualization for [1,3,5,7] and [2,4,6,8]:
    
    Combined: [1,2,3,4 | 5,6,7,8]
    Median = (4 + 5) / 2 = 4.5
    
    We need to partition both arrays such that:
    1. Total elements on left = Total elements on right
    2. max(left_part) <= min(right_part)
    
    Array 1: [1, 3 | 5, 7]
    Array 2: [2, 4 | 6, 8]
    
    Left part: {1, 3, 2, 4}, Right part: {5, 7, 6, 8}
    max(left) = 4, min(right) = 5
    Median = (4 + 5) / 2 = 4.5
    
    Algorithm:
    1. Ensure nums1 is the smaller array (for efficiency)
    2. Binary search on nums1 to find partition
    3. Calculate corresponding partition in nums2
    4. Check if partition is valid:
       - maxLeft1 <= minRight2
       - maxLeft2 <= minRight1
    5. If valid, calculate median
    6. If not, adjust binary search bounds
    """
    # Ensure nums1 is smaller for efficiency
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    
    # Binary search on nums1
    left, right = 0, m
    
    while left <= right:
        # Partition nums1
        partition1 = (left + right) // 2
        
        # Partition nums2 to balance the halves
        partition2 = (m + n + 1) // 2 - partition1
        
        # Handle edge cases where partition is at boundary
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        # Check if we found the correct partition
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            # Found the correct partition!
            # Calculate median based on total length (odd or even)
            if (m + n) % 2 == 0:
                return (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
            else:
                return float(max(maxLeft1, maxLeft2))
        
        elif maxLeft1 > minRight2:
            # Too far right in nums1, move left
            right = partition1 - 1
        else:
            # Too far left in nums1, move right
            left = partition1 + 1
    
    # Should never reach here if inputs are valid
    raise ValueError("Input arrays are not sorted")


def find_median_sorted_arrays_with_trace(nums1: List[int], nums2: List[int]) -> float:
    """
    Version with detailed trace for learning
    """
    print(f"Finding median of {nums1} and {nums2}")
    print(f"Total length: {len(nums1) + len(nums2)}")
    print()
    
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
        print(f"Swapped arrays to make nums1 smaller")
        print(f"nums1: {nums1}, nums2: {nums2}\n")
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    iteration = 0
    
    while left <= right:
        iteration += 1
        print(f"Iteration {iteration}:")
        print(f"  Binary search range: [{left}, {right}]")
        
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        print(f"  Partition1: {partition1}, Partition2: {partition2}")
        
        # Get boundary values
        maxLeft1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        minRight1 = float('inf') if partition1 == m else nums1[partition1]
        maxLeft2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        minRight2 = float('inf') if partition2 == n else nums2[partition2]
        
        print(f"  nums1: {nums1[:partition1]} | {nums1[partition1:]}")
        print(f"  nums2: {nums2[:partition2]} | {nums2[partition2:]}")
        print(f"  maxLeft1={maxLeft1}, minRight1={minRight1}")
        print(f"  maxLeft2={maxLeft2}, minRight2={minRight2}")
        
        # Check partition validity
        if maxLeft1 <= minRight2 and maxLeft2 <= minRight1:
            print(f"  ✓ Valid partition found!")
            
            if (m + n) % 2 == 0:
                median = (max(maxLeft1, maxLeft2) + min(minRight1, minRight2)) / 2
                print(f"  Even length: median = ({max(maxLeft1, maxLeft2)} + {min(minRight1, minRight2)}) / 2 = {median}")
            else:
                median = float(max(maxLeft1, maxLeft2))
                print(f"  Odd length: median = {median}")
            
            return median
        
        elif maxLeft1 > minRight2:
            print(f"  maxLeft1 ({maxLeft1}) > minRight2 ({minRight2})")
            print(f"  → Move left in nums1")
            right = partition1 - 1
        else:
            print(f"  maxLeft2 ({maxLeft2}) > minRight1 ({minRight1})")
            print(f"  → Move right in nums1")
            left = partition1 + 1
        
        print()
    
    raise ValueError("Input arrays are not sorted")


# Test cases
def test_all_solutions():
    print("Testing all solutions...\n")
    
    test_cases = [
        ([1, 3], [2], 2.0),
        ([1, 2], [3, 4], 2.5),
        ([], [1], 1.0),
        ([2], [], 2.0),
        ([1, 2, 3], [4, 5, 6, 7, 8], 4.5),
        ([1, 1], [1, 2], 1.0),
        ([-5, -3, -1], [0, 2, 4], -0.5),
    ]
    
    solutions = [
        ("Merge approach", find_median_sorted_arrays_merge),
        ("Binary search", find_median_sorted_arrays),
    ]
    
    for name, func in solutions:
        print(f"Testing {name}...")
        for nums1, nums2, expected in test_cases:
            result = func(nums1, nums2)
            status = "✓" if abs(result - expected) < 1e-5 else "✗"
            print(f"  {status} {nums1}, {nums2} -> {result} (expected {expected})")
        print()
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_all_solutions()
    
    # Detailed walkthrough
    print("\n" + "="*70)
    print("DETAILED WALKTHROUGH")
    print("="*70 + "\n")
    find_median_sorted_arrays_with_trace([1, 3, 5, 7], [2, 4, 6, 8])
    
    print("\n" + "="*70)
    find_median_sorted_arrays_with_trace([1, 2], [3, 4])
    
    # Complexity analysis
    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS")
    print("="*70)
    print("""
Merge Approach:
- Time: O(m + n) - merge both arrays
- Space: O(m + n) - store merged array
- Pros: Simple to understand and implement
- Cons: Doesn't meet O(log(min(m,n))) requirement

Binary Search Approach:
- Time: O(log(min(m, n))) - binary search on smaller array
- Space: O(1) - only use constant extra space
- Pros: Optimal time complexity
- Cons: Complex to understand and implement

Why Binary Search?
- We're searching for the correct partition point
- The search space is the smaller array (size m)
- Each iteration eliminates half the search space
- Result: O(log m) where m = min(m, n)
    """)
    
    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. Median Property:
   - Divides array into two equal halves
   - All elements in left half <= all elements in right half

2. Partition Approach:
   - Instead of merging, find where to partition each array
   - If we partition nums1 at i, we must partition nums2 at j
   - Where j = (m + n + 1) // 2 - i (to balance halves)

3. Valid Partition Conditions:
   - maxLeft1 <= minRight2 (nums1 left <= nums2 right)
   - maxLeft2 <= minRight1 (nums2 left <= nums1 right)

4. Binary Search Logic:
   - If maxLeft1 > minRight2: partition1 too far right, move left
   - If maxLeft2 > minRight1: partition1 too far left, move right

5. Edge Cases:
   - Empty arrays: use -inf and +inf for boundaries
   - Odd vs even total length: affects median calculation
    """)
    
    # Common mistakes
    print("\n" + "="*70)
    print("COMMON MISTAKES")
    print("="*70)
    print("""
1. Not ensuring nums1 is smaller
   → Binary search should be on smaller array for efficiency

2. Off-by-one in partition calculation
   ✗ partition2 = (m + n) // 2 - partition1
   ✓ partition2 = (m + n + 1) // 2 - partition1
   (The +1 handles both odd and even cases correctly)

3. Not handling edge cases
   → Use -inf and +inf when partition is at array boundary

4. Wrong median calculation
   - Even length: average of two middle elements
   - Odd length: the middle element (max of left halves)

5. Forgetting to check both partition conditions
   → Must verify BOTH maxLeft1 <= minRight2 AND maxLeft2 <= minRight1
    """)

