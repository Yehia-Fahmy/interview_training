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
    # REVIEW: Correct and passes the tests, but the duplicate-suppression logic
    # (via `seen` and `check`) makes the nested loops harder to follow. A
    # simpler brute-force can use indices i<j and a set of normalized pairs to
    # deduplicate. Complexity remains O(n^2) time, O(k) space for unique pairs.
    tuples = []
    seen = []
    def check(number):
        for n in seen:
            if number == n: return True
        return False

    for i in range(len(numbers)):
        if check(numbers[i]):
            continue
        for j in range(i+1, len(numbers)):
            if check(numbers[j]):
                continue
            if numbers[i] + numbers[j] == target:
                seen.append(numbers[j])
                tuples.append((numbers[i], numbers[j]))
        seen.append(numbers[i])
    return tuples


def find_pairs_optimized(numbers, target):
    """
    Optimized approach - use a set for O(1) lookups.
    
    Requirements:
    - Find all pairs (i, j) where i + j == target
    - Avoid duplicates (if (1, 2) is found, don't include (2, 1))
    - Return list of tuples
    - Use a set to achieve O(n) time complexity
    """
    # REVIEW: Good use of a set for O(1) complement lookups; overall O(n) time
    # and O(n) space. Returning pairs as (target - num, num) provides a stable
    # ordering and avoids (a,b)/(b,a) duplicates by skipping repeats.
    # Consider normalizing pair order or documenting it.
    seen = set()
    tuples = []
    for num in numbers:
        if num in seen: continue
        if (target - num) in seen:
            tuples.append((target - num, num))
        seen.add(num)

    return tuples


if __name__ == "__main__":
    # Basic test
    numbers = [1, 1, 2, 2]
    target = 3
    
    result1 = find_pairs_brute_force(numbers, target)
    result2 = find_pairs_optimized(numbers, target)
    
    print(f"Brute force result: {result1}")
    print(f"Optimized result: {result2}")
    
    # Note: Order may differ, but should contain same pairs
    print(f"Expected: [(1, 2)]")
