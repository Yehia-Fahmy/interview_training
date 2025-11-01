"""
Exercise 4: Understanding Algorithm Complexity

Analyze and optimize the given code snippets. Determine time and space
complexity, and provide optimized versions where possible.
"""

def find_max(numbers):
    """
    Snippet 1: Find maximum value in a list.
    Analyze the time and space complexity.
    """
    # REVIEW: Straightforward single-pass max. Time: O(n), Space: O(1).
    # Minor: you can start from the first element and iterate the rest to avoid
    # a redundant comparison of numbers[0] with itself (micro-optimization).
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val


def has_duplicate(numbers):
    """
    Snippet 2: Check if list has duplicates.
    Analyze complexity and optimize if possible.
    """
    # REVIEW: Correct but O(n^2). Kept for contrast with the optimized version.
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j]:
                return True
    return False


def reverse_list(lst):
    """
    Snippet 3: Reverse a list.
    Analyze complexity and optimize if possible.
    """
    # REVIEW: Returns a new reversed list in O(n) time and O(n) space. Clear.
    n = len(lst)
    result = []
    for i in range(n):
        result.append(lst[n - 1 - i])
    return result


# Your optimized implementations
def has_duplicate_optimized(numbers):
    """
    Optimized version of has_duplicate.
    Improve the time complexity.
    """
    # REVIEW: Good use of a set: O(n) time, O(n) space. Early exit keeps it fast.
    seen = set()
    for n in numbers:
        if n in seen: return True
        seen.add(n)
    return False


def reverse_list_optimized(lst):
    """
    Optimized version of reverse_list.
    Consider improving space complexity or using in-place reversal.
    """
    # REVIEW: In-place reversal in O(n) time and O(1) extra space. Note this
    # mutates the input list; document this behavior so callers are aware.
    # Alternatives: `lst[::-1]` (new list) or `lst.reverse()` (in-place).
    l = len(lst)
    for i in range(l // 2):
        tmp = lst[i]
        lst[i] = lst[l - i - 1]
        lst[l - i - 1] = tmp
    return lst


if __name__ == "__main__":
    # Test find_max
    test1 = [3, 1, 4, 1, 5, 9, 2, 6]
    print(f"Max of {test1}: {find_max(test1)}")  # Expected: 9
    
    # Test has_duplicate
    test2 = [1, 2, 3, 4, 5]
    test3 = [1, 2, 3, 2, 5]
    print(f"has_duplicate({test2}): {has_duplicate(test2)}")  # Expected: False
    print(f"has_duplicate({test3}): {has_duplicate(test3)}")  # Expected: True
    
    # Test reverse_list
    test4 = [1, 2, 3, 4, 5]
    print(f"reverse_list({test4}): {reverse_list(test4)}")  # Expected: [5, 4, 3, 2, 1]
