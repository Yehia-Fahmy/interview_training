"""
Problem: Longest Substring Without Repeating Characters

Difficulty: Medium
Time: 30-45 minutes

Description:
Given a string s, find the length of the longest substring without repeating characters.

Constraints:
- 0 <= s.length <= 5 * 10^4
- s consists of English letters, digits, symbols and spaces

Examples:
    Input: s = "abcabcbb"
    Output: 3
    Explanation: The answer is "abc", with length 3

    Input: s = "bbbbb"
    Output: 1
    Explanation: The answer is "b", with length 1

    Input: s = "pwwkew"
    Output: 3
    Explanation: The answer is "wke", with length 3
    Note: "pwke" is a subsequence, not a substring

    Input: s = ""
    Output: 0

Learning Objectives:
- Sliding window technique
- Hash table for tracking characters
- Window expansion and contraction
- Optimization from O(n²) to O(n)

Hints:
1. Brute force: Check all substrings - O(n³) or O(n²)
2. Can you use a "window" that slides through the string?
3. What happens when you find a duplicate?
4. How do you track which characters are in the current window?
"""


def length_of_longest_substring_brute_force(s: str) -> int:
    """
    Brute force: Check all substrings
    Time Complexity: O(n²) or O(n³) depending on implementation
    Space Complexity: O(min(n, m)) where m is charset size
    
    Not optimal but good to understand the problem.
    """
    # TODO: Implement brute force
    pass


def length_of_longest_substring(s: str) -> int:
    """
    Sliding window with hash table
    Time Complexity: O(n) - each character visited at most twice
    Space Complexity: O(min(n, m)) where m is charset size
    
    This is the optimal solution.
    """
    # TODO: Implement optimal solution
    pass


# Test cases
def test_length_of_longest_substring():
    # Test case 1
    assert length_of_longest_substring("abcabcbb") == 3
    
    # Test case 2
    assert length_of_longest_substring("bbbbb") == 1
    
    # Test case 3
    assert length_of_longest_substring("pwwkew") == 3
    
    # Test case 4: Empty string
    assert length_of_longest_substring("") == 0
    
    # Test case 5: All unique
    assert length_of_longest_substring("abcdef") == 6
    
    # Test case 6: Single character
    assert length_of_longest_substring("a") == 1
    
    # Test case 7: With spaces and symbols
    assert length_of_longest_substring("a b c a") == 3
    
    print("All tests passed!")


if __name__ == "__main__":
    test_length_of_longest_substring()
    
    # Example usage
    test_cases = ["abcabcbb", "bbbbb", "pwwkew", ""]
    for s in test_cases:
        result = length_of_longest_substring(s)
        print(f"Input: '{s}' -> Output: {result}")

