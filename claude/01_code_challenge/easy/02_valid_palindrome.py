"""
Problem: Valid Palindrome

Difficulty: Easy
Time: 15-20 minutes

Description:
A phrase is a palindrome if, after converting all uppercase letters into lowercase 
letters and removing all non-alphanumeric characters, it reads the same forward 
and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

Constraints:
- 1 <= s.length <= 2 * 10^5
- s consists only of printable ASCII characters

Examples:
    Input: s = "A man, a plan, a canal: Panama"
    Output: true
    Explanation: "amanaplanacanalpanama" is a palindrome

    Input: s = "race a car"
    Output: false
    Explanation: "raceacar" is not a palindrome

    Input: s = " "
    Output: true
    Explanation: Empty string after removing non-alphanumeric is palindrome

Learning Objectives:
- Two-pointer technique
- String manipulation
- In-place comparison without extra space
- Character validation

Hints:
1. You could clean the string first, but that uses O(n) extra space
2. Can you check palindrome while filtering?
3. Two pointers from both ends moving inward
"""


def is_palindrome_simple(s: str) -> bool:
    """
    Simple approach: Clean string first, then check
    Time Complexity: O(n)
    Space Complexity: O(n) - creates new cleaned string
    
    Good starting point but not optimal for space.
    """
    # TODO: Implement simple solution
    pass


def is_palindrome(s: str) -> bool:
    """
    Optimized approach: Two pointers without extra space
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    This is the optimal solution.
    """
    # TODO: Implement optimal solution
    pass


# Test cases
def test_is_palindrome():
    # Test case 1: Standard palindrome
    assert is_palindrome("A man, a plan, a canal: Panama") == True
    
    # Test case 2: Not a palindrome
    assert is_palindrome("race a car") == False
    
    # Test case 3: Empty/whitespace
    assert is_palindrome(" ") == True
    
    # Test case 4: Single character
    assert is_palindrome("a") == True
    
    # Test case 5: Numbers and letters
    assert is_palindrome("0P") == False
    
    # Test case 6: Only special characters
    assert is_palindrome(".,") == True
    
    print("All tests passed!")


if __name__ == "__main__":
    test_is_palindrome()
    
    # Example usage
    test_strings = [
        "A man, a plan, a canal: Panama",
        "race a car",
        "Was it a car or a cat I saw?"
    ]
    
    for s in test_strings:
        result = is_palindrome(s)
        print(f"'{s}' -> {result}")

