"""
Solution: Valid Palindrome

This file contains multiple approaches with detailed explanations.
"""


def is_palindrome_simple(s: str) -> bool:
    """
    Simple approach: Clean string first, then check
    Time Complexity: O(n)
    Space Complexity: O(n) - creates new cleaned string
    """
    # Clean the string: lowercase and keep only alphanumeric
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    
    # Check if it's a palindrome
    return cleaned == cleaned[::-1]


def is_palindrome(s: str) -> bool:
    """
    Optimized approach: Two pointers without extra space
    Time Complexity: O(n)
    Space Complexity: O(1) - only uses two pointer variables
    
    Key Insight:
    Instead of creating a cleaned string, we can use two pointers
    and skip non-alphanumeric characters on the fly.
    
    Algorithm:
    1. Use left pointer starting at beginning
    2. Use right pointer starting at end
    3. Skip non-alphanumeric characters
    4. Compare characters (case-insensitive)
    5. If mismatch found, return False
    6. If pointers meet, return True
    """
    left = 0
    right = len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        # Move both pointers
        left += 1
        right -= 1
    
    return True


def is_palindrome_pythonic(s: str) -> bool:
    """
    Most Pythonic version using filter and itertools
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Filter and lowercase in one pass
    chars = [c.lower() for c in s if c.isalnum()]
    return chars == chars[::-1]


def is_palindrome_with_trace(s: str) -> bool:
    """
    Version with detailed trace for learning
    """
    print(f"Checking: '{s}'")
    left = 0
    right = len(s) - 1
    step = 0
    
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            print(f"  Skipping '{s[left]}' at left position {left}")
            left += 1
        
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            print(f"  Skipping '{s[right]}' at right position {right}")
            right -= 1
        
        step += 1
        left_char = s[left].lower()
        right_char = s[right].lower()
        
        print(f"Step {step}: Compare '{left_char}' (pos {left}) with '{right_char}' (pos {right})")
        
        if left_char != right_char:
            print(f"  ✗ Mismatch! Not a palindrome")
            return False
        
        print(f"  ✓ Match!")
        left += 1
        right -= 1
    
    print(f"✓ Is a palindrome!")
    return True


# Test cases
def test_is_palindrome():
    print("Running tests...\n")
    
    # Test case 1
    assert is_palindrome("A man, a plan, a canal: Panama") == True
    print("✓ Test 1 passed: 'A man, a plan, a canal: Panama'")
    
    # Test case 2
    assert is_palindrome("race a car") == False
    print("✓ Test 2 passed: 'race a car'")
    
    # Test case 3
    assert is_palindrome(" ") == True
    print("✓ Test 3 passed: ' ' (empty after cleaning)")
    
    # Test case 4
    assert is_palindrome("a") == True
    print("✓ Test 4 passed: single character")
    
    # Test case 5
    assert is_palindrome("0P") == False
    print("✓ Test 5 passed: numbers and letters")
    
    # Test case 6
    assert is_palindrome(".,") == True
    print("✓ Test 6 passed: only special characters")
    
    # Test case 7
    assert is_palindrome("ab_a") == True
    print("✓ Test 7 passed: with underscore")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_is_palindrome()
    
    # Detailed trace
    print("\n" + "="*60)
    print("DETAILED WALKTHROUGH")
    print("="*60 + "\n")
    is_palindrome_with_trace("A man, a plan, a canal: Panama")
    
    print("\n" + "="*60)
    is_palindrome_with_trace("race a car")
    
    # Complexity comparison
    print("\n" + "="*60)
    print("COMPLEXITY COMPARISON")
    print("="*60)
    print("""
Simple Approach (Clean First):
- Time: O(n) - one pass to clean, one to check
- Space: O(n) - stores cleaned string
- Pros: Easy to understand and implement
- Cons: Uses extra memory

Two Pointers (Optimal):
- Time: O(n) - single pass with two pointers
- Space: O(1) - only pointer variables
- Pros: Minimal memory usage
- Cons: Slightly more complex logic

Pythonic Approach:
- Time: O(n)
- Space: O(n)
- Pros: Most readable
- Cons: Uses extra memory

Key Takeaway:
Two-pointer technique is powerful for in-place string/array operations.
It's a must-know pattern for optimization interviews.
    """)
    
    # Common mistakes
    print("\n" + "="*60)
    print("COMMON MISTAKES TO AVOID")
    print("="*60)
    print("""
1. Forgetting case-insensitive comparison
   ✗ if s[left] != s[right]
   ✓ if s[left].lower() != s[right].lower()

2. Not handling non-alphanumeric characters
   ✗ Comparing punctuation directly
   ✓ Skip non-alphanumeric with isalnum()

3. Off-by-one errors with pointers
   ✗ while left <= right  (can compare same char twice)
   ✓ while left < right

4. Not checking bounds when skipping
   ✗ while not s[left].isalnum(): left += 1
   ✓ while left < right and not s[left].isalnum(): left += 1
    """)

