"""
Solution: Longest Substring Without Repeating Characters

Multiple approaches with detailed explanations.
"""


def length_of_longest_substring_brute_force(s: str) -> int:
    """
    Brute force: Check all substrings
    Time Complexity: O(n²) - two nested loops
    Space Complexity: O(min(n, m)) - set to track characters
    """
    def has_unique_chars(substr: str) -> bool:
        return len(substr) == len(set(substr))
    
    n = len(s)
    max_length = 0
    
    for i in range(n):
        for j in range(i + 1, n + 1):
            substr = s[i:j]
            if has_unique_chars(substr):
                max_length = max(max_length, len(substr))
    
    return max_length


def length_of_longest_substring(s: str) -> int:
    """
    Sliding window with hash table (Optimal)
    Time Complexity: O(n) - each character visited at most twice
    Space Complexity: O(min(n, m)) where m is charset size
    
    Key Insight:
    Use a sliding window [left, right] to represent current substring.
    - Expand window by moving right pointer
    - When duplicate found, contract from left until no duplicate
    - Track characters in window using a set or dict
    
    Algorithm:
    1. Use two pointers: left and right (window boundaries)
    2. Use a set to track characters in current window
    3. Expand window by moving right pointer
    4. If character already in window:
       - Remove characters from left until duplicate is gone
    5. Track maximum window size seen
    """
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # If character already in window, shrink from left
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        
        # Add current character to window
        char_set.add(s[right])
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length


def length_of_longest_substring_optimized(s: str) -> int:
    """
    Optimized sliding window with hash map
    Time Complexity: O(n) - single pass
    Space Complexity: O(min(n, m))
    
    Further optimization: Instead of removing characters one by one,
    jump left pointer directly to the position after the duplicate.
    """
    char_index = {}  # {character: last_seen_index}
    left = 0
    max_length = 0
    
    for right, char in enumerate(s):
        # If character seen before and is in current window
        if char in char_index and char_index[char] >= left:
            # Jump left pointer to position after the duplicate
            left = char_index[char] + 1
        
        # Update character's last seen position
        char_index[char] = right
        
        # Update max length
        max_length = max(max_length, right - left + 1)
    
    return max_length


def length_of_longest_substring_with_trace(s: str) -> int:
    """
    Version with detailed trace for learning
    """
    print(f"Finding longest substring without repeating chars in: '{s}'")
    print()
    
    char_set = set()
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        print(f"Step {right + 1}: right={right}, char='{s[right]}'")
        
        # Shrink window if duplicate found
        while s[right] in char_set:
            print(f"  Duplicate found! Removing '{s[left]}' from left")
            char_set.remove(s[left])
            left += 1
        
        # Add current character
        char_set.add(s[right])
        window_size = right - left + 1
        
        print(f"  Window: [{left}:{right+1}] = '{s[left:right+1]}'")
        print(f"  Characters in window: {char_set}")
        print(f"  Window size: {window_size}")
        
        if window_size > max_length:
            max_length = window_size
            print(f"  ★ New max length: {max_length}")
        
        print()
    
    print(f"Result: {max_length}")
    return max_length


# Test cases
def test_all_solutions():
    print("Testing all solutions...\n")
    
    test_cases = [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("", 0),
        ("abcdef", 6),
        ("a", 1),
        ("au", 2),
        ("dvdf", 3),
    ]
    
    solutions = [
        ("Brute Force", length_of_longest_substring_brute_force),
        ("Sliding Window (Set)", length_of_longest_substring),
        ("Sliding Window (Dict)", length_of_longest_substring_optimized),
    ]
    
    for name, func in solutions:
        print(f"Testing {name}...")
        for s, expected in test_cases:
            result = func(s)
            status = "✓" if result == expected else "✗"
            print(f"  {status} '{s}' -> {result} (expected {expected})")
        print()
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    # Run tests
    test_all_solutions()
    
    # Detailed walkthrough
    print("\n" + "="*70)
    print("DETAILED WALKTHROUGH")
    print("="*70 + "\n")
    length_of_longest_substring_with_trace("abcabcbb")
    
    # Complexity analysis
    print("\n" + "="*70)
    print("COMPLEXITY ANALYSIS")
    print("="*70)
    print("""
Brute Force:
- Time: O(n²) - check all substrings
- Space: O(min(n, m)) - set for checking uniqueness
- Approach: Generate all substrings, check each for uniqueness

Sliding Window (Set):
- Time: O(n) - each char visited at most twice (once by right, once by left)
- Space: O(min(n, m)) - set stores current window chars
- Approach: Expand window right, contract left when duplicate found

Sliding Window (Dict - Most Optimized):
- Time: O(n) - single pass, left pointer jumps directly
- Space: O(min(n, m)) - dict stores char positions
- Approach: Jump left pointer directly to after duplicate position

Key Insight:
The sliding window technique transforms a O(n²) problem into O(n)
by maintaining a valid window and only processing each element once.
    """)
    
    # Common patterns
    print("\n" + "="*70)
    print("SLIDING WINDOW PATTERN")
    print("="*70)
    print("""
The sliding window pattern is useful when you need to find:
- Longest/shortest substring with certain properties
- Maximum/minimum sum of subarray of size k
- Substrings with at most k distinct characters

Template:
    left = 0
    for right in range(len(array)):
        # Add array[right] to window
        
        while window_invalid():
            # Remove array[left] from window
            left += 1
        
        # Update result with current window
    
When to use:
- Problem involves contiguous sequence (substring/subarray)
- Need to find optimal window (longest/shortest/max/min)
- Can determine window validity in O(1) time
    """)
    
    # Common mistakes
    print("\n" + "="*70)
    print("COMMON MISTAKES")
    print("="*70)
    print("""
1. Not handling empty string
   ✓ Check if s is empty at start

2. Off-by-one in window size calculation
   ✗ max_length = right - left
   ✓ max_length = right - left + 1

3. Not removing from tracking structure when shrinking window
   ✗ left += 1  (forgot to remove s[left] from set)
   ✓ char_set.remove(s[left]); left += 1

4. Checking duplicate after adding to set
   ✗ char_set.add(s[right]); if s[right] in char_set
   ✓ if s[right] in char_set: ...; char_set.add(s[right])
    """)

