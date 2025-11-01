"""
Exercise 2: String Builder Optimization

Optimize the build_string function to improve time complexity.
The naive approach using += for string concatenation is inefficient
because strings are immutable in Python.
"""

def build_string(parts):
    """
    Current inefficient implementation - uses += for concatenation.
    DO NOT MODIFY THIS FUNCTION - it's used for comparison in tests.
    """
    result = ""
    for part in parts:
        result += part
    return result


def build_string_optimized(parts):
    """
    Your optimized implementation here.
    
    Requirements:
    - Handle lists with 10,000+ string parts efficiently
    - Maintain functionality: concatenate all parts in order
    - Improve time complexity compared to += approach
    """
    pass


if __name__ == "__main__":
    # Basic test
    test_data = ["Hello", " ", "World", "!"]
    result = build_string_optimized(test_data)
    print(f"Result: {result}")
    print(f"Expected: 'Hello World!'")
