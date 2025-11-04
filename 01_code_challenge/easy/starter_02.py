"""
Exercise 2: String Builder Optimization

Optimize the build_string function to improve time complexity.
The naive approach using += for string concatenation is inefficient
because strings are immutable in Python.
"""

from io import StringIO


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
    # REVIEW: For pre-collected parts, "".join(parts) is typically the fastest
    # approach because it computes the total size once and performs a single
    # allocation/copy. StringIO is still O(n) and shines when parts are produced
    # incrementally or conditionally. Both pass; pick based on data flow.
    #
    # Example alternative (uncomment to use):
    # result = "".join(parts)
    # return result

    buf = StringIO()
    for part in parts:
        buf.write(part)
    return buf.getvalue()

# REVIEW: Ensure all elements in `parts` are strings. If types may vary, wrap as
# `buf.write(str(part))` or use `"".join(map(str, parts))` to make it robust.


if __name__ == "__main__":
    # Basic test
    test_data = ["Hello", " ", "World", "!"]
    result = build_string_optimized(test_data)
    print(f"Result: {result}")
    print(f"Expected: 'Hello World!'")
