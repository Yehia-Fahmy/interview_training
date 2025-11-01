"""
Solution for Exercise 2: String Builder Optimization

This file contains the reference solution.
"""

def build_string_optimized(parts):
    """
    Optimized using str.join() which is O(n) time complexity
    compared to O(n²) for repeated += operations.
    """
    return ''.join(parts)


# Alternative solution using StringIO (useful for incremental building)
def build_string_optimized_v2(parts):
    """
    Using StringIO for cases where you need incremental building.
    """
    import io
    buffer = io.StringIO()
    for part in parts:
        buffer.write(part)
    return buffer.getvalue()
