"""
Solution for Exercise 3: Memory Profiling and Optimization

This file contains the reference solution.
"""

import os


def process_large_dataset_optimized(file_path):
    """
    Optimized version that processes data incrementally.
    Uses generators to avoid loading entire file into memory.
    """
    result = []
    
    # Process line by line instead of loading all at once
    with open(file_path, 'r') as f:
        for line in f:
            # Parse line directly without storing intermediate structures
            parts = line.strip().split(',')
            
            # Filter early - don't create dict if filtered out
            if not parts or not parts[0].isdigit():
                continue
            
            if int(parts[0]) > 100:
                # Only create the final transformed dict for filtered rows
                transformed = {
                    'id': parts[0],
                    'value': parts[1] if len(parts) > 1 else '',
                    'category': parts[2] if len(parts) > 2 else ''
                }
                result.append(transformed)
    
    return result


# Generator-based version for even better memory efficiency
def process_large_dataset_generator(file_path):
    """
    Generator version that yields results one at a time.
    This is the most memory-efficient approach.
    """
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
            if not parts or not parts[0].isdigit():
                continue
            
            if int(parts[0]) > 100:
                yield {
                    'id': parts[0],
                    'value': parts[1] if len(parts) > 1 else '',
                    'category': parts[2] if len(parts) > 2 else ''
                }


# If we need a list (for compatibility)
def process_large_dataset_optimized_list(file_path):
    """
    Optimized version that returns a list (for compatibility with tests).
    """
    return list(process_large_dataset_generator(file_path))

