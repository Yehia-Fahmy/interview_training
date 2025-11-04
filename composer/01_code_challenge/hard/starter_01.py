"""
Exercise 1: Memory-Mapped File Processing

Implement efficient processing of large binary files using memory-mapped files.
Handle files larger than available RAM without loading entire file into memory.

Requirements:
- Read and process a large binary file (e.g., 10GB+) containing integers
- Find the maximum value, minimum value, and sum
- Use memory mapping to avoid loading entire file into memory
- Handle edge cases (empty file, negative numbers, etc.)
"""

import mmap
import struct
import os


def generate_test_file(filename, num_integers=1000000):
    """
    Generate a binary file with integers for testing.
    
    Args:
        filename: Path to output file
        num_integers: Number of integers to write
    """
    # TODO: Implement file generation
    # Write 4-byte integers using struct.pack('i', value)
    pass


def process_with_mmap(filename):
    """
    Process file using memory mapping.
    
    Args:
        filename: Path to binary file
        
    Returns:
        dict with keys: 'min', 'max', 'sum', 'count'
    """
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # TODO: Implement using mmap
    # Hint: Use mmap.mmap() with ACCESS_READ
    # Process in chunks (e.g., 1MB at a time) to handle very large files
    # Use struct.unpack() to read integers
    
    with open(filename, 'rb') as f:
        # Your implementation here
        pass
    
    return stats


def process_standard_io(filename):
    """
    Process file using standard I/O (for comparison).
    
    Args:
        filename: Path to binary file
        
    Returns:
        dict with keys: 'min', 'max', 'sum', 'count'
    """
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # TODO: Implement standard file I/O version
    # Read file normally and process integers
    
    return stats


if __name__ == "__main__":
    test_file = 'large_data.bin'
    
    # Generate test file (uncomment when ready)
    # generate_test_file(test_file, 10000000)
    
    # Compare approaches
    # if os.path.exists(test_file):
    #     mmap_stats = process_with_mmap(test_file)
    #     std_stats = process_standard_io(test_file)
    #     
    #     print("Memory-mapped:", mmap_stats)
    #     print("Standard I/O:", std_stats)
    # else:
    #     print(f"Test file {test_file} not found. Generate it first!")

