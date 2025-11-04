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

from math import e
import mmap
import struct
import os
import random


def generate_test_file(filename, num_integers=1000000):
    """
    Generate a binary file with integers for testing.
    
    Args:
        filename: Path to output file
        num_integers: Number of integers to write
    """
    num_chunks = 1000
    chunk_size = num_integers // num_chunks
    with open(filename, 'wb') as f:
        for i in range(num_chunks):
            chunk = [random.randint(-2147483648, 2147483647) for _ in range(chunk_size)]
            f.write(struct.pack(f"{chunk_size}i", *chunk))


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
    INT_SIZE = 4
    CHUNK_SIZE = 1000
    
    with open(filename, 'rb') as f:
        file_size = os.path.getsize(filename)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            total_ints = file_size // INT_SIZE

            for i in range(total_ints // CHUNK_SIZE):
                s = i * INT_SIZE
                end = min((i + CHUNK_SIZE) * INT_SIZE, file_size)
                chunk = mm[s:end]

                ints_in_chunk = len(chunk) // INT_SIZE
                a = struct.unpack(f"{ints_in_chunk}i", chunk)

                for n in a:
                    stats['min'] = min(n, stats['min'])
                    stats['max'] = max(n, stats['max'])
                    stats['sum'] += n
                stats['count'] += ints_in_chunk
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
    INT_SIZE = 4
    CHUNK_SIZE = 1000  # Number of integers per chunk
    BYTES_PER_CHUNK = CHUNK_SIZE * INT_SIZE
    
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(BYTES_PER_CHUNK)
            if not chunk:
                break
            
            ints_in_chunk = len(chunk) // INT_SIZE
            if ints_in_chunk == 0:
                break
            
            integers = struct.unpack(f"{ints_in_chunk}i", chunk)
            
            for n in integers:
                stats['min'] = min(n, stats['min'])
                stats['max'] = max(n, stats['max'])
                stats['sum'] += n
            stats['count'] += ints_in_chunk
    
    return stats


if __name__ == "__main__":
    test_file = 'large_data.bin'
    
    # Generate test file (uncomment when ready)
    generate_test_file(test_file, 10000000)
    
    # Compare approaches
    if os.path.exists(test_file):
        mmap_stats = process_with_mmap(test_file)
        std_stats = process_standard_io(test_file)
        
        print("Memory-mapped:", mmap_stats)
        print("Standard I/O:", std_stats)
    else:
        print(f"Test file {test_file} not found. Generate it first!")

