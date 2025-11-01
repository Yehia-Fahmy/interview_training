# Exercise 1: Memory-Mapped File Processing

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Memory-mapped files, handling datasets larger than RAM

## Problem

You need to process a binary file containing millions of integers, where the file is larger than available RAM. Using standard file I/O causes memory issues.

**Requirements:**
- Read and process a large binary file (e.g., 10GB+) containing integers
- Find the maximum value, minimum value, and sum
- Use memory mapping to avoid loading entire file into memory
- Handle edge cases (empty file, negative numbers, etc.)

## Tasks

1. **Create** a binary file generator for testing
2. **Implement** processing using memory-mapped files (`mmap`)
3. **Compare** performance and memory usage vs standard file I/O
4. **Handle** file sizes larger than available RAM

## Solution Template

```python
import mmap
import struct
import os

def generate_test_file(filename, num_integers=1000000):
    """Generate a binary file with integers"""
    with open(filename, 'wb') as f:
        for i in range(num_integers):
            # Write 4-byte integers
            f.write(struct.pack('i', i % 1000000))

def process_with_mmap(filename):
    """Process file using memory mapping"""
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # Your implementation using mmap
    with open(filename, 'rb') as f:
        # Create memory map
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Process in chunks to handle large files
            # mm can be accessed like bytes
            pass
    
    return stats

def process_standard_io(filename):
    """Process file using standard I/O (for comparison)"""
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # Standard implementation
    pass
    
    return stats

if __name__ == "__main__":
    test_file = 'large_data.bin'
    
    # Generate test file (uncomment when ready)
    # generate_test_file(test_file, 10000000)
    
    # Compare approaches
    # mmap_stats = process_with_mmap(test_file)
    # std_stats = process_standard_io(test_file)
    
    # print("Memory-mapped:", mmap_stats)
    # print("Standard I/O:", std_stats)
```

## Key Learning Points

1. **Memory Mapping:** Allows accessing file data without loading into RAM
2. **Chunk Processing:** Handle files in chunks for very large files
3. **Binary Data:** Using `struct` for binary I/O

## Advanced Considerations

- Process file in chunks (e.g., 1MB at a time)
- Handle endianness if cross-platform
- Consider using `numpy.memmap` for numeric data

