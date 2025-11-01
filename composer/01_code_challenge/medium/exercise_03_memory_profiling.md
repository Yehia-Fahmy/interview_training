# Exercise 3: Memory Profiling and Optimization

**Difficulty:** Medium  
**Time Limit:** 40 minutes  
**Focus:** Memory profiling tools, identifying memory bottlenecks

## Problem

You have a function that processes a large CSV-like dataset and you suspect it has memory issues. Your task is to:

1. Profile the memory usage
2. Identify memory bottlenecks
3. Optimize the code

**Given Function:**
```python
def process_large_dataset(file_path):
    """Process a large CSV file"""
    # Read entire file into memory
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    processed = []
    for line in lines:
        parts = line.strip().split(',')
        # Create dictionary for each row
        row_dict = {}
        for i, part in enumerate(parts):
            row_dict[f'col_{i}'] = part
        processed.append(row_dict)
    
    # Filter rows
    filtered = []
    for row in processed:
        if int(row['col_0']) > 100:
            filtered.append(row)
    
    # Transform data
    result = []
    for row in filtered:
        transformed = {
            'id': row['col_0'],
            'value': row['col_1'],
            'category': row['col_2']
        }
        result.append(transformed)
    
    return result
```

## Tasks

1. **Profile** the memory usage using `memory_profiler` or similar tools
2. **Identify** the memory bottlenecks
3. **Optimize** the function to use less memory
4. **Measure** the improvement

## Solution Template

```python
# Install: pip install memory-profiler
# Usage: python -m memory_profiler your_script.py

from memory_profiler import profile
import sys

@profile
def process_large_dataset_optimized(file_path):
    """Your optimized version"""
    # Use generators, process line by line
    pass

def compare_memory_usage():
    # Create test file
    test_file = 'test_data.csv'
    with open(test_file, 'w') as f:
        for i in range(100000):
            f.write(f"{i},{i*2},{i%10}\n")
    
    # Profile original (if you implement it)
    # Profile optimized
    result = process_large_dataset_optimized(test_file)
    
    print(f"Processed {len(result)} rows")

if __name__ == "__main__":
    compare_memory_usage()
```

## Key Learning Points

1. **Memory Profiling Tools:** `memory_profiler`, `tracemalloc`, `pympler`
2. **Generator Patterns:** Processing data incrementally
3. **Memory-Efficient Parsing:** Reading files line-by-line

## Optimization Strategies

- Use generators instead of lists
- Process data incrementally
- Avoid creating intermediate data structures
- Use appropriate data types (e.g., `array.array` for numeric data)

