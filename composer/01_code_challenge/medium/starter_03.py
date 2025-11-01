"""
Exercise 3: Memory Profiling and Optimization

Profile a memory-inefficient CSV processing function and optimize it
to handle large datasets without running out of memory.

Task:
1. Profile memory usage of the given function
2. Identify memory bottlenecks
3. Optimize to use less memory
4. Measure the improvement
"""

import os
import sys


def process_large_dataset(file_path):
    """
    Original memory-inefficient implementation.
    DO NOT MODIFY - used for comparison in tests.
    """
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


def process_large_dataset_optimized(file_path):
    """
    Your optimized implementation.
    
    Requirements:
    - Process data incrementally (line by line)
    - Use generators where possible
    - Avoid creating large intermediate data structures
    - Produce the same output format as the original
    """
    pass


def create_test_file(filename='test_data.csv', num_rows=100000):
    """Helper function to create test CSV file"""
    with open(filename, 'w') as f:
        for i in range(num_rows):
            f.write(f"{i},{i*2},{i%10}\n")
    print(f"Created test file: {filename} with {num_rows} rows")


def compare_memory_usage():
    """Compare memory usage between implementations"""
    test_file = 'test_data.csv'
    
    if not os.path.exists(test_file):
        create_test_file(test_file, num_rows=50000)
    
    print("Processing with original implementation...")
    # Note: For actual profiling, use: python -m memory_profiler starter_03.py
    result_original = process_large_dataset(test_file)
    print(f"Original: Processed {len(result_original)} rows")
    
    print("\nProcessing with optimized implementation...")
    result_optimized = process_large_dataset_optimized(test_file)
    print(f"Optimized: Processed {len(result_optimized)} rows")
    
    # Verify results match
    assert len(result_original) == len(result_optimized)
    assert result_original == result_optimized, "Results don't match!"
    print("\n✓ Results match!")


if __name__ == "__main__":
    # Create test file if it doesn't exist
    if not os.path.exists('test_data.csv'):
        create_test_file('test_data.csv', num_rows=1000)
    
    # Test optimized version
    result = process_large_dataset_optimized('test_data.csv')
    print(f"Processed {len(result)} rows")
    if result:
        print(f"First result: {result[0]}")

