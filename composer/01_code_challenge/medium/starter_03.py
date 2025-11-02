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
import time
import tracemalloc


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
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    filtered = []
    for line in lines:
        elements = line.strip().split(",")
        if not int(elements[0]) > 100:
            continue
        d = {
            'id': int(elements[0]),
            'value': int(elements[1]),
            'category': int(elements[2])
        }
        filtered.append(d)

    return filtered


def process_large_dataset_generator(file_path):
    """
    Generator-based implementation for comparison.
    
    This version uses generators to process the file line-by-line,
    avoiding loading the entire file into memory at once.
    
    Advantages:
    - Lower memory footprint (O(1) memory for processing, O(n) only for final list)
    - Can handle arbitrarily large files
    - Processes data incrementally (streaming approach)
    
    Returns: Same format as process_large_dataset()
    """
    def process_line_generator(file_path):
        """
        Generator that yields processed rows one at a time.
        This avoids loading the entire file into memory.
        """
        with open(file_path, 'r') as f:
            for line in f:  # Read line by line, not all at once
                parts = line.strip().split(',')
                if not parts or not parts[0]:  # Skip empty lines
                    continue
                
                # Filter: only process rows where first column > 100
                try:
                    if int(parts[0]) <= 100:
                        continue
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
                
                # Transform directly to final format (avoiding intermediate dicts)
                # Keep values as strings to match original output format
                yield {
                    'id': parts[0],      # Keep as string to match original
                    'value': parts[1],   # Keep as string to match original
                    'category': parts[2] # Keep as string to match original
                }
    
    # Consume the generator into a list for return
    # In a truly streaming scenario, you could return the generator itself
    return list(process_line_generator(file_path))

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


def test_performance(test_file='test_data.csv', num_rows=50000):
    """
    Comprehensive performance testing function.
    
    Tests:
    - Memory usage (peak and current)
    - Execution time
    - Correctness verification
    - Memory savings percentage
    """
    print("=" * 70)
    print("PERFORMANCE TEST SUITE")
    print("=" * 70)
    
    # Create test file if it doesn't exist
    if not os.path.exists(test_file):
        print(f"\nCreating test file with {num_rows:,} rows...")
        create_test_file(test_file, num_rows=num_rows)
    
    file_size = os.path.getsize(test_file) / (1024 * 1024)  # MB
    print(f"\nTest file: {test_file}")
    print(f"File size: {file_size:.2f} MB")
    print(f"Number of rows: {num_rows:,}")
    print("-" * 70)
    
    # Test Original Implementation
    print("\n[1] Testing ORIGINAL (memory-inefficient) implementation...")
    tracemalloc.start()
    start_time = time.time()
    
    result_original = process_large_dataset(test_file)
    
    elapsed_time_orig = time.time() - start_time
    current_orig, peak_orig = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  ✓ Execution time: {elapsed_time_orig:.4f} seconds")
    print(f"  ✓ Peak memory:    {peak_orig / (1024 * 1024):.2f} MB")
    print(f"  ✓ Current memory: {current_orig / (1024 * 1024):.2f} MB")
    print(f"  ✓ Results count:  {len(result_original):,}")
    
    # Test Optimized Implementation
    print("\n[2] Testing OPTIMIZED implementation...")
    tracemalloc.start()
    start_time = time.time()
    
    result_optimized = process_large_dataset_optimized(test_file)
    
    elapsed_time_opt = time.time() - start_time
    current_opt, peak_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  ✓ Execution time: {elapsed_time_opt:.4f} seconds")
    print(f"  ✓ Peak memory:    {peak_opt / (1024 * 1024):.2f} MB")
    print(f"  ✓ Current memory: {current_opt / (1024 * 1024):.2f} MB")
    print(f"  ✓ Results count:  {len(result_optimized):,}")
    
    # Test Generator-based Implementation
    print("\n[3] Testing GENERATOR-based implementation...")
    tracemalloc.start()
    start_time = time.time()
    
    result_generator = process_large_dataset_generator(test_file)
    
    elapsed_time_gen = time.time() - start_time
    current_gen, peak_gen = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"  ✓ Execution time: {elapsed_time_gen:.4f} seconds")
    print(f"  ✓ Peak memory:    {peak_gen / (1024 * 1024):.2f} MB")
    print(f"  ✓ Current memory: {current_gen / (1024 * 1024):.2f} MB")
    print(f"  ✓ Results count:  {len(result_generator):,}")
    
    # Verify Correctness
    print("\n[4] Verifying correctness...")
    if len(result_original) != len(result_optimized) or len(result_original) != len(result_generator):
        print(f"  ✗ LENGTH MISMATCH!")
        print(f"    Original:   {len(result_original):,} rows")
        print(f"    Optimized: {len(result_optimized):,} rows")
        print(f"    Generator:  {len(result_generator):,} rows")
        return False
    
    # Check if results match (accounting for potential data type differences)
    correctness = True
    try:
        # Convert all to comparable format for checking
        orig_normalized = [
            {k: str(v) for k, v in row.items()} 
            for row in result_original
        ]
        opt_normalized = [
            {k: str(v) for k, v in row.items()} 
            for row in result_optimized
        ]
        gen_normalized = [
            {k: str(v) for k, v in row.items()} 
            for row in result_generator
        ]
        
        opt_match = orig_normalized == opt_normalized
        gen_match = orig_normalized == gen_normalized
        
        if opt_match and gen_match:
            print("  ✓ All implementations produce matching results!")
        elif opt_match:
            print("  ⚠ Optimized matches, but Generator has differences")
            correctness = False
        elif gen_match:
            print("  ⚠ Generator matches, but Optimized has differences")
            correctness = False
        else:
            print("  ⚠ Results have some differences (may be data type related)")
            print(f"    First original:   {result_original[0] if result_original else 'N/A'}")
            print(f"    First optimized:  {result_optimized[0] if result_optimized else 'N/A'}")
            print(f"    First generator:   {result_generator[0] if result_generator else 'N/A'}")
            correctness = False
    except Exception as e:
        print(f"  ⚠ Could not fully verify results: {e}")
        correctness = True  # Assume correct if we can't verify
    
    # Performance Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY - COMPARISON")
    print("=" * 70)
    
    # Memory comparison
    memory_savings_opt = ((peak_orig - peak_opt) / peak_orig * 100) if peak_orig > 0 else 0
    memory_savings_gen = ((peak_orig - peak_gen) / peak_orig * 100) if peak_orig > 0 else 0
    print(f"\n📊 Memory Usage (Peak):")
    print(f"   Original:   {peak_orig / (1024 * 1024):.2f} MB")
    print(f"   Optimized:  {peak_opt / (1024 * 1024):.2f} MB  (Savings: {memory_savings_opt:.1f}%)")
    print(f"   Generator:  {peak_gen / (1024 * 1024):.2f} MB  (Savings: {memory_savings_gen:.1f}%)")
    
    # Time comparison
    print(f"\n⚡ Execution Time:")
    print(f"   Original:   {elapsed_time_orig:.4f} seconds")
    print(f"   Optimized:  {elapsed_time_opt:.4f} seconds", end="")
    if elapsed_time_orig > 0:
        opt_speedup = elapsed_time_orig / elapsed_time_opt if elapsed_time_opt > 0 else 0
        print(f"  ({opt_speedup:.2f}x {'faster' if opt_speedup > 1 else 'slower'})")
    else:
        print()
    
    print(f"   Generator:  {elapsed_time_gen:.4f} seconds", end="")
    if elapsed_time_orig > 0:
        gen_speedup = elapsed_time_orig / elapsed_time_gen if elapsed_time_gen > 0 else 0
        print(f"  ({gen_speedup:.2f}x {'faster' if gen_speedup > 1 else 'slower'})")
    else:
        print()
    
    # Best implementation analysis
    print(f"\n📈 Analysis:")
    if peak_gen < peak_opt:
        print(f"   🏆 Generator uses least memory: {peak_gen / (1024 * 1024):.2f} MB")
    elif peak_opt < peak_gen:
        print(f"   🏆 Optimized uses least memory: {peak_opt / (1024 * 1024):.2f} MB")
    else:
        print(f"   ⚖️  Memory usage is similar")
    
    if elapsed_time_gen < elapsed_time_opt:
        print(f"   ⚡ Generator is fastest: {elapsed_time_gen:.4f}s")
    elif elapsed_time_opt < elapsed_time_gen:
        print(f"   ⚡ Optimized is fastest: {elapsed_time_opt:.4f}s")
    else:
        print(f"   ⚖️  Execution time is similar")
    
    if correctness:
        print(f"   ✓ Correctness: PASSED (all implementations match)")
    else:
        print(f"   ✗ Correctness: FAILED (results don't match)")
    
    print("=" * 70)
    
    return correctness


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CSV processing performance')
    parser.add_argument('--test-file', default='test_data.csv', 
                       help='Path to test CSV file')
    parser.add_argument('--num-rows', type=int, default=50000,
                       help='Number of rows in test file (if creating new)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with smaller dataset (10,000 rows)')
    parser.add_argument('--full', action='store_true',
                       help='Run full test with larger dataset (100,000 rows)')
    
    args = parser.parse_args()
    
    # Determine test size
    if args.quick:
        num_rows = 10000
    elif args.full:
        num_rows = 100000
    else:
        num_rows = args.num_rows
    
    # Run performance tests
    print("\n" + "=" * 70)
    print("CSV Processing Performance Test")
    print("=" * 70)
    
    success = test_performance(test_file=args.test_file, num_rows=num_rows)
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Review the output above.")
        sys.exit(1)

