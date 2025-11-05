"""
Exercise 1: Memory-Mapped File Processing

Implement efficient processing of large binary files using memory-mapped files.
Handle files larger than available RAM without loading entire file into memory.

Requirements:
- Read and process a large binary file (e.g., 10GB+) containing integers
- Find the maximum value, minimum value, and sum
- Use memory mapping to avoid loading entire file into memory
- Handle edge cases (empty file, negative numbers, etc.)

Extension: Multiprocessing
- Implement parallel processing using multiple CPU cores
- Split file into chunks and process them concurrently
- Aggregate results from all worker processes
- Compare performance with single-threaded approaches
"""

import mmap
import struct
import os
import random
import time
from functools import partial
from multiprocessing import Pool, cpu_count


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

            # Process full chunks
            for i in range(total_ints // CHUNK_SIZE):
                s = i * CHUNK_SIZE * INT_SIZE
                end = min((i + 1) * CHUNK_SIZE * INT_SIZE, file_size)
                chunk = mm[s:end]

                ints_in_chunk = len(chunk) // INT_SIZE
                if ints_in_chunk > 0:
                    a = struct.unpack(f"{ints_in_chunk}i", chunk)

                    for n in a:
                        stats['min'] = min(n, stats['min'])
                        stats['max'] = max(n, stats['max'])
                        stats['sum'] += n
                    stats['count'] += ints_in_chunk
            
            # Process remainder
            remainder_start = (total_ints // CHUNK_SIZE) * CHUNK_SIZE * INT_SIZE
            if remainder_start < file_size:
                chunk = mm[remainder_start:file_size]
                ints_in_chunk = len(chunk) // INT_SIZE
                if ints_in_chunk > 0:
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


# ============================================================================
# MULTIPROCESSING EXTENSION (Guided Exercise)
# ============================================================================
# Implement parallel processing of large files using multiprocessing.
# 
# Key Concepts:
# - Split file into chunks that can be processed independently
# - Use multiprocessing.Pool to distribute work across CPU cores
# - Ensure integer boundary alignment to avoid reading partial integers
# - Aggregate results from all worker processes
#
# Implementation Steps:
# 1. Complete _process_chunk_worker() - processes a single chunk
# 2. Complete _split_file_into_chunks() - divides file into chunks
# 3. Complete process_with_multiprocessing() - orchestrates parallel processing
# ============================================================================


def _process_chunk_worker(args):
    """
    Worker function for multiprocessing. Processes a chunk of the file.
    
    This function is called by each worker process to process a specific byte range
    of the file. It must handle integer boundary alignment to avoid reading partial integers.
    
    Args:
        args: Tuple of (filename, start_offset, end_offset)
        
    Returns:
        dict with keys: 'min', 'max', 'sum', 'count'
    """
    filename, start_offset, end_offset = args
    INT_SIZE = 4
    
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # TODO: Align offsets to integer boundaries
    # Hint: Ensure start_offset and end_offset are multiples of INT_SIZE (4 bytes)
    # If start_offset is not aligned, round it forward to the next integer boundary
    # Round end_offset backward to the nearest integer boundary
    while not start_offset % INT_SIZE == 0:
        start_offset += 1
    
    while not end_offset % INT_SIZE == 0:
        end_offset -= 1
    
    # TODO: Check if aligned offsets are valid
    # If start_offset >= end_offset after alignment, return empty stats
    if start_offset >= end_offset: return stats
    
    # TODO: Open file and create memory map for this chunk
    # Hint: Use 'with open(filename, 'rb') as f:' and 'with mmap.mmap(...) as mm:'
    # Use mmap.ACCESS_READ for read-only access
    # Extract the chunk using slicing: mm[start_offset:end_offset]
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            chunk = mm[start_offset:end_offset]
    if len(chunk) == 0: return stats
    # TODO: Process the chunk
    # Ensure chunk length is a multiple of INT_SIZE (truncate if needed)
    # Use struct.unpack() to convert bytes to integers
    # Update stats for each integer: min, max, sum, count
    ints_in_chunk = len(chunk) // INT_SIZE
    aligned_size = ints_in_chunk * INT_SIZE
    chunk = chunk[:aligned_size]
    ints = struct.unpack(f"{ints_in_chunk}i", chunk)

    for i in ints:
        stats['max'] = max(stats['max'], i)
        stats['min'] = min(stats['min'], i)
        stats['sum'] += i
    stats['count'] += ints_in_chunk
    
    return stats


def _split_file_into_chunks(filename, num_processes):
    """
    Split file into chunks for parallel processing.
    Ensures chunks are aligned to integer boundaries.
    
    This function divides the file into approximately equal-sized chunks that can be
    processed in parallel. Each chunk must be aligned to 4-byte boundaries to ensure
    we never read partial integers.
    
    Args:
        filename: Path to binary file
        num_processes: Number of processes to use
        
    Returns:
        List of tuples (filename, start_offset, end_offset)
    """
    INT_SIZE = 4
    file_size = os.path.getsize(filename)
    
    if file_size == 0:
        return []
    
    # TODO: Calculate chunk size per process
    # Hint: Divide file_size by num_processes using ceiling division
    # Formula: (file_size + num_processes - 1) // num_processes
    chunk_size = (file_size + num_processes - 1) // num_processes
    
    # TODO: Align chunk size to integer boundaries
    # Hint: Round chunk_size up to the nearest multiple of INT_SIZE
    # Formula: ((chunk_size + INT_SIZE - 1) // INT_SIZE) * INT_SIZE
    aligned_chunk_size = ((chunk_size + INT_SIZE - 1) // INT_SIZE) * INT_SIZE
    # TODO: Create chunks list
    # Initialize an empty list to store chunk tuples
    chunks = []
    
    # TODO: Iterate through file and create chunks
    # Start from offset 0
    # While start < file_size:
    #   - Calculate end = min(start + chunk_size, file_size)
    #   - Align end to integer boundary: (end // INT_SIZE) * INT_SIZE
    #   - Append tuple (filename, start, end) to chunks
    #   - Update start = end
    start = 0
    while start < file_size:
        end = min(start + aligned_chunk_size, file_size)
        end = (end // INT_SIZE) * INT_SIZE
        chunks.append((filename, start, end))
        start = end
    
    return chunks


def process_with_multiprocessing(filename, num_processes=None):
    """
    Process file using multiprocessing with memory mapping.
    
    This function splits the file into chunks and processes them in parallel
    across multiple CPU cores. Each worker process uses memory mapping for
    efficient access to its assigned chunk.
    
    Steps:
    1. Determine number of processes (use cpu_count() if None)
    2. Handle edge case: empty file
    3. Split file into chunks using _split_file_into_chunks()
    4. Use multiprocessing.Pool to process chunks in parallel
    5. Aggregate results from all workers
    
    Args:
        filename: Path to binary file
        num_processes: Number of processes to use (default: cpu_count())
        
    Returns:
        dict with keys: 'min', 'max', 'sum', 'count'
    """
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    # TODO: Set default number of processes
    # If num_processes is None, use cpu_count() to use all available CPU cores
    if not num_processes:
        num_processes = os.cpu_count()
    
    
    # TODO: Handle empty file edge case
    # Get file size using os.path.getsize()
    # If file_size == 0, return empty stats dict
    file_size = os.path.getsize(filename)
    if file_size == 0: return stats
    
    # TODO: Split file into chunks
    # Call _split_file_into_chunks(filename, num_processes) to get list of chunks
    # Each chunk is a tuple: (filename, start_offset, end_offset)
    chunks = _split_file_into_chunks(filename, num_processes)
    
    # TODO: Check if chunks list is empty
    # If chunks list is empty, return empty stats dict
    if len(chunks) == 0: return stats
    
    # TODO: Process chunks in parallel using multiprocessing.Pool
    # Hint: Use 'with Pool(processes=num_processes) as pool:'
    # Use pool.map(_process_chunk_worker, chunks) to process all chunks
    # This returns a list of result dictionaries, one per chunk
    with Pool(num_processes) as pool:
        results_list = pool.map(_process_chunk_worker, chunks)
    
    # TODO: Aggregate results from all workers
    # Initialize stats dict with default values (inf for min, -inf for max, 0 for sum/count)
    # Iterate through results list
    # For each result that has count > 0:
    #   - Update stats['min'] = min(stats['min'], result['min'])
    #   - Update stats['max'] = max(stats['max'], result['max'])
    #   - Add result['sum'] to stats['sum']
    #   - Add result['count'] to stats['count']
    for res in results_list:
        if res['count'] > 0:
            stats['count'] += res['count']
            stats['sum'] += res['sum']
            stats['min'] = min(stats['min'], res['min'])
            stats['max'] = max(stats['max'], res['max'])
    
    # Placeholder return (replace with actual implementation)
    return stats


def benchmark_approach(func, filename, name, num_runs=3, warmup_runs=1):
    """
    Benchmark a processing function with multiple runs.
    
    Args:
        func: Function to benchmark
        filename: Path to test file
        name: Name of the approach
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        Tuple of (average_time, results_dict)
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = func(filename)
    
    # Benchmark runs
    times = []
    results = None
    for _ in range(num_runs):
        start = time.time()
        result = func(filename)
        elapsed = time.time() - start
        times.append(elapsed)
        if results is None:
            results = result
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        'name': name,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'times': times,
        'results': results
    }


def compare_performance(filename, num_runs=3):
    """
    Compare performance of different processing approaches.
    
    Args:
        filename: Path to test file
        num_runs: Number of benchmark runs per approach
    """
    if not os.path.exists(filename):
        print(f"Test file {filename} not found!")
        return
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    file_size_bytes = os.path.getsize(filename)
    
    print("=" * 80)
    print("PERFORMANCE BENCHMARK: File Processing Comparison")
    print("=" * 80)
    print(f"File: {filename}")
    print(f"Size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    print(f"CPU Cores: {cpu_count()}")
    print(f"Benchmark runs per approach: {num_runs}")
    print("=" * 80)
    
    # Benchmark all approaches
    benchmarks = []
    
    print("\n[1/3] Benchmarking: Memory-mapped (single-threaded)...")
    b1 = benchmark_approach(process_with_mmap, filename, 
                           "Memory-mapped (single-threaded)", num_runs)
    benchmarks.append(b1)
    
    print("[2/3] Benchmarking: Standard I/O (single-threaded)...")
    b2 = benchmark_approach(process_standard_io, filename,
                           "Standard I/O (single-threaded)", num_runs)
    benchmarks.append(b2)
    
    print("[3/3] Benchmarking: Multiprocessing...")
    num_procs = cpu_count()
    mp_func = partial(process_with_multiprocessing, num_processes=num_procs)
    b3 = benchmark_approach(mp_func, filename,
                           f"Multiprocessing ({num_procs} processes)", num_runs)
    benchmarks.append(b3)
    
    # Also test with different numbers of processes
    for num_procs in [2, 4, 8]:
        if num_procs <= cpu_count():
            print(f"[Bonus] Benchmarking: Multiprocessing ({num_procs} processes)...")
            mp_func = partial(process_with_multiprocessing, num_processes=num_procs)
            b = benchmark_approach(mp_func, filename,
                                  f"Multiprocessing ({num_procs} processes)", num_runs)
            benchmarks.append(b)
    
    # Display results table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Approach':<40} {'Avg Time (s)':<15} {'Throughput (MB/s)':<20} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_time = benchmarks[0]['avg_time']  # Memory-mapped as baseline
    
    for b in benchmarks:
        throughput = file_size_mb / b['avg_time'] if b['avg_time'] > 0 else 0
        speedup = baseline_time / b['avg_time'] if b['avg_time'] > 0 else 0
        print(f"{b['name']:<40} {b['avg_time']:<15.3f} {throughput:<20.2f} {speedup:<10.2f}x")
    
    # Detailed timing information
    print("\n" + "=" * 80)
    print("DETAILED TIMING INFORMATION")
    print("=" * 80)
    for b in benchmarks:
        print(f"\n{b['name']}:")
        print(f"  Average: {b['avg_time']:.3f}s")
        print(f"  Minimum: {b['min_time']:.3f}s")
        print(f"  Maximum: {b['max_time']:.3f}s")
        print(f"  Individual runs: {[f'{t:.3f}' for t in b['times']]}")
        print(f"  Throughput: {file_size_mb / b['avg_time']:.2f} MB/s")
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    # Use first benchmark result as reference
    ref_results = benchmarks[0]['results']
    all_match = True
    
    for b in benchmarks:
        match = (
            ref_results['min'] == b['results']['min'] and
            ref_results['max'] == b['results']['max'] and
            ref_results['sum'] == b['results']['sum'] and
            ref_results['count'] == b['results']['count']
        )
        status = "✓ MATCH" if match else "✗ MISMATCH"
        print(f"{b['name']:<40} {status}")
        if not match:
            all_match = False
            print(f"  Reference: {ref_results}")
            print(f"  Actual:    {b['results']}")
    
    if all_match:
        print("\n✓ All approaches produce identical results!")
    
    # Performance recommendations
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    fastest = min(benchmarks, key=lambda x: x['avg_time'])
    print(f"Fastest approach: {fastest['name']} ({fastest['avg_time']:.3f}s)")
    
    if len(benchmarks) > 1:
        improvements = []
        for b in benchmarks:
            if b['name'] != fastest['name']:
                improvement = ((b['avg_time'] - fastest['avg_time']) / b['avg_time']) * 100
                improvements.append(f"{fastest['name']} is {improvement:.1f}% faster than {b['name']}")
        
        if improvements:
            print("\nPerformance improvements:")
            for imp in improvements[:3]:  # Show top 3
                print(f"  • {imp}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_file = 'large_data.bin'
    
    # Generate test file
    print("Generating test file...")
    num_integers = 10000000  # 10 million integers ≈ 38 MB
    generate_test_file(test_file, num_integers)
    
    # Run performance comparison
    compare_performance(test_file, num_runs=3)

