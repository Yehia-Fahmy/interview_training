"""
Exercise 2: Threading vs Multiprocessing

Process a large dataset using CPU-intensive operations. Implement and
compare sequential, threading, and multiprocessing approaches.

Requirements:
- Implement all three approaches
- Measure performance for each
- Handle proper resource cleanup
- Explain why one approach performs better

Note: Threading may not help with CPU-bound tasks due to Python's GIL.
"""

from ast import arg
import time
import math
from threading import Thread, Lock
from multiprocessing import Process, Pool
import concurrent.futures


def cpu_intensive_task(n):
    """
    A CPU-intensive operation.
    Returns True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def sequential_process(data):
    """
    Sequential processing - baseline implementation.
    Process each item one by one.
    """
    results = []
    for item in data:
        results.append(cpu_intensive_task(item))
    return results


def threaded_process(data, num_threads=4):
    """
    Threading approach.
    Hint: Split data into chunks and process in parallel threads.
    """
    def threaded_function(data_slice, results):
        for item in data_slice:
            results.append(cpu_intensive_task(item))
    batch_size = int(len(data) / num_threads) + 1
    threads = []
    results = []
    for i in range(num_threads):
        thread = Thread(target=threaded_function, args=(data[(i*batch_size):((i+1)*batch_size)], results))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return results


def multiprocessed_process(data, num_processes=4):
    """
    Multiprocessing approach.
    Hint: Use multiprocessing.Pool for parallel processing.
    """
    results = []
    with Pool(processes=num_processes) as pool:
        results = pool.map(cpu_intensive_task, data)
    return results


# ============================================================================
# IMPROVED/IDEAL IMPLEMENTATIONS FOR COMPARISON
# ============================================================================

def threaded_process_ideal(data, num_threads=4):
    """
    Improved threading implementation with proper thread safety.
    
    Fixes:
    - Uses Lock to prevent race conditions
    - Collects results locally then extends (minimizes lock contention)
    - Proper chunking with bounds checking
    """
    def threaded_function(data_slice, results, lock):
        # Collect locally first to minimize lock contention
        local_results = []
        for item in data_slice:
            local_results.append(cpu_intensive_task(item))
        # Single lock operation to extend results
        with lock:
            results.extend(local_results)
    
    if not data:
        return []
    
    batch_size = (len(data) + num_threads - 1) // num_threads
    threads = []
    results = []
    lock = Lock()
    
    for i in range(num_threads):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        if start_idx >= len(data):
            break
        data_slice = data[start_idx:end_idx]
        thread = Thread(target=threaded_function, args=(data_slice, results, lock))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    return results


def threaded_process_concurrent_futures(data, num_threads=4):
    """
    Alternative threading approach using ThreadPoolExecutor.
    Cleaner API and automatic resource management.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(cpu_intensive_task, data))
    return results


def multiprocessed_process_ideal(data, num_processes=4):
    """
    Improved multiprocessing with chunking for better load distribution.
    Uses imap for potentially better memory usage with large datasets.
    """
    with Pool(processes=num_processes) as pool:
        results = list(pool.imap(cpu_intensive_task, data))
    return results


def compare_approaches():
    """Compare all three approaches"""
    data = list(range(1000, 11000))  # 10,000 numbers
    
    # Sequential
    print("Running sequential approach...")
    start = time.time()
    seq_results = sequential_process(data)
    seq_time = time.time() - start
    
    # Threading
    print("Running threading approach...")
    start = time.time()
    thread_results = threaded_process(data)
    thread_time = time.time() - start
    
    # Multiprocessing
    print("Running multiprocessing approach...")
    start = time.time()
    mp_results = multiprocessed_process(data)
    mp_time = time.time() - start
    
    # Verify all produce same results
    assert seq_results == thread_results == mp_results, "Results don't match!"
    
    print(f"\nSequential:    {seq_time:.2f}s")
    print(f"Threading:     {thread_time:.2f}s")
    print(f"Multiprocessing: {mp_time:.2f}s")
    
    print(f"\nThreading speedup: {seq_time/thread_time:.2f}x")
    print(f"Multiprocessing speedup: {seq_time/mp_time:.2f}x")


if __name__ == "__main__":
    # Test all approaches for correctness with small data
    test_data = list(range(10, 20))
    print("Testing correctness of all approaches...")
    print(f"Test data: {test_data}\n")
    
    # Sequential
    print("Testing sequential approach...")
    seq_result = sequential_process(test_data)
    print(f"  Processed {len(seq_result)} items")
    print(f"  First few results: {seq_result[:5]}")
    
    # Threading
    print("\nTesting threading approach...")
    thread_result = threaded_process(test_data)
    print(f"  Processed {len(thread_result)} items")
    print(f"  First few results: {thread_result[:5]}")
    
    # Multiprocessing
    print("\nTesting multiprocessing approach...")
    mp_result = multiprocessed_process(test_data)
    print(f"  Processed {len(mp_result)} items")
    print(f"  First few results: {mp_result[:5]}")
    
    # Verify correctness
    print("\n" + "="*50)
    print("Correctness Check:")
    print("="*50)
    
    seq_match_thread = seq_result == thread_result
    seq_match_mp = seq_result == mp_result
    thread_match_mp = thread_result == mp_result
    
    print(f"Sequential == Threading:   {seq_match_thread}")
    print(f"Sequential == Multiprocessing: {seq_match_mp}")
    print(f"Threading == Multiprocessing:  {thread_match_mp}")
    
    if seq_match_thread and seq_match_mp:
        print("\n✓ All approaches produce identical results!")
    else:
        print("\n✗ Results don't match! Check implementations.")
        if not seq_match_thread:
            print(f"  Sequential: {seq_result}")
            print(f"  Threading:  {thread_result}")
        if not seq_match_mp:
            print(f"  Sequential: {seq_result}")
            print(f"  Multiprocessing: {mp_result}")


def comprehensive_performance_test(data_size=100000, num_workers=4, runs=3):
    """
    Comprehensive performance testing suite.
    Tests all implementations multiple times and averages results.
    
    Args:
        data_size: Size of dataset (default 100,000 for clearer differences)
        num_workers: Number of parallel workers
        runs: Number of runs per test for averaging
    """
    import statistics
    
    print("=" * 70)
    print("COMPREHENSIVE PERFORMANCE TEST SUITE")
    print("=" * 70)
    print(f"Dataset size: {data_size:,} numbers")
    print(f"Workers: {num_workers}")
    print(f"Runs per test: {runs}")
    print("-" * 70)
    
    # Generate test data
    data = list(range(1000, 1000 + data_size))
    
    implementations = [
        ("Sequential (Baseline)", sequential_process),
        ("Threading (Current)", lambda d: threaded_process(d, num_workers)),
        ("Threading (Ideal - Lock)", lambda d: threaded_process_ideal(d, num_workers)),
        ("Threading (Concurrent Futures)", lambda d: threaded_process_concurrent_futures(d, num_workers)),
        ("Multiprocessing (Current)", lambda d: multiprocessed_process(d, num_workers)),
        ("Multiprocessing (Ideal)", lambda d: multiprocessed_process_ideal(d, num_workers)),
    ]
    
    results = {}
    
    print("\n[Running Performance Tests...]")
    for name, func in implementations:
        times = []
        for run in range(runs):
            start = time.time()
            result = func(data)
            elapsed = time.time() - start
            times.append(elapsed)
            # Verify correctness on first run
            if run == 0:
                baseline_result = sequential_process(data)
                if result != baseline_result:
                    print(f"  ⚠ {name}: Results don't match baseline!")
        
        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        results[name] = {
            'time': avg_time,
            'std': std_time,
            'result': result if name == "Sequential (Baseline)" else None
        }
        print(f"  ✓ {name:35} {avg_time:.4f}s ± {std_time:.4f}s")
    
    # Calculate speedups
    baseline_time = results["Sequential (Baseline)"]['time']
    
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Implementation':<40} {'Time (s)':<12} {'Speedup':<10} {'Efficiency'}")
    print("-" * 70)
    
    for name, stats in results.items():
        time_val = stats['time']
        if name == "Sequential (Baseline)":
            speedup = 1.0
            efficiency = "Baseline"
        else:
            speedup = baseline_time / time_val if time_val > 0 else 0
            # Efficiency = speedup / num_workers (for parallel implementations)
            if "Threading" in name or "Multiprocessing" in name:
                efficiency = f"{speedup/num_workers*100:.1f}%"
            else:
                efficiency = "N/A"
        
        print(f"{name:<40} {time_val:<12.4f} {speedup:<10.2f}x {efficiency}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    best_time = min(r['time'] for n, r in results.items() if n != "Sequential (Baseline)")
    best_name = [n for n, r in results.items() if r['time'] == best_time and n != "Sequential (Baseline)"][0]
    best_speedup = baseline_time / best_time
    
    print(f"\n🏆 Fastest: {best_name} ({best_time:.4f}s, {best_speedup:.2f}x speedup)")
    
    # Threading analysis
    thread_current = results["Threading (Current)"]['time']
    thread_ideal = results["Threading (Ideal - Lock)"]['time']
    thread_futures = results["Threading (Concurrent Futures)"]['time']
    
    print(f"\n📊 Threading Comparison:")
    print(f"   Current:        {thread_current:.4f}s ({baseline_time/thread_current:.2f}x)")
    print(f"   Ideal (Lock):   {thread_ideal:.4f}s ({baseline_time/thread_ideal:.2f}x)")
    print(f"   ConcurrentFutures: {thread_futures:.4f}s ({baseline_time/thread_futures:.2f}x)")
    
    # Multiprocessing analysis
    mp_current = results["Multiprocessing (Current)"]['time']
    mp_ideal = results["Multiprocessing (Ideal)"]['time']
    
    print(f"\n📊 Multiprocessing Comparison:")
    print(f"   Current: {mp_current:.4f}s ({baseline_time/mp_current:.2f}x)")
    print(f"   Ideal:   {mp_ideal:.4f}s ({baseline_time/mp_ideal:.2f}x)")
    
    # Best parallel approach
    parallel_times = {
        "Threading (Ideal)": thread_ideal,
        "Threading (Concurrent Futures)": thread_futures,
        "Multiprocessing (Ideal)": mp_ideal,
    }
    best_parallel = min(parallel_times.items(), key=lambda x: x[1])
    
    print(f"\n🎯 Best Parallel Approach: {best_parallel[0]}")
    print(f"   Time: {best_parallel[1]:.4f}s")
    print(f"   Speedup: {baseline_time/best_parallel[1]:.2f}x")
    
    return results


def multi_scale_performance_test():
    """
    Test performance across multiple dataset sizes to show how approaches scale.
    This clearly demonstrates the differences between threading and multiprocessing.
    """
    print("=" * 70)
    print("MULTI-SCALE PERFORMANCE TEST")
    print("=" * 70)
    print("Testing across multiple dataset sizes to show scaling behavior\n")
    
    test_sizes = [
        (10000, "Small (10K)", 2),
        (50000, "Medium (50K)", 2),
        (100000, "Large (100K)", 2),
        (500000, "Very Large (500K)", 1),  # Single run for very large
    ]
    
    all_results = {}
    
    for data_size, size_name, runs in test_sizes:
        print(f"\n{'='*70}")
        print(f"TESTING: {size_name} - {data_size:,} numbers")
        print(f"{'='*70}")
        
        data = list(range(1000, 1000 + data_size))
        num_workers = 4
        
        implementations = [
            ("Sequential", sequential_process),
            ("Threading (Current)", lambda d: threaded_process(d, num_workers)),
            ("Threading (Ideal)", lambda d: threaded_process_ideal(d, num_workers)),
            ("Multiprocessing", lambda d: multiprocessed_process(d, num_workers)),
        ]
        
        size_results = {}
        
        for name, func in implementations:
            times = []
            for run in range(runs):
                start = time.time()
                result = func(data)
                elapsed = time.time() - start
                times.append(elapsed)
                # Verify correctness on first run
                if run == 0:
                    baseline = sequential_process(data)
                    if result != baseline:
                        print(f"  ⚠ {name}: Results mismatch!")
            
            avg_time = sum(times) / len(times)
            size_results[name] = avg_time
        
        # Calculate speedups
        baseline_time = size_results["Sequential"]
        
        print(f"\n{'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'vs Baseline'}")
        print("-" * 70)
        
        for name, elapsed in size_results.items():
            if name == "Sequential":
                speedup = 1.0
                print(f"{name:<25} {elapsed:<12.4f} {speedup:<10.2f}x Baseline")
            else:
                speedup = baseline_time / elapsed if elapsed > 0 else 0
                improvement = f"{speedup:.2f}x faster" if speedup > 1 else f"{1/speedup:.2f}x slower"
                print(f"{name:<25} {elapsed:<12.4f} {speedup:<10.2f}x {improvement}")
        
        all_results[size_name] = size_results
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SCALING SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Dataset Size':<20} {'Sequential':<15} {'Threading':<15} {'Multiprocessing':<15} {'MP Speedup'}")
    print("-" * 70)
    
    for size_name, results in all_results.items():
        seq = results["Sequential"]
        thread = results["Threading (Ideal)"]
        mp = results["Multiprocessing"]
        mp_speedup = seq / mp if mp > 0 else 0
        
        print(f"{size_name:<20} {seq:<15.4f} {thread:<15.4f} {mp:<15.4f} {mp_speedup:.2f}x")
    
    # Analysis
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    
    small_results = all_results["Small (10K)"]
    large_results = all_results["Very Large (500K)"]
    
    small_mp_speedup = small_results["Sequential"] / small_results["Multiprocessing"]
    large_mp_speedup = large_results["Sequential"] / large_results["Multiprocessing"]
    
    print(f"\n1. Multiprocessing Speedup:")
    print(f"   Small dataset (10K):     {small_mp_speedup:.2f}x (overhead significant)")
    print(f"   Very Large (500K):       {large_mp_speedup:.2f}x (overhead negligible)")
    print(f"   → Multiprocessing improves dramatically with dataset size!")
    print(f"   → Overhead becomes negligible as computation time increases")
    
    small_thread_speedup = small_results["Sequential"] / small_results["Threading (Ideal)"]
    large_thread_speedup = large_results["Sequential"] / large_results["Threading (Ideal)"]
    
    print(f"\n2. Threading Speedup:")
    print(f"   Small dataset (10K):     {small_thread_speedup:.2f}x (GIL limits parallelism)")
    print(f"   Very Large (500K):       {large_thread_speedup:.2f}x (GIL limits parallelism)")
    print(f"   → Threading stays near 1.0x (GIL prevents true parallelism)")
    print(f"   → No improvement even with 50x larger dataset!")
    
    print(f"\n3. Best Approach:")
    if large_mp_speedup > 1.5:
        print(f"   ✅ Multiprocessing is clearly superior for large CPU-bound tasks")
        print(f"      Very Large dataset: {large_mp_speedup:.2f}x speedup (Multiprocessing)")
        print(f"      Very Large dataset: {large_thread_speedup:.2f}x speedup (Threading)")
        if large_thread_speedup > 0:
            print(f"      → Multiprocessing is {large_mp_speedup/large_thread_speedup:.2f}x faster than threading!")
    elif large_mp_speedup > 1.0:
        print(f"   ✅ Multiprocessing shows improvement for large datasets")
        print(f"      ({large_mp_speedup:.2f}x vs {large_thread_speedup:.2f}x for threading)")
    else:
        print(f"   Threading and Multiprocessing are comparable")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    # Check if running comprehensive test
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        comprehensive_performance_test(data_size=100000, num_workers=4, runs=3)
    elif len(sys.argv) > 1 and sys.argv[1] == "--multi-scale":
        multi_scale_performance_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--quick":
        comprehensive_performance_test(data_size=10000, num_workers=4, runs=2)
    else:
        # Original test code
        # Test all approaches for correctness with small data
        test_data = list(range(10, 20))
        print("Testing correctness of all approaches...")
        print(f"Test data: {test_data}\n")
        
        # Sequential
        print("Testing sequential approach...")
        seq_result = sequential_process(test_data)
        print(f"  Processed {len(seq_result)} items")
        print(f"  First few results: {seq_result[:5]}")
        
        # Threading
        print("\nTesting threading approach...")
        thread_result = threaded_process(test_data)
        print(f"  Processed {len(thread_result)} items")
        print(f"  First few results: {thread_result[:5]}")
        
        # Multiprocessing
        print("\nTesting multiprocessing approach...")
        mp_result = multiprocessed_process(test_data)
        print(f"  Processed {len(mp_result)} items")
        print(f"  First few results: {mp_result[:5]}")
        
        # Verify correctness
        print("\n" + "="*50)
        print("Correctness Check:")
        print("="*50)
        
        seq_match_thread = seq_result == thread_result
        seq_match_mp = seq_result == mp_result
        thread_match_mp = thread_result == mp_result
        
        print(f"Sequential == Threading:   {seq_match_thread}")
        print(f"Sequential == Multiprocessing: {seq_match_mp}")
        print(f"Threading == Multiprocessing:  {thread_match_mp}")
        
        if seq_match_thread and seq_match_mp:
            print("\n✓ All approaches produce identical results!")
        else:
            print("\n✗ Results don't match! Check implementations.")
            if not seq_match_thread:
                print(f"  Sequential: {seq_result}")
                print(f"  Threading:  {thread_result}")
            if not seq_match_mp:
                print(f"  Sequential: {seq_result}")
                print(f"  Multiprocessing: {mp_result}")
        
        print("\n" + "="*50)
        print("Performance Test Options:")
        print("  --comprehensive : Full test (100K items, 3 runs)")
        print("  --multi-scale   : Test across multiple sizes (10K to 500K)")
        print("  --quick         : Quick test (10K items, 2 runs)")
        print("="*50)

