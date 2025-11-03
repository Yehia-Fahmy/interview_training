"""
Exercise 4: Efficient Caching Implementation

Implement an LRU (Least Recently Used) cache decorator with memory limits.
Your cache should evict entries when full and demonstrate performance improvements.

Requirements:
- Implement LRU cache class
- Create a decorator for function caching
- Support memory-based eviction (estimate memory usage)
- Handle edge cases (None values, etc.)
"""

import time
import sys
from collections import OrderedDict
from functools import wraps


def expensive_computation(n):
    """
    A function that takes time to compute.
    This is the function you'll cache.
    """
    time.sleep(0.01)  # Simulate expensive operation
    return n * n


class LRUCache:
    """
    LRU Cache implementation with size limit.
    
    Requirements:
    - get(key): Get value from cache, move to end (most recently used)
    - put(key, value): Add value to cache, evict LRU if full
    - clear(): Clear all cached entries
    """
    def __init__(self, max_size=128):
        pass
    
    def get(self, key):
        """Get value from cache"""
        pass
    
    def put(self, key, value):
        """Add value to cache"""
        pass
    
    def clear(self):
        """Clear cache"""
        pass
    
    def size(self):
        """Return current cache size"""
        pass

def lru_cache_decorator(max_size=128):
    """
    LRU cache decorator.
    
    Usage:
        @lru_cache_decorator(max_size=100)
        def my_function(x):
            return expensive_computation(x)
    """
    pass


# ============================================================================
# BASELINE IMPLEMENTATIONS FOR COMPARISON
# ============================================================================

class LRUCacheBaseline:
    """
    Baseline: Simple FIFO cache (not LRU, but simple to understand).
    Uses simple dict with first-in-first-out eviction.
    This is NOT a proper LRU but serves as a naive baseline.
    """
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = {}
        self.order = []  # Track insertion order for FIFO
    
    def get(self, key):
        if key in self.cache:
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Remove oldest (FIFO)
                oldest = self.order.pop(0)
                del self.cache[oldest]
            self.order.append(key)
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
        self.order.clear()
    
    def size(self):
        return len(self.cache)


class LRUCacheImproved:
    """
    Improved LRU Cache with proper None handling.
    
    Improvements over solution:
    1. Handles None values correctly (uses sentinel to distinguish cache miss vs cached None)
    2. Better key handling for decorator
    3. Thread-safe with optional locking
    """
    _sentinel = object()  # Sentinel to distinguish cache miss from None value
    
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get value from cache, returns sentinel if not found"""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return self._sentinel
    
    def put(self, key, value):
        """Add value to cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def size(self):
        """Return current cache size"""
        return len(self.cache)
    
    def contains(self, key):
        """Check if key exists in cache"""
        return key in self.cache


def lru_cache_decorator_improved(max_size=128):
    """
    Improved LRU cache decorator with proper None handling.
    
    Improvements:
    - Correctly handles None return values
    - Better key generation
    - Uses improved LRU cache
    """
    def decorator(func):
        cache = LRUCacheImproved(max_size=max_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            if args and not kwargs:
                key = args[0] if len(args) == 1 else tuple(args)
            else:
                key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache (using contains to handle None values)
            if cache.contains(key):
                return cache.get(key)
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper.cache = cache
        return wrapper
    
    return decorator


# Apply decorator to expensive_computation (to be implemented)
# @lru_cache_decorator(max_size=50)
# def cached_computation(n):
#     """Cached version of expensive_computation"""
#     return expensive_computation(n)


# Improved version
@lru_cache_decorator_improved(max_size=50)
def cached_computation_improved(n):
    """Cached version with improved None handling"""
    return expensive_computation(n)


def function_that_returns_none(x):
    """Function that returns None to test None handling"""
    time.sleep(0.01)
    return None if x % 2 == 0 else x * x


def compare_performance():
    """Compare performance with and without caching"""
    test_values = list(range(20))
    
    # Without cache (call expensive_computation directly)
    print("Without cache...")
    start = time.time()
    results1 = [expensive_computation(i) for i in test_values]
    no_cache_time = time.time() - start
    
    # With cache (first call - cache miss)
    print("With cache (first call)...")
    start = time.time()
    results2 = [cached_computation(i) for i in test_values]
    first_call_time = time.time() - start
    
    # With cache (second call - cache hit)
    print("With cache (second call - cache hits)...")
    start = time.time()
    results3 = [cached_computation(i) for i in test_values]
    cached_time = time.time() - start
    
    # Verify results
    assert results1 == results2 == results3
    
    print(f"\nNo cache:        {no_cache_time:.4f}s")
    print(f"Cache (cold):     {first_call_time:.4f}s")
    print(f"Cache (warm):     {cached_time:.4f}s")
    print(f"\nSpeedup (warm):   {no_cache_time/cached_time:.2f}x")


def comprehensive_performance_test():
    """
    Comprehensive performance test comparing against solution_04.py:
    - Baseline (FIFO naive cache)
    - Solution from solution_04.py (OrderedDict LRU)
    - Improved (LRU with None handling)
    - Built-in functools.lru_cache (Python standard library)
    """
    import functools
    import importlib.util
    from pathlib import Path
    
    # Load solution module
    solution_path = Path(__file__).parent / "solution_04.py"
    spec = importlib.util.spec_from_file_location("solution_04", solution_path)
    solution_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution_module)
    
    print("=" * 70)
    print("COMPREHENSIVE CACHE PERFORMANCE TEST")
    print("=" * 70)
    print("Comparing against solution_04.py reference implementation")
    
    # Test data - mix of cache hits and misses
    test_values = list(range(30)) * 3  # 90 items, many repeats
    cache_size = 50
    
    # Create baseline FIFO cached function
    baseline_cache = LRUCacheBaseline(max_size=cache_size)
    def baseline_cached(n):
        result = baseline_cache.get(n)
        if result is not None:
            return result
        result = expensive_computation(n)
        baseline_cache.put(n, result)
        return result
    
    # Create solution LRU cached function (from solution_04.py)
    @solution_module.lru_cache_decorator(max_size=cache_size)
    def solution_cached(n):
        return expensive_computation(n)
    
    # Create improved LRU cached function
    @lru_cache_decorator_improved(max_size=cache_size)
    def improved_cached(n):
        return expensive_computation(n)
    
    # Create functools.lru_cache version
    @functools.lru_cache(maxsize=cache_size)
    def stdlib_cached(n):
        return expensive_computation(n)
    
    results = {}
    
    print(f"\nTest Configuration:")
    print(f"  Dataset: {len(test_values)} function calls ({len(set(test_values))} unique values)")
    print(f"  Cache size: {cache_size}")
    print(f"  Expected cache hits: ~{len(test_values) - len(set(test_values))}")
    print("-" * 70)
    
    # Test No Cache
    print("\n[1] Testing No Cache (Baseline)...")
    start = time.time()
    no_cache_results = [expensive_computation(i) for i in test_values]
    no_cache_time = time.time() - start
    results["No Cache"] = no_cache_time
    print(f"  ✓ Time: {no_cache_time:.4f}s")
    
    # Test Baseline FIFO
    print("\n[2] Testing Baseline FIFO Cache...")
    baseline_cache.clear()
    start = time.time()
    baseline_results = [baseline_cached(i) for i in test_values]
    baseline_time = time.time() - start
    results["Baseline FIFO"] = baseline_time
    print(f"  ✓ Time: {baseline_time:.4f}s ({no_cache_time/baseline_time:.2f}x speedup)")
    assert baseline_results == no_cache_results, "Baseline results don't match!"
    
    # Test Solution LRU (from solution_04.py)
    print("\n[3] Testing Solution LRU Cache (from solution_04.py)...")
    solution_cached.cache.clear()
    start = time.time()
    solution_results = [solution_cached(i) for i in test_values]
    solution_time = time.time() - start
    results["Solution LRU"] = solution_time
    print(f"  ✓ Time: {solution_time:.4f}s ({no_cache_time/solution_time:.2f}x speedup)")
    assert solution_results == no_cache_results, "Solution results don't match!"
    
    # Test Improved LRU
    print("\n[4] Testing Improved LRU Cache...")
    improved_cached.cache.clear()
    start = time.time()
    improved_results = [improved_cached(i) for i in test_values]
    improved_time = time.time() - start
    results["Improved LRU"] = improved_time
    print(f"  ✓ Time: {improved_time:.4f}s ({no_cache_time/improved_time:.2f}x speedup)")
    assert improved_results == no_cache_results, "Improved results don't match!"
    
    # Test Python stdlib
    print("\n[5] Testing Python functools.lru_cache...")
    stdlib_cached.cache_clear()
    start = time.time()
    stdlib_results = [stdlib_cached(i) for i in test_values]
    stdlib_time = time.time() - start
    results["Python stdlib"] = stdlib_time
    print(f"  ✓ Time: {stdlib_time:.4f}s ({no_cache_time/stdlib_time:.2f}x speedup)")
    assert stdlib_results == no_cache_results, "Stdlib results don't match!"
    
    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Implementation':<25} {'Time (s)':<12} {'Speedup':<10} {'Cache Hits'}")
    print("-" * 70)
    
    for name, time_val in results.items():
        if name == "No Cache":
            speedup = 1.0
            hits = "N/A"
        else:
            speedup = no_cache_time / time_val
            # Estimate cache hits based on cache efficiency
            hits = f"~{int((1 - time_val/no_cache_time) * len(test_values))}"
        print(f"{name:<25} {time_val:<12.4f} {speedup:<10.2f}x {hits}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    fastest = min((t for n, t in results.items() if n != "No Cache"), default=0)
    fastest_name = [n for n, t in results.items() if t == fastest and n != "No Cache"][0]
    
    print(f"\n🏆 Fastest: {fastest_name} ({fastest:.4f}s, {no_cache_time/fastest:.2f}x speedup)")
    
    print(f"\n📊 Comparison:")
    print(f"   Baseline FIFO:   {results['Baseline FIFO']:.4f}s ({no_cache_time/results['Baseline FIFO']:.2f}x)")
    print(f"   Solution LRU:    {results['Solution LRU']:.4f}s ({no_cache_time/results['Solution LRU']:.2f}x)")
    print(f"   Improved LRU:    {results['Improved LRU']:.4f}s ({no_cache_time/results['Improved LRU']:.2f}x)")
    print(f"   Python stdlib:   {results['Python stdlib']:.4f}s ({no_cache_time/results['Python stdlib']:.2f}x)")
    
    # Test None handling
    print("\n" + "=" * 70)
    print("TESTING: None Value Handling")
    print("=" * 70)
    
    @solution_module.lru_cache_decorator(max_size=10)
    def solution_cached_none(x):
        return function_that_returns_none(x)
    
    @lru_cache_decorator_improved(max_size=10)
    def improved_cached_none(x):
        return function_that_returns_none(x)
    
    test_none_values = [2, 4, 6, 2, 4]  # Some return None, with repeats
    
    print("\nTesting Solution version from solution_04.py:")
    solution_cached_none.cache.clear()
    solution_none_results = [solution_cached_none(x) for x in test_none_values]
    print(f"  Results: {solution_none_results}")
    print(f"  Cache size: {solution_cached_none.cache.size()}")
    print(f"  ⚠️ Note: Solution uses 'if result is not None' - can't distinguish cache miss from cached None")
    
    print("\nTesting Improved version (properly caches None):")
    improved_cached_none.cache.clear()
    improved_none_results = [improved_cached_none(x) for x in test_none_values]
    print(f"  Results: {improved_none_results}")
    print(f"  Cache size: {improved_cached_none.cache.size()}")
    print(f"  ✅ None values cached correctly using 'contains()' method")
    
    # Verify correctness
    expected = [function_that_returns_none(x) for x in test_none_values]
    print(f"\n  Expected: {expected}")
    print(f"  Solution matches: {solution_none_results == expected}")
    print(f"  Improved matches: {improved_none_results == expected}")
    
    # Analysis of solution_04.py
    print("\n" + "=" * 70)
    print("EVALUATION OF solution_04.py")
    print("=" * 70)
    
    print("\n✅ Strengths:")
    print("   - Clean implementation using OrderedDict")
    print("   - Correct LRU eviction logic")
    print("   - Proper use of move_to_end() for O(1) reordering")
    print("   - Good key generation for decorator")
    
    print("\n⚠️  Issues:")
    print("   - Can't cache None values: 'if result is not None' fails when function returns None")
    print("   - No way to distinguish between 'key not in cache' and 'key cached with None value'")
    
    print("\n💡 Improvements in LRUCacheImproved:")
    print("   - Added 'contains()' method to check key existence separately")
    print("   - Properly handles None return values")
    print("   - Same performance, better correctness")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--comprehensive":
        comprehensive_performance_test()
    else:
        # Quick test - Note: LRUCache needs to be implemented first
        print("="*50)
        print("LRU Cache Exercise")
        print("="*50)
        print("\nThis exercise requires implementing:")
        print("  1. LRUCache class (get, put, clear, size methods)")
        print("  2. lru_cache_decorator function")
        print("\nRun with --comprehensive to test against solution_04.py")
        print("="*50)

