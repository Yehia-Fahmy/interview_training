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


# Apply decorator to expensive_computation
@lru_cache_decorator(max_size=50)
def cached_computation(n):
    """Cached version of expensive_computation"""
    return expensive_computation(n)


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


if __name__ == "__main__":
    # Quick test
    cache = LRUCache(max_size=3)
    cache.put(1, "one")
    cache.put(2, "two")
    cache.put(3, "three")
    print(f"Cache size: {cache.size()}")
    print(f"Get 1: {cache.get(1)}")
    cache.put(4, "four")  # Should evict 2 (LRU)
    print(f"Get 2: {cache.get(2)}")  # Should return None or raise KeyError

