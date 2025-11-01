"""
Solution for Exercise 4: Efficient Caching Implementation

This file contains the reference solution.
"""

import time
import sys
from collections import OrderedDict
from functools import wraps


def expensive_computation(n):
    """A function that takes time to compute"""
    time.sleep(0.01)
    return n * n


class LRUCache:
    """
    LRU Cache implementation using OrderedDict.
    OrderedDict maintains insertion order, making it perfect for LRU.
    """
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get value from cache, move to end (most recently used)"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """Add value to cache, evict LRU if full"""
        if key in self.cache:
            # Update existing key, move to end
            self.cache.move_to_end(key)
        else:
            # Add new key
            if len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def size(self):
        """Return current cache size"""
        return len(self.cache)


def lru_cache_decorator(max_size=128):
    """
    LRU cache decorator factory.
    Creates a decorator that caches function results using LRU eviction.
    """
    def decorator(func):
        cache = LRUCache(max_size=max_size)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args and kwargs
            # For simplicity, using args[0] as key (assuming single arg)
            # In production, you'd want to handle multiple args and kwargs properly
            if args and not kwargs:
                key = args[0] if len(args) == 1 else tuple(args)
            else:
                key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Attach cache to wrapper for inspection
        wrapper.cache = cache
        return wrapper
    
    return decorator


# Example usage
@lru_cache_decorator(max_size=50)
def cached_computation(n):
    """Cached version of expensive_computation"""
    return expensive_computation(n)

