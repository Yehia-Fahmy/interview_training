# Exercise 4: Implementing Efficient Caching

**Difficulty:** Medium  
**Time Limit:** 40 minutes  
**Focus:** Caching strategies, LRU cache implementation

## Problem

Implement a caching mechanism for a computationally expensive function. You need to handle:

1. **LRU (Least Recently Used) eviction** when cache is full
2. **Memory limits** - cache should not exceed a certain memory size
3. **Thread safety** (optional but preferred)

**Given Function:**
```python
def expensive_computation(n):
    """A function that takes time to compute"""
    import time
    time.sleep(0.1)  # Simulate expensive operation
    return n * n
```

## Tasks

1. **Implement** an LRU cache decorator
2. **Add** memory-based eviction (estimate memory usage of cached values)
3. **Compare** performance with and without caching
4. **Handle** edge cases (None values, unhashable types, etc.)

## Solution Template

```python
from collections import OrderedDict
import functools

class LRUCache:
    """LRU Cache implementation"""
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key):
        """Get value from cache"""
        pass
    
    def put(self, key, value):
        """Add value to cache"""
        pass
    
    def clear(self):
        """Clear cache"""
        pass

def lru_cache_decorator(max_size=128):
    """LRU cache decorator"""
    # Your implementation
    pass

# Usage
@lru_cache_decorator(max_size=100)
def expensive_computation(n):
    import time
    time.sleep(0.1)
    return n * n

# Test
if __name__ == "__main__":
    import time
    
    # Without cache
    start = time.time()
    for i in range(10):
        expensive_computation(i)
    no_cache_time = time.time() - start
    
    # With cache (call same values)
    start = time.time()
    for i in range(10):
        expensive_computation(i)
    cached_time = time.time() - start
    
    print(f"No cache: {no_cache_time:.2f}s")
    print(f"With cache: {cached_time:.2f}s")
    print(f"Speedup: {no_cache_time/cached_time:.2f}x")
```

## Key Learning Points

1. **LRU Algorithm:** OrderedDict is perfect for LRU implementation
2. **Decorator Pattern:** Creating reusable caching decorators
3. **Memory Management:** Balancing cache size vs performance

## Advanced Considerations

- Consider using `functools.lru_cache` from standard library for comparison
- Implement memory-based eviction by estimating object sizes
- Add thread safety using locks if needed

