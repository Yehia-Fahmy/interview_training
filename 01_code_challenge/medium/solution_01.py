"""
Solution for Exercise 1: Custom Memory-Efficient Data Structure

This file contains the reference solution. It's kept separate so you
can't see it while working on your implementation.
"""

import sys
from collections import Counter


class EfficientCounter:
    """
    Memory-efficient counter using collections.Counter.
    Counter is optimized for counting and uses less memory than defaultdict
    when there are many duplicates because it stores counts more efficiently.
    """
    def __init__(self):
        self._counter = Counter()
    
    def add(self, value):
        """Add or increment count for value"""
        self._counter[value] += 1
    
    def get_count(self, value):
        """Return count for value, 0 if not present"""
        return self._counter.get(value, 0)
    
    def get_all_items(self):
        """Return list of (value, count) tuples"""
        return list(self._counter.items())
    
    def memory_usage(self):
        """Return approximate memory usage in bytes"""
        # Counter is a dict subclass, so we can estimate its size
        base_size = sys.getsizeof(self._counter)
        # Add size of keys and values
        for key, count in self._counter.items():
            base_size += sys.getsizeof(key) + sys.getsizeof(count)
        return base_size


# Alternative solution using __slots__ for even better memory efficiency
class EfficientCounterSlots:
    """
    Even more memory-efficient version using __slots__.
    This prevents creation of __dict__ for each instance.
    """
    __slots__ = ('_counter',)
    
    def __init__(self):
        self._counter = Counter()
    
    def add(self, value):
        self._counter[value] += 1
    
    def get_count(self, value):
        return self._counter.get(value, 0)
    
    def get_all_items(self):
        return list(self._counter.items())
    
    def memory_usage(self):
        base_size = sys.getsizeof(self._counter)
        for key, count in self._counter.items():
            base_size += sys.getsizeof(key) + sys.getsizeof(count)
        return base_size

