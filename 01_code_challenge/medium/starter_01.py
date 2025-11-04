"""
Exercise 1: Custom Memory-Efficient Data Structure

Implement a memory-efficient counter that handles millions of entries
with many duplicates. Your solution should use significantly less memory
than a standard dictionary while maintaining O(1) average-case access.

Requirements:
- add(value): Add or increment count for a value
- get_count(value): Return count for value, 0 if not present
- get_all_items(): Return list of (value, count) tuples
- memory_usage(): Return approximate memory usage in bytes
"""

import sys
from collections import defaultdict


class EfficientCounter:
    """
    Your custom memory-efficient counter implementation.
    
    Hint: Consider using collections.Counter or defaultdict as baseline,
    then optimize for memory when there are many duplicates.
    """
    def __init__(self):
        self.list = []
        self.max_elements = 0
    
    def add(self, value):
        """Add or increment count for value"""
        if value >= self.max_elements:
            self.max_elements = value
            new_list = [0 for _ in range(value+1)]
            for v in range(len(self.list)):
                new_list[v] = self.list[v]
            self.list = new_list
        self.list[value] += 1
    
    def get_count(self, value):
        """Return count for value, 0 if not present"""
        if self.max_elements == 0: return 0
        if value > self.max_elements: return 0
        return self.list[value]
    
    def get_all_items(self):
        """Return list of (value, count) tuples"""
        if self.max_elements == 0: return []
        ans = []
        for val in range(self.max_elements+1):
            if self.list[val] > 0:
                ans.append((val, self.list[val]))
        return ans
    
    def memory_usage(self):
        """Return approximate memory usage in bytes"""
        return self.max_elements * 26


def compare_implementations():
    """Compare your implementation with standard dictionary"""
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] * 100000
    
    # Standard dictionary approach
    standard = defaultdict(int)
    for val in data:
        standard[val] += 1
    
    # Your implementation
    efficient = EfficientCounter()
    for val in data:
        efficient.add(val)
    
    # Verify correctness
    assert efficient.get_count(1) == standard[1]
    assert efficient.get_count(2) == standard[2]
    assert efficient.get_count(3) == standard[3]
    
    # Compare memory
    standard_memory = sys.getsizeof(standard) + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in standard.items())
    efficient_memory = efficient.memory_usage()
    
    print(f"Standard dict memory: {standard_memory:,} bytes")
    print(f"Efficient counter memory: {efficient_memory:,} bytes")
    print(f"Memory savings: {(1 - efficient_memory/standard_memory)*100:.1f}%")


if __name__ == "__main__":
    # Quick test
    counter = EfficientCounter()
    counter.add(1)
    counter.add(2)
    counter.add(2)
    print(f"Count of 1: {counter.get_count(1)}")
    print(f"Count of 2: {counter.get_count(2)}")
    print(f"All items: {counter.get_all_items()}")

