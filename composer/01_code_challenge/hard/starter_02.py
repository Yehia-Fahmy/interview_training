"""
Exercise 2: Garbage Collection Optimization

Optimize code that creates many temporary objects, causing frequent garbage 
collection pauses that impact performance.

Requirements:
- Profile garbage collection behavior (GC counts, time spent)
- Optimize by managing object lifecycles better
- Disable or tune GC for performance-critical sections
- Compare performance before and after
"""

import gc
import time
import sys


class Node:
    """Original Node class - may create circular references"""
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)


def build_tree(depth, breadth):
    """
    Build a tree structure that may create circular references.
    
    Args:
        depth: Depth of tree
        breadth: Number of children per node
        
    Returns:
        Root node of tree
    """
    if depth == 0:
        return None
    
    node = Node(depth)
    for i in range(breadth):
        child = build_tree(depth - 1, breadth)
        if child:
            node.add_child(child)
    return node


def process_data():
    """
    Function that creates many temporary objects (original version).
    
    Returns:
        List of trees (may cause memory issues)
    """
    trees = []
    for i in range(100):
        tree = build_tree(5, 3)
        trees.append(tree)
        # Process tree somehow
        # trees are no longer needed but kept in list
    return trees


def profile_gc(func):
    """
    Decorator to profile GC behavior.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapper function that measures GC stats
    """
    def wrapper(*args, **kwargs):
        # TODO: Implement GC profiling
        # Disable GC, collect, then enable and measure
        # Track GC counts before and after
        # Measure execution time
        
        result = func(*args, **kwargs)
        
        # Print GC statistics
        # print(f"Time: {elapsed:.2f}s")
        # print(f"GC collections: {gc_collections}")
        
        return result
    return wrapper


class OptimizedNode:
    """
    Optimized Node class with better memory management.
    
    TODO: Implement optimizations:
    - Use __slots__ to reduce memory overhead
    - Add cleanup method to break circular references
    """
    # TODO: Add __slots__
    
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
    
    def cleanup(self):
        """Explicitly break references to help GC"""
        # TODO: Implement cleanup
        pass


def build_tree_optimized(depth, breadth):
    """
    Build tree using optimized Node class.
    
    Args:
        depth: Depth of tree
        breadth: Number of children per node
        
    Returns:
        Root node of tree
    """
    # TODO: Implement using OptimizedNode
    pass


@profile_gc
def process_data_optimized():
    """
    Optimized version that manages object lifecycles better.
    
    TODO: Implement optimizations:
    1. Break circular references explicitly
    2. Reuse objects where possible
    3. Consider disabling GC for short periods
    4. Clean up objects when done
    """
    # TODO: Implement optimized version
    pass


if __name__ == "__main__":
    print("Testing garbage collection optimization...")
    
    # Test original version
    print("\n--- Original Version ---")
    start = time.time()
    result1 = process_data()
    elapsed1 = time.time() - start
    print(f"Time: {elapsed1:.2f}s")
    
    # Test optimized version
    print("\n--- Optimized Version ---")
    result2 = process_data_optimized()
    
    # Compare results
    print("\n--- Comparison ---")
    print(f"Original time: {elapsed1:.2f}s")
    # print(f"Optimized time: {elapsed2:.2f}s")
    # print(f"Speedup: {elapsed1/elapsed2:.2f}x")

