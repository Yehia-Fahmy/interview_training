# Exercise 2: Garbage Collection Optimization

**Difficulty:** Hard  
**Time Limit:** 50 minutes  
**Focus:** Python's garbage collector, circular references, optimization

## Problem

You have code that creates many temporary objects, causing frequent garbage collection pauses that impact performance. You need to optimize the garbage collection behavior.

**Given Code:**
```python
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)

def build_tree(depth, breadth):
    """Build a tree structure that may create circular references"""
    if depth == 0:
        return None
    
    node = Node(depth)
    for i in range(breadth):
        child = build_tree(depth - 1, breadth)
        if child:
            node.add_child(child)
    return node

def process_data():
    """Function that creates many temporary objects"""
    trees = []
    for i in range(100):
        tree = build_tree(5, 3)
        trees.append(tree)
        # Process tree somehow
        # trees are no longer needed but kept in list
    return trees
```

## Tasks

1. **Profile** garbage collection behavior (GC counts, time spent)
2. **Optimize** by managing object lifecycles better
3. **Disable or tune** GC for performance-critical sections
4. **Compare** performance before and after

## Solution Template

```python
import gc
import time

def profile_gc(func):
    """Decorator to profile GC behavior"""
    def wrapper(*args, **kwargs):
        # Disable GC for measurement
        gc.disable()
        gc.collect()
        
        # Enable and measure
        gc.enable()
        gc_counts_before = [gc.get_count()]
        
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        gc_counts_after = gc.get_count()
        gc_collections = sum(gc_counts_after) - sum(gc_counts_before)
        
        print(f"Time: {elapsed:.2f}s")
        print(f"GC collections: {gc_collections}")
        print(f"GC stats: {gc.get_stats()}")
        
        return result
    return wrapper

@profile_gc
def process_data_optimized():
    """Optimized version"""
    # Your optimizations:
    # 1. Break circular references explicitly
    # 2. Reuse objects where possible
    # 3. Use __slots__ to reduce memory
    # 4. Consider disabling GC for short periods
    pass

# Alternative: Optimized Node class
class OptimizedNode:
    """Node with optimizations"""
    __slots__ = ['value', 'children']  # Reduce memory overhead
    
    def __init__(self, value):
        self.value = value
        self.children = []
    
    def cleanup(self):
        """Explicitly break references"""
        self.children = None
```

## Key Learning Points

1. **GC Behavior:** Understanding when GC runs and its impact
2. **Circular References:** Breaking them explicitly
3. **GC Tuning:** Using `gc.disable()`/`gc.enable()` strategically
4. **Object Pools:** Reusing objects to reduce allocations

## Advanced Strategies

- Use `__slots__` to reduce memory per object
- Break circular references explicitly
- Disable GC during critical sections
- Use object pools for frequently created objects

