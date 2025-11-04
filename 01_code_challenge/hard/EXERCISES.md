# Code Challenge - Hard Exercises

These exercises focus on advanced Python concepts, memory management, and complex optimization techniques.

---

## Exercise 1: Memory-Mapped File Processing

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Memory-mapped files, handling datasets larger than RAM

### Problem

You need to process a binary file containing millions of integers, where the file is larger than available RAM. Using standard file I/O causes memory issues.

**Requirements:**
- Read and process a large binary file (e.g., 10GB+) containing integers
- Find the maximum value, minimum value, and sum
- Use memory mapping to avoid loading entire file into memory
- Handle edge cases (empty file, negative numbers, etc.)

### Tasks

1. **Create** a binary file generator for testing
2. **Implement** processing using memory-mapped files (`mmap`)
3. **Compare** performance and memory usage vs standard file I/O
4. **Handle** file sizes larger than available RAM

### Solution Template

```python
import mmap
import struct
import os

def generate_test_file(filename, num_integers=1000000):
    """Generate a binary file with integers"""
    with open(filename, 'wb') as f:
        for i in range(num_integers):
            # Write 4-byte integers
            f.write(struct.pack('i', i % 1000000))

def process_with_mmap(filename):
    """Process file using memory mapping"""
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # Your implementation using mmap
    with open(filename, 'rb') as f:
        # Create memory map
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Process in chunks to handle large files
            # mm can be accessed like bytes
            pass
    
    return stats

def process_standard_io(filename):
    """Process file using standard I/O (for comparison)"""
    stats = {
        'min': float('inf'),
        'max': float('-inf'),
        'sum': 0,
        'count': 0
    }
    
    # Standard implementation
    pass
    
    return stats

if __name__ == "__main__":
    test_file = 'large_data.bin'
    
    # Generate test file (uncomment when ready)
    # generate_test_file(test_file, 10000000)
    
    # Compare approaches
    # mmap_stats = process_with_mmap(test_file)
    # std_stats = process_standard_io(test_file)
    
    # print("Memory-mapped:", mmap_stats)
    # print("Standard I/O:", std_stats)
```

### Key Learning Points

1. **Memory Mapping:** Allows accessing file data without loading into RAM
2. **Chunk Processing:** Handle files in chunks for very large files
3. **Binary Data:** Using `struct` for binary I/O

### Advanced Considerations

- Process file in chunks (e.g., 1MB at a time)
- Handle endianness if cross-platform
- Consider using `numpy.memmap` for numeric data

---

## Exercise 2: Garbage Collection Optimization

**Difficulty:** Hard  
**Time Limit:** 50 minutes  
**Focus:** Python's garbage collector, circular references, optimization

### Problem

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

### Tasks

1. **Profile** garbage collection behavior (GC counts, time spent)
2. **Optimize** by managing object lifecycles better
3. **Disable or tune** GC for performance-critical sections
4. **Compare** performance before and after

### Solution Template

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

### Key Learning Points

1. **GC Behavior:** Understanding when GC runs and its impact
2. **Circular References:** Breaking them explicitly
3. **GC Tuning:** Using `gc.disable()`/`gc.enable()` strategically
4. **Object Pools:** Reusing objects to reduce allocations

### Advanced Strategies

- Use `__slots__` to reduce memory per object
- Break circular references explicitly
- Disable GC during critical sections
- Use object pools for frequently created objects

---

## Exercise 3: Optimized Graph Algorithm

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Graph algorithms, memory optimization, performance tuning

### Problem

Implement an efficient graph data structure and algorithms for a social network use case:

1. **Graph representation** that's memory-efficient
2. **Shortest path** algorithm (Dijkstra's or BFS)
3. **Connected components** finder
4. Handle graphs with millions of nodes and edges

**Use Case:** Social network where you need to:
- Find shortest connection path between two users
- Find all users in a friend group (connected component)
- Efficiently store and query relationships

### Requirements

1. **Memory-efficient** representation (consider adjacency lists vs matrices)
2. **Fast queries** (O(log n) or better where possible)
3. **Scalable** to millions of nodes

### Solution Template

```python
from collections import deque, defaultdict
import heapq

class EfficientGraph:
    """Memory-efficient graph representation"""
    def __init__(self):
        # Choose: adjacency list vs other structures
        self.graph = defaultdict(set)  # or list, or custom structure
    
    def add_edge(self, u, v):
        """Add undirected edge"""
        pass
    
    def get_neighbors(self, node):
        """Get neighbors of a node"""
        pass
    
    def shortest_path(self, start, end):
        """Find shortest path using BFS (unweighted) or Dijkstra (weighted)"""
        # BFS for unweighted, Dijkstra for weighted
        pass
    
    def connected_components(self):
        """Find all connected components"""
        # Use DFS or BFS
        pass
    
    def memory_usage(self):
        """Estimate memory usage"""
        pass

# Test
if __name__ == "__main__":
    g = EfficientGraph()
    
    # Build a test graph
    edges = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
    for u, v in edges:
        g.add_edge(u, v)
    
    # Test shortest path
    path = g.shortest_path(1, 4)
    print(f"Path from 1 to 4: {path}")  # Expected: [1, 2, 3, 4]
    
    # Test connected components
    components = g.connected_components()
    print(f"Components: {components}")
```

### Key Learning Points

1. **Graph Representations:** Adjacency list (sparse) vs matrix (dense)
2. **Algorithm Choice:** BFS vs Dijkstra's vs A*
3. **Memory Optimization:** Using sets vs lists, sparse representations

### Advanced Optimizations

- Use `array.array` for integer node IDs to save memory
- Implement bidirectional BFS for shortest path
- Cache frequently accessed neighbors
- Consider using `networkx` for comparison (but understand trade-offs)

---

## Exercise 4: Concurrent Producer-Consumer Pattern

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Concurrency, thread safety, queue optimization

### Problem

Implement a high-performance producer-consumer system where:

1. **Multiple producers** generate data items
2. **Multiple consumers** process items
3. **Bounded queue** to prevent memory overflow
4. **Thread-safe** operations
5. **Graceful shutdown** handling

**Requirements:**
- Handle 10 producers and 5 consumers
- Queue should be bounded (max 1000 items)
- Ensure all items are processed
- Measure throughput (items/second)

### Solution Template

```python
import threading
import queue
import time
import random

class ProducerConsumerSystem:
    """Thread-safe producer-consumer system"""
    def __init__(self, num_producers=10, num_consumers=5, queue_size=1000):
        self.queue = queue.Queue(maxsize=queue_size)
        self.num_producers = num_producers
        self.num_consumers = num_consumers
        self.producers = []
        self.consumers = []
        self.shutdown_event = threading.Event()
        self.processed_count = 0
        self.lock = threading.Lock()
    
    def producer(self, producer_id):
        """Producer thread function"""
        items_produced = 0
        while not self.shutdown_event.is_set():
            try:
                # Generate item
                item = f"item_{producer_id}_{items_produced}"
                
                # Add to queue with timeout
                self.queue.put(item, timeout=1)
                items_produced += 1
                
                # Simulate work
                time.sleep(random.uniform(0.001, 0.01))
            except queue.Full:
                continue
        
        print(f"Producer {producer_id} produced {items_produced} items")
    
    def consumer(self, consumer_id):
        """Consumer thread function"""
        items_processed = 0
        while not self.shutdown_event.is_set() or not self.queue.empty():
            try:
                # Get item with timeout
                item = self.queue.get(timeout=1)
                
                # Process item (simulate work)
                self.process_item(item)
                
                self.queue.task_done()
                items_processed += 1
                
                with self.lock:
                    self.processed_count += 1
            except queue.Empty:
                continue
        
        print(f"Consumer {consumer_id} processed {items_processed} items")
    
    def process_item(self, item):
        """Process a single item"""
        # Simulate processing
        time.sleep(random.uniform(0.005, 0.02))
    
    def start(self):
        """Start producers and consumers"""
        # Create and start producer threads
        for i in range(self.num_producers):
            p = threading.Thread(target=self.producer, args=(i,))
            p.start()
            self.producers.append(p)
        
        # Create and start consumer threads
        for i in range(self.num_consumers):
            c = threading.Thread(target=self.consumer, args=(i,))
            c.start()
            self.consumers.append(c)
    
    def stop(self):
        """Gracefully shutdown the system"""
        # Signal shutdown
        self.shutdown_event.set()
        
        # Wait for producers
        for p in self.producers:
            p.join()
        
        # Wait for queue to empty
        self.queue.join()
        
        # Wait for consumers
        for c in self.consumers:
            c.join()
        
        print(f"Total items processed: {self.processed_count}")

if __name__ == "__main__":
    system = ProducerConsumerSystem(num_producers=10, num_consumers=5)
    
    start_time = time.time()
    system.start()
    
    # Run for 10 seconds
    time.sleep(10)
    
    system.stop()
    elapsed = time.time() - start_time
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {system.processed_count / elapsed:.2f} items/sec")
```

### Key Learning Points

1. **Thread Safety:** Using `queue.Queue` for safe communication
2. **Bounded Queues:** Preventing memory issues with `maxsize`
3. **Graceful Shutdown:** Coordinating thread termination
4. **Synchronization:** Using events and locks appropriately

### Advanced Considerations

- Compare with `multiprocessing.Queue` for CPU-bound tasks
- Implement priority queues for prioritized processing
- Add monitoring/metrics collection
- Handle exceptions gracefully

