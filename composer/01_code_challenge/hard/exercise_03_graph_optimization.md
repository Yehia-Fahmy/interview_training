# Exercise 3: Optimized Graph Algorithm

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Graph algorithms, memory optimization, performance tuning

## Problem

Implement an efficient graph data structure and algorithms for a social network use case:

1. **Graph representation** that's memory-efficient
2. **Shortest path** algorithm (Dijkstra's or BFS)
3. **Connected components** finder
4. Handle graphs with millions of nodes and edges

**Use Case:** Social network where you need to:
- Find shortest connection path between two users
- Find all users in a friend group (connected component)
- Efficiently store and query relationships

## Requirements

1. **Memory-efficient** representation (consider adjacency lists vs matrices)
2. **Fast queries** (O(log n) or better where possible)
3. **Scalable** to millions of nodes

## Solution Template

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

## Key Learning Points

1. **Graph Representations:** Adjacency list (sparse) vs matrix (dense)
2. **Algorithm Choice:** BFS vs Dijkstra's vs A*
3. **Memory Optimization:** Using sets vs lists, sparse representations

## Advanced Optimizations

- Use `array.array` for integer node IDs to save memory
- Implement bidirectional BFS for shortest path
- Cache frequently accessed neighbors
- Consider using `networkx` for comparison (but understand trade-offs)

