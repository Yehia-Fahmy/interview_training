"""
Exercise 3: Optimized Graph Algorithm

Implement an efficient graph data structure and algorithms for a social network use case.
Handle graphs with millions of nodes and edges efficiently.

Requirements:
- Memory-efficient graph representation
- Shortest path algorithm (Dijkstra's or BFS)
- Connected components finder
- Scalable to millions of nodes
"""

from collections import deque, defaultdict
import heapq


class EfficientGraph:
    """
    Memory-efficient graph representation.
    
    Choose appropriate data structure:
    - Adjacency list for sparse graphs (recommended)
    - Consider memory optimizations (sets vs lists, etc.)
    """
    def __init__(self):
        # TODO: Choose appropriate data structure
        # self.graph = defaultdict(set)  # or list, or custom structure
        pass
    
    def add_edge(self, u, v, weight=1):
        """
        Add undirected edge between nodes u and v.
        
        Args:
            u: First node
            v: Second node
            weight: Edge weight (for weighted graphs)
        """
        # TODO: Implement edge addition
        pass
    
    def get_neighbors(self, node):
        """
        Get neighbors of a node.
        
        Args:
            node: Node ID
            
        Returns:
            List of neighbor nodes
        """
        # TODO: Implement neighbor retrieval
        pass
    
    def shortest_path(self, start, end):
        """
        Find shortest path between start and end nodes.
        
        Args:
            start: Starting node
            end: Target node
            
        Returns:
            List of nodes representing the shortest path, or None if no path exists
        """
        # TODO: Implement shortest path
        # Use BFS for unweighted graphs
        # Use Dijkstra's algorithm for weighted graphs
        pass
    
    def connected_components(self):
        """
        Find all connected components in the graph.
        
        Returns:
            List of sets, where each set contains nodes in one component
        """
        # TODO: Implement connected components
        # Use DFS or BFS to traverse each component
        pass
    
    def memory_usage(self):
        """
        Estimate memory usage of the graph.
        
        Returns:
            Approximate memory usage in bytes
        """
        # TODO: Implement memory estimation
        # Consider: nodes, edges, overhead
        pass


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
    print(f"Components: {components}")  # Expected: [{1,2,3,4}, {5,6,7}]
    
    # Test memory usage
    memory = g.memory_usage()
    print(f"Estimated memory usage: {memory} bytes")

