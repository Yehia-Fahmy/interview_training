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
        self.graph = defaultdict(set)
        self.weighted = False
    
    def add_edge(self, u, v, weight=1):
        """
        Add undirected edge between nodes u and v.
        
        Args:
            u: First node
            v: Second node
            weight: Edge weight (for weighted graphs)
        """
        # TODO: Implement edge addition
        if self.weighted:
            self.graph[u].add((v, weight))
            self.graph[v].add((u, weight))
        else:
            self.graph[u].add(v)
            self.graph[v].add(u)
    
    def get_neighbors(self, node):
        """
        Get neighbors of a node.
        
        Args:
            node: Node ID
            
        Returns:
            List of neighbor nodes
        """
        # TODO: Implement neighbor retrieval
        neighbors = self.graph[node]
        if self.weighted:
            neighbors = [n for n, _ in neighbors]
        return neighbors
    
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
        if self.weighted:
            if start == end:
                return [start]
            solved_paths = defaultdict()
            possible = []
            solved_paths[start] = [start]
            for n, w in self.graph[start]:
                heapq.heappush(possible, (w, (n, [start])))
            while possible:
                # add new node to solved
                w, (n, path) = heapq.heappop(possible)
                if n in solved_paths: continue
                path.append(n)
                solved_paths[n] = path
                if n == end: return solved_paths[n]
                # add new neighbors to heap
                for neighbor, weight in self.graph[n]:
                    heapq.heappush(possible, (w + weight, (neighbor, path.copy())))
            return None
        else:
            q = deque()
            q.append((start, [start], set([start])))
            best_path = None
            while q:
                cur, p, s = q.popleft()
                if cur == end:
                    if not best_path or len(p) < len(best_path): best_path = p
                neighbors = self.get_neighbors(cur)
                for n in neighbors:
                    if not n in s:
                        temp_path = p.copy()
                        temp_seen = s.copy()
                        temp_path.append(n)
                        temp_seen.add(n)
                        q.append((n, temp_path, temp_seen))
        return best_path
    
    def connected_components(self):
        """
        Find all connected components in the graph.
        
        Returns:
            List of sets, where each set contains nodes in one component
        """
        # TODO: Implement connected components
        # Use DFS or BFS to traverse each component
        visited = set()
        res = []
        for key in self.graph.keys():
            if key in visited: continue
            local_visited = set()
            local_visited.add(key)
            visited.add(key)
            q = deque()
            q.append(key)
            while q:
                cur = q.popleft()
                neighbors = self.get_neighbors(cur)
                for n in neighbors:
                    if n in visited: continue
                    visited.add(n)
                    local_visited.add(n)
                    q.append(n)
            res.append(local_visited)
        return res
    
    def memory_usage(self):
        """
        Estimate memory usage of the graph.
        
        Returns:
            Approximate memory usage in bytes
        """
        # TODO: Implement memory estimation
        # Consider: nodes, edges, overhead
        pass


def run_tests():
    """
    Comprehensive test suite for EfficientGraph implementation.
    Tests all functionality including edge cases.
    """
    print("="*70)
    print("Comprehensive Test Suite for EfficientGraph")
    print("="*70)
    
    passed = 0
    failed = 0
    
    def assert_test(condition, test_name):
        nonlocal passed, failed
        if condition:
            print(f"✓ {test_name}")
            passed += 1
        else:
            print(f"✗ {test_name}")
            failed += 1
    
    # Test 1: Basic Graph Construction
    print("\n" + "="*70)
    print("Test: Basic Graph Construction")
    print("="*70)
    try:
        g = EfficientGraph()
        assert_test(g.graph is not None, "Graph initialized")
        assert_test(not g.weighted, "Default graph is unweighted")
        assert_test(len(g.graph) == 0, "Empty graph has no nodes")
    except Exception as e:
        print(f"✗ Graph construction failed: {e}")
        failed += 1
    
    # Test 2: Edge Addition (Unweighted)
    print("\n" + "="*70)
    print("Test: Edge Addition (Unweighted)")
    print("="*70)
    try:
        g = EfficientGraph()
        g.add_edge(1, 2)
        assert_test(2 in g.graph[1], "Edge 1->2 added")
        assert_test(1 in g.graph[2], "Edge 2->1 added (undirected)")
        assert_test(len(g.graph[1]) == 1, "Node 1 has one neighbor")
        assert_test(len(g.graph[2]) == 1, "Node 2 has one neighbor")
        
        g.add_edge(1, 3)
        assert_test(3 in g.graph[1], "Edge 1->3 added")
        assert_test(len(g.graph[1]) == 2, "Node 1 has two neighbors")
        
        # Test duplicate edge (should be handled by set)
        g.add_edge(1, 2)
        assert_test(len(g.graph[1]) == 2, "Duplicate edge doesn't create extra entries")
    except Exception as e:
        print(f"✗ Edge addition failed: {e}")
        failed += 1
    
    # Test 3: Edge Addition (Weighted)
    print("\n" + "="*70)
    print("Test: Edge Addition (Weighted)")
    print("="*70)
    try:
        g = EfficientGraph()
        g.weighted = True
        g.add_edge(1, 2, weight=5)
        assert_test((2, 5) in g.graph[1], "Weighted edge 1->2 added")
        assert_test((1, 5) in g.graph[2], "Weighted edge 2->1 added")
        
        g.add_edge(1, 3, weight=10)
        assert_test((3, 10) in g.graph[1], "Weighted edge 1->3 added")
    except Exception as e:
        print(f"✗ Weighted edge addition failed: {e}")
        failed += 1
    
    # Test 4: Neighbor Retrieval
    print("\n" + "="*70)
    print("Test: Neighbor Retrieval")
    print("="*70)
    try:
        g = EfficientGraph()
        g.add_edge(1, 2)
        g.add_edge(1, 3)
        g.add_edge(1, 4)
        
        neighbors = g.get_neighbors(1)
        assert_test(set(neighbors) == {2, 3, 4}, "Correct neighbors retrieved")
        assert_test(len(neighbors) == 3, "Correct number of neighbors")
        
        neighbors = g.get_neighbors(5)
        assert_test(len(neighbors) == 0, "Non-existent node returns empty neighbors")
        
        # Test weighted neighbors
        g.weighted = True
        g.graph.clear()
        g.add_edge(1, 2, weight=5)
        g.add_edge(1, 3, weight=10)
        neighbors = g.get_neighbors(1)
        assert_test(set(neighbors) == {2, 3}, "Weighted graph neighbors retrieved correctly")
    except Exception as e:
        print(f"✗ Neighbor retrieval failed: {e}")
        failed += 1
    
    # Test 5: Shortest Path (Unweighted - BFS)
    print("\n" + "="*70)
    print("Test: Shortest Path (Unweighted - BFS)")
    print("="*70)
    try:
        g = EfficientGraph()
        # Linear path: 1-2-3-4
        edges = [(1, 2), (2, 3), (3, 4)]
        for u, v in edges:
            g.add_edge(u, v)
        
        path = g.shortest_path(1, 4)
        assert_test(path == [1, 2, 3, 4], "Linear path found correctly")
        
        # Branching path
        g.add_edge(2, 5)
        g.add_edge(5, 4)
        path = g.shortest_path(1, 4)
        assert_test(path == [1, 2, 3, 4], "Shortest path chosen (not longer branch)")
        
        # Same start and end
        path = g.shortest_path(1, 1)
        assert_test(path == [1], "Path from node to itself")
        
        # No path exists
        g2 = EfficientGraph()
        g2.add_edge(1, 2)
        g2.add_edge(3, 4)
        path = g2.shortest_path(1, 4)
        assert_test(path is None, "No path returns None")
        
        # Single edge
        g3 = EfficientGraph()
        g3.add_edge(1, 2)
        path = g3.shortest_path(1, 2)
        assert_test(path == [1, 2], "Single edge path")
    except Exception as e:
        print(f"✗ Shortest path (unweighted) failed: {e}")
        failed += 1
        import traceback
        traceback.print_exc()
    
    # Test 6: Shortest Path (Weighted - Dijkstra)
    print("\n" + "="*70)
    print("Test: Shortest Path (Weighted - Dijkstra)")
    print("="*70)
    try:
        g = EfficientGraph()
        g.weighted = True
        # Create graph: 1-2(1)-3(1)-4 vs 1-5(10)-4
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=1)
        g.add_edge(3, 4, weight=1)
        g.add_edge(1, 5, weight=10)
        g.add_edge(5, 4, weight=10)
        
        path = g.shortest_path(1, 4)
        assert_test(path == [1, 2, 3, 4], "Dijkstra finds shortest weighted path")
        
        # Test with different weights
        g2 = EfficientGraph()
        g2.weighted = True
        g2.add_edge(1, 2, weight=10)
        g2.add_edge(1, 3, weight=1)
        g2.add_edge(3, 2, weight=1)
        path = g2.shortest_path(1, 2)
        assert_test(path == [1, 3, 2], "Dijkstra finds path with lower total weight")
        
        # Same start and end
        path = g.shortest_path(1, 1)
        assert_test(path == [1], "Weighted: path from node to itself")
    except Exception as e:
        print(f"✗ Shortest path (weighted) failed: {e}")
        failed += 1
        import traceback
        traceback.print_exc()
    
    # Test 7: Connected Components
    print("\n" + "="*70)
    print("Test: Connected Components")
    print("="*70)
    try:
        g = EfficientGraph()
        # Two separate components: {1,2,3,4} and {5,6,7}
        edges = [(1, 2), (2, 3), (3, 4), (5, 6), (6, 7)]
        for u, v in edges:
            g.add_edge(u, v)
        
        components = g.connected_components()
        component_sets = [set(comp) for comp in components]
        assert_test(len(components) == 2, "Two connected components found")
        assert_test({1, 2, 3, 4} in component_sets, "First component correct")
        assert_test({5, 6, 7} in component_sets, "Second component correct")
        
        # Single component
        g2 = EfficientGraph()
        g2.add_edge(1, 2)
        g2.add_edge(2, 3)
        components = g2.connected_components()
        assert_test(len(components) == 1, "Single component found")
        assert_test(set(components[0]) == {1, 2, 3}, "Single component correct")
        
        # Isolated nodes
        g3 = EfficientGraph()
        g3.add_edge(1, 2)
        g3.graph[3] = set()  # Isolated node
        components = g3.connected_components()
        component_sets = [set(comp) for comp in components]
        assert_test(len(components) == 2, "Isolated node creates separate component")
        assert_test({3} in component_sets, "Isolated node in separate component")
        
        # Empty graph
        g4 = EfficientGraph()
        components = g4.connected_components()
        assert_test(len(components) == 0, "Empty graph has no components")
    except Exception as e:
        print(f"✗ Connected components failed: {e}")
        failed += 1
        import traceback
        traceback.print_exc()
    
    # Test 8: Complex Graph Scenarios
    print("\n" + "="*70)
    print("Test: Complex Graph Scenarios")
    print("="*70)
    try:
        # Star graph
        g = EfficientGraph()
        center = 1
        for i in range(2, 6):
            g.add_edge(center, i)
        path = g.shortest_path(2, 5)
        assert_test(path == [2, 1, 5], "Star graph shortest path")
        
        # Cycle
        g2 = EfficientGraph()
        edges = [(1, 2), (2, 3), (3, 4), (4, 1)]
        for u, v in edges:
            g2.add_edge(u, v)
        path = g2.shortest_path(1, 3)
        assert_test(path == [1, 2, 3] or path == [1, 4, 3], "Cycle shortest path")
        
        # Large path
        g3 = EfficientGraph()
        for i in range(1, 100):
            g3.add_edge(i, i+1)
        path = g3.shortest_path(1, 100)
        assert_test(len(path) == 100, "Long path found correctly")
        assert_test(path[0] == 1 and path[-1] == 100, "Long path start/end correct")
    except Exception as e:
        print(f"✗ Complex scenarios failed: {e}")
        failed += 1
    
    # Test 9: Edge Cases
    print("\n" + "="*70)
    print("Test: Edge Cases")
    print("="*70)
    try:
        # Graph with self-loop (if supported)
        g = EfficientGraph()
        g.add_edge(1, 1)
        neighbors = g.get_neighbors(1)
        assert_test(1 in neighbors, "Self-loop handled")
        
        # String nodes
        g2 = EfficientGraph()
        g2.add_edge('a', 'b')
        g2.add_edge('b', 'c')
        path = g2.shortest_path('a', 'c')
        assert_test(path == ['a', 'b', 'c'], "String nodes work correctly")
        
        # Negative weights (for weighted graphs)
        g3 = EfficientGraph()
        g3.weighted = True
        g3.add_edge(1, 2, weight=-1)
        # Note: Dijkstra doesn't handle negative weights, but test that it doesn't crash
        try:
            path = g3.shortest_path(1, 2)
            assert_test(path is not None, "Negative weights don't crash")
        except:
            pass  # Expected if negative weights not supported
    except Exception as e:
        print(f"✗ Edge cases failed: {e}")
        failed += 1
    
    # Test 10: Memory Usage
    print("\n" + "="*70)
    print("Test: Memory Usage")
    print("="*70)
    try:
        g = EfficientGraph()
        g.add_edge(1, 2)
        memory = g.memory_usage()
        if memory is not None:
            assert_test(isinstance(memory, (int, float)), "Memory usage returns numeric value")
            assert_test(memory >= 0, "Memory usage is non-negative")
        else:
            print("⚠ Memory usage not implemented (skipping)")
    except Exception as e:
        print(f"⚠ Memory usage test: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Final Summary")
    print("="*70)
    print(f"Total Passed: {passed}")
    print(f"Total Failed: {failed}")
    if failed == 0:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Review the implementation.")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    # Run comprehensive tests
    run_tests()
    
    # Also run the original simple example
    print("\n" + "="*70)
    print("Simple Example")
    print("="*70)
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

