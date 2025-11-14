"""
Exercise 2: FX Graph Manipulation

Objective: Learn to extract and manipulate FX graphs from PyTorch models.

Tasks:
1. Extract FX graph from a model
2. Visualize the graph
3. Write a custom FX pass to transform the graph
4. Apply the transformation
"""

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
from torch.fx.node import Node
from typing import Any


class SimpleConvModel(nn.Module):
    """Simple convolutional model for FX graph manipulation"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def print_graph(graph_module: GraphModule, title: str = "Graph"):
    """Print the FX graph structure"""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(graph_module.graph)
    print(f"{'=' * 60}\n")


def fuse_bn_relu_pass(graph_module: GraphModule) -> GraphModule:
    """
    Custom FX pass to fuse BatchNorm + ReLU operations.
    
    This is a simplified example. In practice, you'd need to handle
    the actual fusion logic more carefully.
    
    Args:
        graph_module: Input GraphModule
    
    Returns:
        Transformed GraphModule
    """
    graph = graph_module.graph
    
    # TODO: Implement graph transformation
    # 1. Find patterns: BatchNorm -> ReLU
    # 2. Replace with fused operation (if available) or keep as-is
    # 3. Update graph
    
    # Hint: Use graph.nodes() to iterate through nodes
    # Hint: Use graph.inserting_after() or graph.inserting_before() to modify
    
    # For now, just return the original graph
    # In a real implementation, you would:
    # - Find BN -> ReLU patterns
    # - Replace with fused BNReLU (if available in PyTorch)
    # - Or mark for later fusion
    
    return graph_module


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 2: FX Graph Manipulation")
    print("=" * 60)
    
    # Create model
    model = SimpleConvModel()
    model.eval()
    
    # TODO: Extract FX graph using symbolic_trace
    # Hint: Use torch.fx.symbolic_trace(model)
    
    print("\n1. Extracting FX graph...")
    # graph_module = symbolic_trace(model)
    # print_graph(graph_module, "Original Graph")
    
    # TODO: Visualize the graph
    # You can print it or use graph visualization tools
    
    # TODO: Write and apply custom FX pass
    print("\n2. Applying custom FX pass (BN+ReLU fusion)...")
    # transformed_module = fuse_bn_relu_pass(graph_module)
    # print_graph(transformed_module, "Transformed Graph")
    
    # TODO: Verify the transformation works
    # Test with sample input
    
    print("\n" + "=" * 60)
    print("Exercise complete! Implement the TODOs above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

