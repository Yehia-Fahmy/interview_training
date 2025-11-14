"""
Exercise 1: torch.compile Basics

Objective: Learn basic usage of torch.compile and understand compilation overhead vs runtime benefits.

Tasks:
1. Create a simple neural network
2. Compile it with torch.compile
3. Profile compilation time vs runtime speedup
4. Experiment with different backends
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List


class SimpleModel(nn.Module):
    """Simple feedforward network for testing torch.compile"""
    
    def __init__(self, input_size: int = 1024, hidden_size: int = 2048, output_size: int = 512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
    
    def forward(self, x):
        return self.layers(x)


def profile_model(model: nn.Module, input_tensor: torch.Tensor, num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
    """
    Profile model inference time.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor
        num_warmup: Number of warmup runs
        num_runs: Number of runs to average
    
    Returns:
        Dictionary with timing statistics
    """
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Synchronize GPU if available
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Profile
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return {
        'avg_time_ms': avg_time * 1000,
        'total_time_s': end_time - start_time,
    }


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 1: torch.compile Basics")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and input
    model = SimpleModel().to(device)
    input_tensor = torch.randn(32, 1024).to(device)
    
    print("\n1. Profiling uncompiled model...")
    uncompiled_stats = profile_model(model, input_tensor)
    print(f"   Average inference time: {uncompiled_stats['avg_time_ms']:.3f} ms")
    
    # TODO: Compile the model with torch.compile
    # Hint: Use torch.compile(model) and experiment with different backends
    # Backends to try: 'inductor', 'aot_eager', 'cudagraphs'
    
    # TODO: Profile compiled model
    # Compare compilation overhead vs runtime speedup
    
    # TODO: Experiment with different backends
    # Try: 'inductor', 'aot_eager', 'cudagraphs'
    
    print("\n2. Profiling compiled model...")
    # compiled_stats = profile_model(compiled_model, input_tensor)
    # print(f"   Average inference time: {compiled_stats['avg_time_ms']:.3f} ms")
    # print(f"   Speedup: {uncompiled_stats['avg_time_ms'] / compiled_stats['avg_time_ms']:.2f}x")
    
    print("\n" + "=" * 60)
    print("Exercise complete! Implement the TODOs above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

