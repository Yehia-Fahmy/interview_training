"""
Exercise 25: Tensor Parallelism

Objective: Implement tensor parallelism for transformer layers.

Tasks:
1. Understand column and row parallelism
2. Implement tensor parallelism for linear layers
3. Implement tensor parallelism for attention
4. Profile communication overhead
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer.
    
    Splits weight matrix along column dimension.
    Each GPU holds a subset of output features.
    """
    
    def __init__(self, input_size: int, output_size: int, world_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        
        # Each GPU gets output_size / world_size columns
        self.local_output_size = output_size // world_size
        
        # Local weight matrix
        self.weight = nn.Parameter(torch.randn(self.local_output_size, input_size))
        self.bias = nn.Parameter(torch.randn(self.local_output_size))
    
    def forward(self, x):
        """
        Forward pass with column parallelism.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
        
        Returns:
            Output tensor [batch, seq_len, local_output_size]
        """
        # Local computation
        output = torch.matmul(x, self.weight.t()) + self.bias
        
        # TODO: Gather results from all GPUs
        # Use dist.all_gather() to collect outputs from all GPUs
        # Then concatenate along the feature dimension
        
        return output


class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer.
    
    Splits weight matrix along row dimension.
    Each GPU holds a subset of input features.
    """
    
    def __init__(self, input_size: int, output_size: int, world_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        
        # Each GPU gets input_size / world_size rows
        self.local_input_size = input_size // world_size
        
        # Local weight matrix
        self.weight = nn.Parameter(torch.randn(output_size, self.local_input_size))
        self.bias = nn.Parameter(torch.randn(output_size)) if dist.get_rank() == 0 else None
    
    def forward(self, x):
        """
        Forward pass with row parallelism.
        
        Args:
            x: Input tensor [batch, seq_len, local_input_size]
        
        Returns:
            Output tensor [batch, seq_len, output_size]
        """
        # Local computation
        output = torch.matmul(x, self.weight.t())
        
        # TODO: All-reduce to sum partial results
        # Use dist.all_reduce() to sum outputs from all GPUs
        # Add bias only on rank 0 (or broadcast)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


def tensor_parallel_attention(Q, K, V, world_size, rank):
    """
    Tensor-parallel attention implementation.
    
    This is a simplified version. Real implementations handle
    communication more carefully.
    
    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim]
        K: Key tensor [batch, heads, seq_len, head_dim]
        V: Value tensor [batch, heads, seq_len, head_dim]
        world_size: Number of GPUs
        rank: Current GPU rank
    
    Returns:
        Attention output
    """
    # TODO: Implement tensor-parallel attention
    # 1. Split Q, K, V across heads or head_dim
    # 2. Compute local attention
    # 3. All-reduce or all-gather results
    
    # Simplified: standard attention for now
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    
    return output


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 25: Tensor Parallelism")
    print("=" * 60)
    
    # Check if distributed is available
    if not dist.is_available():
        print("PyTorch distributed not available.")
        print("This exercise requires multiple GPUs or MPI.")
        return
    
    # TODO: Initialize distributed environment
    # dist.init_process_group(backend='nccl')
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    
    # For single GPU testing, simulate parallelism
    print("\nNote: This exercise is designed for multi-GPU setup.")
    print("For single GPU, you can simulate by setting world_size=1")
    
    world_size = 1  # Change to actual world_size in distributed setup
    rank = 0  # Change to actual rank in distributed setup
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Test column-parallel linear
    print("\n1. Testing Column-Parallel Linear...")
    input_size = 1024
    output_size = 2048
    
    col_linear = ColumnParallelLinear(input_size, output_size, world_size).to(device)
    x = torch.randn(2, 128, input_size).to(device)
    
    # Forward pass
    output = col_linear(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected local output size: {output_size // world_size}")
    
    # Test row-parallel linear
    print("\n2. Testing Row-Parallel Linear...")
    row_linear = RowParallelLinear(input_size, output_size, world_size).to(device)
    x_local = torch.randn(2, 128, input_size // world_size).to(device)
    
    output = row_linear(x_local)
    print(f"   Input shape: {x_local.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n" + "=" * 60)
    print("Exercise complete!")
    print("=" * 60)
    print("\nNote: Full implementation requires:")
    print("  - Multi-GPU setup")
    print("  - Proper communication (all-gather, all-reduce)")
    print("  - Handling edge cases (uneven splits, etc.)")


if __name__ == "__main__":
    import math
    main()

