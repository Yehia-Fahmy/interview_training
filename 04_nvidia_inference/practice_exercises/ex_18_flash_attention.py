"""
Exercise 18: Flash Attention Implementation

Objective: Understand and implement Flash Attention for memory-efficient attention computation.

Tasks:
1. Understand Flash Attention algorithm
2. Implement basic Flash Attention kernel (simplified)
3. Compare memory usage with standard attention
4. Profile performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def standard_attention(Q, K, V, mask=None):
    """
    Standard attention implementation.
    
    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim]
        K: Key tensor [batch, heads, seq_len, head_dim]
        V: Value tensor [batch, heads, seq_len, head_dim]
        mask: Attention mask (optional)
    
    Returns:
        Output tensor and attention weights
    """
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply to values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights


def flash_attention_simplified(Q, K, V, block_size=64):
    """
    Simplified Flash Attention implementation.
    
    This is a simplified version for educational purposes.
    Real Flash Attention uses online softmax and more optimizations.
    
    Args:
        Q: Query tensor [batch, heads, seq_len, head_dim]
        K: Key tensor [batch, heads, seq_len, head_dim]
        V: Value tensor [batch, heads, seq_len, head_dim]
        block_size: Block size for tiling
    
    Returns:
        Output tensor
    """
    batch, heads, seq_len, head_dim = Q.shape
    
    # TODO: Implement block-wise attention computation
    # This is a simplified version - real Flash Attention uses:
    # 1. Online softmax (numerically stable)
    # 2. Tiling along both sequence and head dimensions
    # 3. Recomputation to avoid storing attention matrix
    
    # For now, use standard attention as placeholder
    # In real implementation, you would:
    # - Tile Q, K, V into blocks
    # - Compute attention for each block
    # - Use online softmax to combine results
    # - Avoid storing full attention matrix
    
    output = torch.zeros_like(Q)
    
    # Placeholder: use standard attention
    # Replace with block-wise computation
    for i in range(0, seq_len, block_size):
        end_i = min(i + block_size, seq_len)
        Q_block = Q[:, :, i:end_i, :]
        
        # Compute attention for this block
        # This is simplified - real implementation tiles K and V too
        scores = torch.matmul(Q_block, K.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        output[:, :, i:end_i, :] = torch.matmul(attn, V)
    
    return output


def profile_memory_usage(func, *args):
    """Profile memory usage of a function"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        result = func(*args)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return result, peak_memory
    else:
        return func(*args), 0


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 18: Flash Attention")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test data
    batch_size = 2
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim).to(device)
    
    print(f"\nInput shape: Q/K/V = {Q.shape}")
    print(f"Memory per tensor: {Q.numel() * 4 / 1024**2:.2f} MB")
    
    # Standard attention
    print("\n1. Standard attention...")
    def std_attn():
        return standard_attention(Q, K, V)
    
    std_output, std_memory = profile_memory_usage(std_attn)
    print(f"   Peak memory: {std_memory:.2f} MB")
    
    # Flash Attention (simplified)
    print("\n2. Flash Attention (simplified)...")
    def flash_attn():
        return flash_attention_simplified(Q, K, V)
    
    flash_output, flash_memory = profile_memory_usage(flash_attn)
    print(f"   Peak memory: {flash_memory:.2f} MB")
    print(f"   Memory reduction: {(1 - flash_memory / std_memory) * 100:.1f}%")
    
    # Verify correctness (approximate)
    if torch.allclose(std_output, flash_output, atol=1e-2):
        print("   Result: PASSED (within tolerance)")
    else:
        print("   Result: Output differs (expected for simplified version)")
    
    print("\n" + "=" * 60)
    print("Exercise complete!")
    print("=" * 60)
    print("\nNote: This is a simplified implementation.")
    print("Real Flash Attention uses online softmax and more optimizations.")
    print("See the Flash Attention paper for full implementation details.")


if __name__ == "__main__":
    main()

