"""
Exercise 15: Triton Basics

Objective: Write your first Triton kernel for vector addition.

Tasks:
1. Install Triton (if not already installed)
2. Write vector addition kernel in Triton
3. Compare performance with PyTorch
"""

try:
    import triton
    import triton.language as tl
except ImportError:
    print("Triton not installed. Install with: pip install triton")
    print("Note: Triton requires CUDA and may not work on all systems")
    exit(1)

import torch
import numpy as np


@triton.jit
def vector_add_kernel(
    x_ptr,  # Pointer to first input vector
    y_ptr,  # Pointer to second input vector
    output_ptr,  # Pointer to output vector
    n_elements,  # Number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size (compile-time constant)
):
    """
    Triton kernel for vector addition.
    
    TODO: Implement the kernel
    Hint: Use tl.program_id(0) to get block ID
    Hint: Use tl.arange(0, BLOCK_SIZE) to get thread indices
    Hint: Use tl.load() and tl.store() for memory operations
    """
    # Get the program ID for this block
    # pid = tl.program_id(0)
    
    # Calculate the block's starting position
    # block_start = pid * BLOCK_SIZE
    
    # Create a range of offsets for this block
    # offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle out-of-bounds access
    # mask = offsets < n_elements
    
    # Load data
    # x = tl.load(x_ptr + offsets, mask=mask)
    # y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform addition
    # output = x + y
    
    # Store result
    # tl.store(output_ptr + offsets, output, mask=mask)
    pass


def vector_add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper function to call Triton kernel.
    
    Args:
        x: First input vector
        y: Second input vector
    
    Returns:
        Sum of x and y
    """
    # Ensure inputs are on GPU and contiguous
    x = x.contiguous().cuda()
    y = y.contiguous().cuda()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine block size
    n_elements = output.numel()
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    vector_add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 15: Triton Basics")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Triton requires CUDA.")
        return
    
    # Create test data
    size = 1 << 20  # 1M elements
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    # PyTorch reference
    print("\n1. PyTorch reference implementation...")
    import time
    torch.cuda.synchronize()
    start = time.time()
    z_ref = x + y
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    print(f"   Time: {pytorch_time * 1000:.3f} ms")
    
    # TODO: Implement and call Triton kernel
    print("\n2. Triton implementation...")
    # z_triton = vector_add_triton(x, y)
    # torch.cuda.synchronize()
    # start = time.time()
    # z_triton = vector_add_triton(x, y)
    # torch.cuda.synchronize()
    # triton_time = time.time() - start
    # print(f"   Time: {triton_time * 1000:.3f} ms")
    # print(f"   Speedup: {pytorch_time / triton_time:.2f}x")
    
    # Verify correctness
    # if torch.allclose(z_ref, z_triton):
    #     print("   Result: PASSED")
    # else:
    #     print("   Result: FAILED")
    
    print("\n" + "=" * 60)
    print("Exercise complete! Implement the TODO above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

