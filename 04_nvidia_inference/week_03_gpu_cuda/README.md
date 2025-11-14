# Week 3: GPU Architecture & CUDA Programming

## Overview

Learn GPU architecture fundamentals and CUDA programming. Understand memory hierarchy, kernel programming, and profiling tools essential for high-performance inference optimization.

## Learning Objectives

By the end of this week, you should be able to:
- Understand GPU memory hierarchy
- Write basic CUDA kernels
- Optimize memory access patterns
- Profile GPU code with `nsys` and `ncu`
- Understand occupancy and resource limits
- Utilize Tensor Core operations

## Day-by-Day Schedule

### Day 1: GPU Architecture Fundamentals (4 hours)

**Morning (2 hours):**
- Study GPU architecture: SM, CUDA cores, Tensor Cores
- Learn memory hierarchy: registers, shared memory, global memory
- Understand warp execution model

**Afternoon (2 hours):**
- Study CUDA programming model: host/device, threads/blocks/grids
- Learn CUDA memory types and their characteristics
- Understand memory coalescing

**Key Concepts:**
- Streaming Multiprocessor (SM): Execution unit
- Warp: 32 threads executing together
- Memory Hierarchy: Registers > Shared > Global
- Coalescing: Efficient memory access patterns

### Day 2: CUDA Kernel Programming (4 hours)

**Morning (2 hours):**
- Learn CUDA kernel syntax and launch configuration
- Study thread indexing and synchronization
- Understand shared memory usage

**Afternoon (2 hours):**
- Complete practice exercise: Vector addition kernel
- Implement matrix multiplication kernel
- Profile kernel performance

**Key Concepts:**
- Kernel: GPU function executed by many threads
- Launch Configuration: Blocks and threads per block
- Shared Memory: Fast on-chip memory
- Synchronization: `__syncthreads()`

### Day 3: Memory Optimization (4 hours)

**Morning (2 hours):**
- Study memory coalescing patterns
- Learn about bank conflicts in shared memory
- Understand memory access optimization

**Afternoon (2 hours):**
- Complete practice exercise: Optimize memory access
- Implement tiled matrix multiplication
- Profile memory bandwidth utilization

**Key Concepts:**
- Coalescing: Consecutive threads access consecutive memory
- Bank Conflicts: Multiple threads accessing same shared memory bank
- Tiling: Using shared memory to reduce global memory access

### Day 4: Tensor Cores & Advanced Topics (4 hours)

**Morning (2 hours):**
- Study Tensor Core architecture
- Learn WMMA (Warp Matrix Multiply Accumulate) API
- Understand GEMM and convolution on Tensor Cores

**Afternoon (2 hours):**
- Complete practice exercise: Tensor Core GEMM
- Implement convolution using Tensor Cores
- Compare Tensor Core vs CUDA core performance

**Key Concepts:**
- Tensor Cores: Specialized units for matrix operations
- WMMA API: Programming interface for Tensor Cores
- Mixed Precision: FP16 input, FP32 accumulation

### Day 5: Profiling & Optimization (3 hours)

**Morning (2 hours):**
- Learn Nsight Systems (`nsys`) for timeline profiling
- Study Nsight Compute (`ncu`) for kernel analysis
- Understand occupancy and resource limits

**Afternoon (1 hour):**
- Complete practice exercise: Profile and optimize kernel
- Identify bottlenecks (compute vs memory)
- Optimize for maximum performance

**Key Concepts:**
- Profiling: Measuring performance
- Occupancy: Active warps per SM
- Bottlenecks: Compute-bound vs memory-bound

## Practice Exercises

See `practice_exercises/` directory for:
1. `ex_10_vector_add.cu` - Basic CUDA kernel (vector addition)
2. `ex_11_matrix_multiply.cu` - Matrix multiplication kernel
3. `ex_12_shared_memory.cu` - Shared memory optimization
4. `ex_13_tensor_core_gemm.cu` - Tensor Core GEMM
5. `ex_14_profiling.py` - Profiling workflow

## Key Resources

### Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### Books
- "Programming Massively Parallel Processors" (Kirk & Hwu)
- "Professional CUDA C Programming" (Wilt)

### Tutorials
- NVIDIA CUDA Tutorials
- CUDA Samples (in CUDA toolkit)

## Common Interview Questions

1. "Explain GPU memory hierarchy"
2. "What is a warp and why is it important?"
3. "How do you optimize memory access in CUDA?"
4. "What are Tensor Cores and how do you use them?"
5. "How do you profile CUDA code?"
6. "What is occupancy and how do you optimize it?"

## Success Checklist

- [ ] Can explain GPU architecture
- [ ] Can write basic CUDA kernels
- [ ] Can optimize memory access patterns
- [ ] Can use Tensor Cores
- [ ] Can profile with nsys and ncu
- [ ] Understand occupancy and resource limits
- [ ] Have prepared a technical deep dive example

## Next Week Preview

Next week focuses on **High-Performance Kernels** using Triton, CUTLASS, and Flash Attention. CUDA knowledge from this week is essential for understanding these higher-level frameworks.

