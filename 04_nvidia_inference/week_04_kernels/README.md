# Week 4: High-Performance Kernels

## Overview

Master high-performance kernel development using Triton, CUTLASS, and efficient attention mechanisms. Learn to write optimized kernels for ML workloads without low-level CUDA programming.

## Learning Objectives

By the end of this week, you should be able to:
- Write kernels in Triton DSL
- Understand CUTLASS for linear algebra
- Implement Flash Attention
- Optimize KV-cache operations
- Profile kernel performance vs cuBLAS
- Understand memory bandwidth bottlenecks

## Day-by-Day Schedule

### Day 1: Triton Fundamentals (4 hours)

**Morning (2 hours):**
- Study Triton architecture and philosophy
- Learn Triton DSL syntax and programming model
- Understand block-level programming

**Afternoon (2 hours):**
- Complete practice exercise: Simple Triton kernel
- Implement vector addition in Triton
- Compare Triton vs CUDA performance

**Key Concepts:**
- Triton: Python-like DSL for GPU kernels
- Block-level: Program at block granularity
- Automatic optimization: Memory coalescing, etc.

### Day 2: Triton Advanced (4 hours)

**Morning (2 hours):**
- Study advanced Triton features: tile operations, reductions
- Learn about automatic memory management
- Understand Triton compilation pipeline

**Afternoon (2 hours):**
- Complete practice exercise: Matrix multiplication in Triton
- Optimize GEMM operations
- Profile vs cuBLAS performance

**Key Concepts:**
- Tile Operations: Block-level matrix operations
- Reductions: Sum, max, etc. across blocks
- Compilation: Triton → LLVM → PTX

### Day 3: CUTLASS (4 hours)

**Morning (2 hours):**
- Study CUTLASS architecture: templates for linear algebra
- Learn GEMM implementations in CUTLASS
- Understand kernel selection and tuning

**Afternoon (2 hours):**
- Complete practice exercise: CUTLASS GEMM
- Compare different CUTLASS kernels
- Profile performance characteristics

**Key Concepts:**
- CUTLASS: CUDA Templates for Linear Algebra
- GEMM: General Matrix Multiply
- Kernel Selection: Choosing optimal kernel for problem size

### Day 4: Flash Attention (4 hours)

**Morning (2 hours):**
- Study Flash Attention algorithm
- Learn about memory-efficient attention computation
- Understand tiling and online softmax

**Afternoon (2 hours):**
- Complete practice exercise: Flash Attention implementation
- Compare vs standard attention
- Profile memory and compute efficiency

**Key Concepts:**
- Flash Attention: Memory-efficient attention
- Tiling: Block-wise computation
- Online Softmax: Numerically stable softmax

### Day 5: KV-Cache Optimization (3 hours)

**Morning (2 hours):**
- Study KV-cache in autoregressive generation
- Learn about cache-efficient attention
- Understand PagedAttention and similar techniques

**Afternoon (1 hour):**
- Complete practice exercise: KV-cache optimization
- Profile cache efficiency
- Review all concepts

**Key Concepts:**
- KV-Cache: Caching key-value pairs for efficiency
- Autoregressive: Generating tokens one at a time
- Cache Efficiency: Minimizing memory access

## Practice Exercises

See `practice_exercises/` directory for:
1. `ex_15_triton_basics.py` - Basic Triton kernel
2. `ex_16_triton_gemm.py` - Matrix multiplication in Triton
3. `ex_17_cutlass_gemm.py` - CUTLASS GEMM
4. `ex_18_flash_attention.py` - Flash Attention implementation
5. `ex_19_kv_cache.py` - KV-cache optimization

## Key Resources

### Documentation
- [Triton Documentation](https://triton-lang.org/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

### Papers
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al.)
- "Efficiently Scaling Transformer Inference" (Pope et al.)

### Tutorials
- Triton Tutorials
- CUTLASS Examples

## Common Interview Questions

1. "What is Triton and why use it over CUDA?"
2. "How does Flash Attention reduce memory usage?"
3. "Explain KV-cache optimization for autoregressive generation"
4. "How do you choose between Triton, CUTLASS, and cuBLAS?"
5. "What are memory bandwidth bottlenecks and how do you address them?"

## Success Checklist

- [ ] Can write Triton kernels
- [ ] Understand CUTLASS usage
- [ ] Can implement Flash Attention
- [ ] Can optimize KV-cache
- [ ] Can profile kernel performance
- [ ] Understand memory bandwidth bottlenecks
- [ ] Have prepared a technical deep dive example

## Next Week Preview

Next week focuses on **TensorRT & TRT-LLM**. Understanding high-performance kernels from this week will help you understand how TensorRT optimizes models and selects kernels.

