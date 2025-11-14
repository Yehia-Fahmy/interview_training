# Week 2: Model Optimization Techniques

## Overview

Learn fundamental model optimization techniques including quantization, pruning, and sparsity. Understand trade-offs between accuracy and inference speed, and how these techniques interact with hardware.

## Learning Objectives

By the end of this week, you should be able to:
- Implement quantization (INT8, INT4, FP16/BF16)
- Apply pruning techniques (structured/unstructured)
- Understand sparsity patterns (2:4, activation sparsity)
- Profile accuracy vs speed trade-offs
- Understand hardware constraints (Tensor Core utilization)

## Day-by-Day Schedule

### Day 1: Quantization Fundamentals (4 hours)

**Morning (2 hours):**
- Study quantization theory: uniform vs non-uniform quantization
- Learn about quantization-aware training (QAT) vs post-training quantization (PTQ)
- Understand calibration methods

**Afternoon (2 hours):**
- Complete practice exercise: INT8 quantization
- Implement calibration for post-training quantization
- Profile accuracy vs speed improvements

**Key Concepts:**
- Quantization: Reducing precision (FP32 → INT8/INT4)
- Calibration: Finding optimal quantization parameters
- QAT vs PTQ: Training-time vs inference-time quantization

### Day 2: Advanced Quantization (4 hours)

**Morning (2 hours):**
- Study mixed precision (FP16/BF16)
- Learn about INT4 quantization and grouping
- Understand per-channel vs per-tensor quantization

**Afternoon (2 hours):**
- Complete practice exercise: Mixed precision training
- Implement INT4 quantization with grouping
- Compare different quantization strategies

**Key Concepts:**
- Mixed Precision: Different precisions for different layers
- Group Quantization: Grouping weights for INT4
- Per-channel: Different scales per channel

### Day 3: Pruning Techniques (4 hours)

**Morning (2 hours):**
- Study pruning theory: structured vs unstructured
- Learn magnitude-based and gradient-based pruning
- Understand iterative pruning strategies

**Afternoon (2 hours):**
- Complete practice exercise: Structured pruning
- Implement magnitude-based pruning
- Profile sparsity vs accuracy trade-offs

**Key Concepts:**
- Structured Pruning: Removing entire structures (channels, filters)
- Unstructured Pruning: Removing individual weights
- Magnitude-based: Remove smallest weights
- Gradient-based: Use gradients to guide pruning

### Day 4: Sparsity Patterns (4 hours)

**Morning (2 hours):**
- Study structured sparsity (2:4 patterns)
- Learn about activation sparsity
- Understand hardware acceleration for sparse operations

**Afternoon (2 hours):**
- Complete practice exercise: 2:4 sparsity pattern
- Implement activation sparsity
- Profile sparse vs dense performance

**Key Concepts:**
- 2:4 Sparsity: 2 non-zero values per 4 consecutive weights
- Activation Sparsity: Sparse activations (ReLU, etc.)
- Hardware Acceleration: Tensor Core support for sparsity

### Day 5: Neural Architecture Search & Review (3 hours)

**Morning (2 hours):**
- Study NAS basics: search spaces, search strategies
- Learn about efficient architectures (MobileNet, EfficientNet)
- Understand architecture-aware optimization

**Afternoon (1 hour):**
- Review all optimization techniques
- Complete final practice: Combined optimization pipeline
- Prepare technical deep dive example

**Key Concepts:**
- NAS: Automatically finding efficient architectures
- Search Space: Set of possible architectures
- Search Strategy: How to explore the space

## Practice Exercises

See `practice_exercises/` directory for:
1. `ex_05_int8_quantization.py` - Post-training INT8 quantization
2. `ex_06_mixed_precision.py` - FP16/BF16 mixed precision
3. `ex_07_pruning.py` - Structured and unstructured pruning
4. `ex_08_sparsity.py` - 2:4 sparsity patterns
5. `ex_09_combined_optimization.py` - Combined optimization pipeline

## Key Resources

### Documentation
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [TensorRT Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#quantization)
- [PyTorch Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

### Papers
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al.)
- "The Lottery Ticket Hypothesis" (Frankle & Carbin)
- "2:4 Sparsity Pattern" (NVIDIA)

### Tutorials
- PyTorch Quantization Tutorial
- TensorRT Quantization Guide

## Common Interview Questions

1. "What's the difference between QAT and PTQ?"
2. "How would you implement INT8 quantization?"
3. "Explain structured vs unstructured pruning"
4. "What is 2:4 sparsity and why is it useful?"
5. "How do you balance accuracy vs speed in optimization?"
6. "What are the hardware constraints for quantization?"

## Success Checklist

- [ ] Can implement INT8 quantization
- [ ] Can apply pruning techniques
- [ ] Understand sparsity patterns
- [ ] Can profile accuracy vs speed trade-offs
- [ ] Understand hardware constraints
- [ ] Have prepared a technical deep dive example

## Next Week Preview

Next week focuses on **GPU Architecture & CUDA Programming**. Understanding hardware constraints from this week will help you write efficient CUDA kernels.

