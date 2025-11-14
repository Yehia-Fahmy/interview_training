# Week 6: Distributed Inference

## Overview

Learn distributed inference strategies for large models. Master tensor parallelism, sequence parallelism, pipeline parallelism, and communication optimization for multi-GPU deployment.

## Learning Objectives

By the end of this week, you should be able to:
- Implement tensor parallelism (column/row parallelism)
- Understand sequence parallelism
- Design model sharding strategies
- Optimize communication (NCCL, all-reduce)
- Balance load across GPUs
- Profile communication overhead

## Day-by-Day Schedule

### Day 1: Tensor Parallelism Fundamentals (4 hours)

**Morning (2 hours):**
- Study tensor parallelism: column and row parallelism
- Learn how to shard transformer layers
- Understand communication patterns (all-reduce, all-gather)

**Afternoon (2 hours):**
- Complete practice exercise: Implement tensor parallelism for linear layer
- Shard attention and MLP layers
- Profile communication overhead

**Key Concepts:**
- Column Parallelism: Sharding along column dimension
- Row Parallelism: Sharding along row dimension
- All-Reduce: Communication primitive for tensor parallelism

### Day 2: Sequence Parallelism (4 hours)

**Morning (2 hours):**
- Study sequence parallelism for long sequences
- Learn about sequence-level sharding
- Understand communication patterns

**Afternoon (2 hours):**
- Complete practice exercise: Implement sequence parallelism
- Optimize for long sequence lengths
- Compare vs tensor parallelism

**Key Concepts:**
- Sequence Parallelism: Sharding along sequence dimension
- Long Sequences: Handling very long inputs
- Communication: Ring attention, etc.

### Day 3: Pipeline Parallelism (4 hours)

**Morning (2 hours):**
- Study pipeline parallelism: model stages across GPUs
- Learn about micro-batching
- Understand pipeline bubbles and efficiency

**Afternoon (2 hours):**
- Complete practice exercise: Implement pipeline parallelism
- Optimize micro-batch size
- Profile pipeline efficiency

**Key Concepts:**
- Pipeline Parallelism: Stages across GPUs
- Micro-batching: Small batches for pipeline
- Pipeline Bubbles: Idle time in pipeline

### Day 4: Model Sharding Strategies (4 hours)

**Morning (2 hours):**
- Study different sharding strategies
- Learn about 3D parallelism (tensor + pipeline + data)
- Understand when to use each strategy

**Afternoon (2 hours):**
- Complete practice exercise: Design sharding for large model
- Compare different strategies
- Profile end-to-end performance

**Key Concepts:**
- 3D Parallelism: Combining tensor, pipeline, data parallelism
- Sharding Strategy: How to partition model
- Model Size: Choosing strategy based on model size

### Day 5: Communication Optimization (3 hours)

**Morning (2 hours):**
- Study NCCL (NVIDIA Collective Communications Library)
- Learn about communication optimization
- Understand overlap of computation and communication

**Afternoon (1 hour):**
- Complete practice exercise: Optimize communication
- Review all concepts
- Prepare technical deep dive example

**Key Concepts:**
- NCCL: Multi-GPU communication library
- Communication Overlap: Hiding communication latency
- All-Reduce: Efficient collective operation

## Practice Exercises

See `practice_exercises/` directory for:
1. `ex_25_tensor_parallelism.py` - Tensor parallelism implementation
2. `ex_26_sequence_parallelism.py` - Sequence parallelism
3. `ex_27_pipeline_parallelism.py` - Pipeline parallelism
4. `ex_28_sharding_strategies.py` - Model sharding strategies
5. `ex_29_communication_optimization.py` - Communication optimization

## Key Resources

### Documentation
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Tensor parallelism reference
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Papers
- "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al.)
- "Sequence Parallelism" (Korthikanti et al.)

### Tutorials
- Megatron-LM Tutorials
- PyTorch Distributed Tutorials

## Common Interview Questions

1. "What's the difference between tensor and pipeline parallelism?"
2. "How do you implement tensor parallelism for transformers?"
3. "When would you use sequence parallelism?"
4. "How do you optimize communication in distributed inference?"
5. "Design a sharding strategy for a 70B parameter model"
6. "How do you balance load across GPUs?"

## Success Checklist

- [ ] Can implement tensor parallelism
- [ ] Understand sequence parallelism
- [ ] Can design sharding strategies
- [ ] Can optimize communication
- [ ] Can profile communication overhead
- [ ] Understand load balancing
- [ ] Have prepared a technical deep dive example

## Next Week Preview

Next week focuses on **Performance Profiling & Optimization**. Understanding distributed inference will help you identify bottlenecks in multi-GPU systems.

