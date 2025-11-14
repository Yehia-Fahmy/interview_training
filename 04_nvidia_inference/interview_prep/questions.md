# Common Interview Questions

## Technical Questions

### PyTorch Ecosystem

**Q: How does `torch.compile` work under the hood?**

**Sample Answer:**
`torch.compile` uses TorchDynamo to capture computation graphs from Python bytecode. The process involves:
1. **TorchDynamo**: Intercepts Python execution and captures FX graphs
2. **Graph Extraction**: Converts Python operations to FX Graph IR
3. **Backend Selection**: Chooses execution backend (Inductor, TensorRT, etc.)
4. **Compilation**: Backend compiles graph to optimized code
5. **Caching**: Compiled graphs are cached for reuse

The key innovation is capturing graphs without modifying user code, making it easy to adopt.

---

**Q: What's the difference between tracing and scripting?**

**Sample Answer:**
- **Tracing** (`torch.jit.trace`): Records execution flow by running the model with example inputs. Captures what actually happens but may miss control flow.
- **Scripting** (`torch.jit.script`): Analyzes source code to create graph. Handles control flow but requires code to be scriptable.

`torch.export` provides a middle ground with better control flow handling.

---

### Model Optimization

**Q: What's the difference between QAT and PTQ?**

**Sample Answer:**
- **QAT (Quantization-Aware Training)**: Model is trained with quantization simulation. Higher accuracy but requires retraining.
- **PTQ (Post-Training Quantization)**: Model is quantized after training. Faster but may have accuracy loss.

QAT is better for accuracy-critical applications, PTQ for quick deployment.

---

**Q: Explain structured vs unstructured pruning**

**Sample Answer:**
- **Structured Pruning**: Removes entire structures (channels, filters). Hardware-friendly, easier to accelerate.
- **Unstructured Pruning**: Removes individual weights. Higher sparsity but requires specialized hardware.

Structured pruning is preferred for standard hardware, unstructured for specialized accelerators.

---

### GPU Architecture

**Q: Explain GPU memory hierarchy**

**Sample Answer:**
GPU memory has multiple levels:
1. **Registers**: Fastest, per-thread, limited capacity
2. **Shared Memory**: Fast, shared within block, on-chip
3. **L1/L2 Cache**: Automatic caching
4. **Global Memory**: Slowest, large capacity, off-chip

Optimization involves maximizing use of faster memory levels.

---

**Q: What is a warp and why is it important?**

**Sample Answer:**
A warp is 32 threads that execute together in lockstep on an SM. Important because:
- Threads in a warp execute the same instruction
- Divergence (different execution paths) serializes execution
- Memory access can be coalesced within a warp
- Warp-level primitives enable efficient operations

Understanding warps is crucial for writing efficient CUDA kernels.

---

### Distributed Inference

**Q: What's the difference between tensor and pipeline parallelism?**

**Sample Answer:**
- **Tensor Parallelism**: Splits individual layers across GPUs. All GPUs participate in each layer. Lower latency but more communication.
- **Pipeline Parallelism**: Splits model into stages. Each GPU handles different stages. Less communication but pipeline bubbles reduce efficiency.

Tensor parallelism is better for latency-sensitive applications, pipeline for very large models.

---

**Q: How do you optimize communication in distributed inference?**

**Sample Answer:**
- Use efficient collectives (all-reduce, all-gather)
- Overlap computation and communication
- Use gradient compression for training
- Choose optimal parallelism strategy
- Minimize synchronization points
- Use high-bandwidth interconnects (NVLink, InfiniBand)

---

### Performance Profiling

**Q: How do you identify if a kernel is compute-bound or memory-bound?**

**Sample Answer:**
- **Compute-bound**: High arithmetic intensity, low memory bandwidth utilization. Optimize by reducing operations or using Tensor Cores.
- **Memory-bound**: Low arithmetic intensity, high memory bandwidth utilization. Optimize by improving memory access patterns, using shared memory, coalescing.

Use roofline model: if performance is below compute roof, compute-bound; below memory roof, memory-bound.

---

## System Design Questions

### Q: Design an automated model optimization platform

**Approach:**
1. **Clarify Requirements**
   - Scale (models, users, requests)
   - Optimization techniques needed
   - Latency/throughput requirements
   - Integration requirements

2. **High-Level Architecture**
   - Model Registry
   - Optimization Pipeline
   - Inference Engine
   - API Gateway
   - Monitoring

3. **Key Components**
   - Model upload and storage
   - Graph extraction (PyTorch 2.0)
   - Optimization passes (quantization, pruning)
   - Model compilation (TensorRT)
   - Deployment and serving
   - Versioning and rollback

4. **Scalability**
   - Horizontal scaling for optimization workers
   - Distributed inference serving
   - Caching optimized models
   - Load balancing

5. **Reliability**
   - Redundancy
   - Health checks
   - Rollback mechanisms
   - Monitoring and alerting

---

### Q: How would you scale inference to handle 1000s of models?

**Approach:**
1. **Model Management**
   - Efficient storage (object storage)
   - Lazy loading
   - Model registry with metadata

2. **Resource Management**
   - GPU pooling
   - Dynamic allocation
   - Model prioritization

3. **Serving Strategy**
   - Multi-tenant serving
   - Request routing
   - Batching strategies
   - Caching frequently used models

4. **Scaling**
   - Horizontal scaling
   - Auto-scaling based on load
   - Regional deployment

---

## Behavioral Questions

### Q: Tell me about a time you optimized GPU performance

**Structure:**
1. **Situation**: Context and problem
2. **Task**: What you needed to achieve
3. **Action**: Steps you took (profiling, optimization, testing)
4. **Result**: Metrics and improvements

**Example:**
"I optimized a transformer inference kernel. Used Nsight Compute to identify memory bottlenecks, implemented shared memory tiling, and achieved 3x speedup."

---

### Q: How do you stay current with ML optimization techniques?

**Answer:**
- Follow research papers (arXiv, conferences)
- Contribute to open-source (PyTorch, HuggingFace)
- Attend conferences and workshops
- Experiment with new techniques
- Collaborate with research teams

---

## Coding Challenges

Common coding problems:
1. Implement attention mechanism
2. Write CUDA kernel for specific operation
3. Optimize PyTorch model for inference
4. Profile and optimize inference latency

See `coding_challenges.md` for detailed problems and solutions.

