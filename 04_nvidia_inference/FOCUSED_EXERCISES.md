# Focused Practice Exercises for NVIDIA Interview

**Priority: High-ROI exercises only**

These exercises are ranked by interview relevance. Focus on completing exercises 1-5 before moving to 6-10.

---

## MUST DO (Exercises 1-5)

### Exercise 1: Extract and Visualize FX Graph ⭐⭐⭐
**Time:** 1-2 hours  
**Priority:** CRITICAL - Core to the role

**Task:**
1. Load a HuggingFace transformer model (e.g., `bert-base-uncased`)
2. Extract FX graph using `torch.fx.symbolic_trace` or `torch.export`
3. Print graph nodes and operations
4. Visualize graph structure (simple text representation is fine)
5. Handle dynamic shapes with `torch.export` constraints

**Skills Tested:**
- PyTorch FX understanding
- Model graph representation
- Debugging graph extraction issues

**Starter Code:**
```python
import torch
from transformers import AutoModel
import torch.fx as fx

# Load model
model = AutoModel.from_pretrained("bert-base-uncased")

# TODO: Extract FX graph
# TODO: Print nodes and operations
# TODO: Handle dynamic shapes
```

**Success Criteria:**
- Successfully extract graph from arbitrary model
- Understand graph structure (nodes, edges, operations)
- Can explain what each node represents

---

### Exercise 2: Implement Post-Training Quantization ⭐⭐⭐
**Time:** 2-3 hours  
**Priority:** CRITICAL - Will likely be asked in coding round

**Task:**
1. Take a pre-trained model (ResNet or BERT)
2. Implement PTQ using `torch.quantization`
3. Calibrate on sample data
4. Measure accuracy before/after
5. Measure inference speed improvement
6. Compare INT8 vs FP32 memory usage

**Skills Tested:**
- Quantization implementation
- Accuracy-speed trade-offs
- Profiling and measurement

**Starter Code:**
```python
import torch
import torch.quantization as quant

# TODO: Implement PTQ pipeline
# 1. Prepare model for quantization
# 2. Calibrate on data
# 3. Convert to quantized model
# 4. Measure accuracy and speed
```

**Success Criteria:**
- Working PTQ implementation
- Clear metrics on accuracy vs speed trade-off
- Understanding of when to use PTQ vs QAT

---

### Exercise 3: Write Custom FX Pass ⭐⭐⭐
**Time:** 2-3 hours  
**Priority:** HIGH - Tests graph manipulation skills

**Task:**
1. Extract FX graph from a model
2. Write a custom pass to replace operations (e.g., replace ReLU with GELU)
3. Write a pass to fuse operations (e.g., Conv + BatchNorm + ReLU)
4. Verify correctness with test inputs
5. Measure performance impact

**Skills Tested:**
- FX graph manipulation
- Compiler-like thinking
- Optimization pattern recognition

**Starter Code:**
```python
import torch.fx as fx

class ReplaceReLUWithGELU(fx.Transformer):
    # TODO: Implement pass to replace ReLU with GELU
    pass

# TODO: Apply pass to model
# TODO: Verify correctness
# TODO: Measure performance
```

**Success Criteria:**
- Correctly modify graph structure
- Maintain model correctness
- Understand when transformations are valid

---

### Exercise 4: Profile Transformer Inference ⭐⭐
**Time:** 1-2 hours  
**Priority:** HIGH - You need to identify bottlenecks

**Task:**
1. Profile a transformer model inference (use HuggingFace BERT or GPT-2)
2. Use `torch.profiler` to identify bottlenecks
3. Identify top 5 slowest operations
4. Categorize: compute-bound vs memory-bound
5. Propose optimization strategies

**Skills Tested:**
- Profiling tools
- Bottleneck identification
- Optimization strategy

**Starter Code:**
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # TODO: Run inference
    pass

# TODO: Analyze results
# TODO: Identify bottlenecks
# TODO: Propose optimizations
```

**Success Criteria:**
- Identify attention as major bottleneck
- Understand compute vs memory bound operations
- Propose concrete optimizations (batching, kernel fusion, etc.)

---

### Exercise 5: Design Automated Optimization Pipeline ⭐⭐⭐
**Time:** 2-3 hours  
**Priority:** CRITICAL - System design question

**Task:**
Design (on paper/diagram) an automated inference optimization platform.

**Requirements:**
- Input: PyTorch model from user
- Output: Optimized model ready for deployment
- Support multiple optimization techniques (quantization, pruning, compilation)
- Handle versioning and rollback
- Scale to 1000s of models

**Components to Design:**
1. Model ingestion and storage
2. Graph extraction module
3. Optimization pipeline (pluggable optimizations)
4. Validation and testing
5. Model registry and versioning
6. Deployment and serving
7. Monitoring and alerting

**Success Criteria:**
- Clear architecture diagram
- Discussion of trade-offs (accuracy vs speed, complexity vs flexibility)
- Scalability considerations
- Failure handling and rollback

---

## SHOULD DO (Exercises 6-8)

### Exercise 6: Implement Structured Pruning ⭐⭐
**Time:** 2-3 hours  
**Priority:** MEDIUM

**Task:**
1. Implement magnitude-based channel pruning for a CNN
2. Prune 30% of channels
3. Fine-tune pruned model
4. Measure accuracy drop and speedup

**Why This Matters:** Understanding pruning trade-offs

---

### Exercise 7: Understand Tensor Parallelism ⭐⭐
**Time:** 2-3 hours  
**Priority:** MEDIUM

**Task:**
1. Implement column/row parallelism for a single transformer layer
2. Calculate memory requirements for 2-GPU vs 4-GPU setup
3. Estimate communication overhead

**Why This Matters:** Distributed inference for large models

**Note:** You don't need to implement full distributed training, just understand the concepts.

---

### Exercise 8: Compare torch.compile Backends ⭐
**Time:** 1-2 hours  
**Priority:** MEDIUM

**Task:**
1. Profile same model with different backends (inductor, eager, etc.)
2. Measure compilation time vs inference speedup
3. Understand when compilation is worth it

**Why This Matters:** Understanding torch.compile tradeoffs

---

## OPTIONAL (Exercises 9-10)

### Exercise 9: Simple CUDA Vector Addition ⭐
**Time:** 2-3 hours  
**Priority:** LOW (nice to have)

**Task:**
Write a simple CUDA kernel for vector addition. Profile vs PyTorch.

**Why Include:** Shows GPU programming awareness, but not critical.

---

### Exercise 10: Implement KV-Cache Management ⭐
**Time:** 2-3 hours  
**Priority:** LOW

**Task:**
Implement basic KV-cache for autoregressive generation.

**Why Include:** LLM-specific optimization, may come up in discussion.

---

## Practice Schedule

### 1-Week Timeline:
- **Day 1:** Exercise 1 (FX Graph Extraction)
- **Day 2:** Exercise 2 (PTQ) + Exercise 3 (Custom FX Pass)
- **Day 3:** Exercise 4 (Profiling)
- **Day 4:** Exercise 5 (System Design - on paper)
- **Day 5:** Exercise 6 (Pruning) OR Exercise 7 (Tensor Parallelism)
- **Day 6:** Mock interview with exercises 1-5
- **Day 7:** Review and refine

### 2-Week Timeline:
- **Week 1:** Exercises 1-5 (deep understanding)
- **Week 2:** Exercises 6-8, mock interviews, review

---

## Code Implementation Tips

1. **Start simple, then optimize:** Get it working first
2. **Measure everything:** Before/after metrics are crucial
3. **Handle edge cases:** Empty inputs, large models, etc.
4. **Write clean code:** Readability matters in interviews
5. **Document trade-offs:** Explain why you made certain choices

---

## Resources for Exercises

### Exercise 1-3 (FX Graphs):
- PyTorch FX docs: https://pytorch.org/docs/stable/fx.html
- torch.export guide: https://pytorch.org/docs/stable/export.html
- HuggingFace transformers docs

### Exercise 2 (Quantization):
- PyTorch quantization tutorial: https://pytorch.org/docs/stable/quantization.html
- Intel Neural Compressor docs

### Exercise 4 (Profiling):
- torch.profiler docs: https://pytorch.org/docs/stable/profiler.html
- PyTorch performance tuning guide

### Exercise 5 (System Design):
- TRT Model Optimizer architecture (NVIDIA blog)
- HuggingFace Optimum architecture
- Review `03_system_design/` for patterns

---

## What NOT to Spend Time On

❌ **Deep CUDA programming** - Understanding > implementation  
❌ **TensorRT plugin development** - They'll train you  
❌ **Triton kernel DSL** - Nice to know, not critical  
❌ **Research papers** - Focus on engineering, not research  
❌ **Every optimization technique** - Focus on quantization, pruning, compilation

---

## Interview Readiness Checklist

After completing exercises 1-5, you should be able to:

- [ ] Extract FX graph from arbitrary PyTorch model
- [ ] Implement PTQ in <30 minutes with clean code
- [ ] Write custom FX pass to modify graph
- [ ] Profile inference and identify bottlenecks
- [ ] Design automated optimization platform on whiteboard
- [ ] Explain trade-offs for each optimization technique
- [ ] Discuss 2-3 projects with concrete metrics

**If you can do all of the above, you're interview-ready for the coding and system design rounds.**

Good luck! 🚀
