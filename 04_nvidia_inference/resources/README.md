# Study Resources

Curated list of resources for NVIDIA Inference Engineer interview preparation.

## Documentation

### PyTorch
- [PyTorch 2.0 Documentation](https://pytorch.org/docs/stable/index.html)
- [torch.compile Tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [FX Graph Documentation](https://pytorch.org/docs/stable/fx.html)
- [torch.export API](https://pytorch.org/docs/stable/export.html)
- [Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)

### CUDA & GPU
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)

### TensorRT
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

### Triton
- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/python-api/triton.html)

### CUTLASS
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/wiki)

## Papers

### Model Optimization
- "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (Jacob et al., 2018)
- "The Lottery Ticket Hypothesis" (Frankle & Carbin, 2019)
- "Neural Architecture Search: A Survey" (Elsken et al., 2019)

### Attention Mechanisms
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
- "Efficiently Scaling Transformer Inference" (Pope et al., 2023)

### Distributed Systems
- "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al., 2021)
- "Sequence Parallelism" (Korthikanti et al., 2023)

### System Design
- "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., 2015)
- "MLOps: Continuous delivery and automation pipelines in ML" (Various)

## Tutorials

### PyTorch
- [PyTorch FX Tutorial](https://pytorch.org/tutorials/intermediate/fx_conv_bn_fuser.html)
- [PyTorch Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [HuggingFace Optimization Guide](https://huggingface.co/docs/transformers/perf_infer_gpu_one)

### CUDA
- [NVIDIA CUDA Tutorials](https://developer.nvidia.com/cuda-tutorial)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)

### TensorRT
- [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
- [TensorRT Python API Tutorial](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

### Distributed Training
- [Megatron-LM Tutorials](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch Distributed Tutorials](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

## Books

- "Programming Massively Parallel Processors" (Kirk & Hwu)
- "Professional CUDA C Programming" (Wilt)
- "Designing Data-Intensive Applications" (Kleppmann) - For system design

## Online Courses

- NVIDIA DLI (Deep Learning Institute) courses
- CUDA Programming courses
- ML System Design courses

## GitHub Repositories

### Reference Implementations
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - Tensor parallelism
- [Flash Attention](https://github.com/Dao-AILab/flash-attention) - Flash Attention implementation
- [TensorRT Examples](https://github.com/NVIDIA/TensorRT) - TensorRT samples
- [PyTorch Examples](https://github.com/pytorch/examples) - PyTorch examples

### Tools
- [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) - PyTorch to TensorRT
- [Triton](https://github.com/openai/triton) - Triton kernel DSL

## Blogs & Articles

- PyTorch Blog (torch.compile, FX, etc.)
- NVIDIA Developer Blog
- HuggingFace Blog (optimization techniques)

## Conferences

- NeurIPS, ICML, ICLR (ML optimization papers)
- GPU Technology Conference (GTC) (NVIDIA-specific)
- PyTorch Developer Conference

## Practice Platforms

- LeetCode (for coding practice)
- System Design Interview (for system design)
- Kaggle (for ML practice)

## Community

- PyTorch Forums
- CUDA Forums
- Stack Overflow (tagged: pytorch, cuda, tensorrt)
- Reddit: r/MachineLearning, r/pytorch

## Quick Reference

### Cheat Sheets
- CUDA Programming Cheat Sheet
- PyTorch API Reference
- TensorRT API Reference

### Tools Installation
- CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- TensorRT: https://developer.nvidia.com/tensorrt
- Nsight Systems: https://developer.nvidia.com/nsight-systems
- Triton: `pip install triton` or conda

---

**Note**: Some resources require NVIDIA Developer account or specific hardware. Check requirements before starting.

