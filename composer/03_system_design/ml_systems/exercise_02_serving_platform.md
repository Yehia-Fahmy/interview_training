# Exercise 2: Design an ML Model Serving Platform

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Production ML, serving infrastructure, reliability

## Problem

Design a platform for serving ML models (especially relevant for 8090's LLM deployment) that:
- Serves multiple models simultaneously
- Handles variable traffic loads (scale up/down)
- Supports both batch and real-time inference
- Provides model versioning and rollback
- Includes monitoring and alerting
- Handles GPU/accelerator resources

## Requirements to Discuss

1. **Architecture**
   - Microservices vs monolithic?
   - API gateway?
   - Model servers vs serverless?

2. **Model Management**
   - Model registry?
   - Versioning strategy?
   - Rollback mechanism?
   - Canary deployments?

3. **Serving Infrastructure**
   - Container orchestration (Kubernetes)?
   - Auto-scaling policies?
   - Resource allocation (CPU/GPU)?
   - Request routing?

4. **Batching & Optimization**
   - How to batch requests?
   - Model optimization (quantization, pruning)?
   - Batch size optimization?

5. **Monitoring & Observability**
   - What metrics to track?
   - Latency, throughput, errors?
   - Model performance metrics?
   - Drift detection?

6. **Reliability**
   - High availability?
   - Load balancing?
   - Circuit breakers?
   - Graceful degradation?

## Key Topics to Cover

- **Model Serving Patterns**: Online, batch, streaming
- **Containerization**: Docker, Kubernetes deployments
- **Auto-scaling**: Horizontal pod autoscaling, queue-based scaling
- **Model Optimization**: ONNX, TensorRT, quantization
- **Monitoring**: Prometheus, custom metrics, SLAs

## Sample Discussion Points

1. "I'd use a microservices architecture. Each model type (e.g., vision, NLP, tabular) would have dedicated services for optimal resource allocation."

2. "For model management, I'd implement a model registry with versioning. Models are containerized and deployed via Kubernetes. Supports canary deployments for gradual rollout."

3. "Serving would use a model server like TorchServe or TensorFlow Serving. Requests are batched automatically. GPU resources shared via Kubernetes resource quotas."

4. "Auto-scaling would be based on queue depth and request latency. Scale up when queue grows, scale down during low traffic."

5. "Monitoring would track: request latency (p50, p95, p99), throughput, error rates, model-specific metrics (prediction distributions). Set up alerts for drift detection."

6. "For reliability, I'd implement circuit breakers. If a model fails, automatically fallback to previous version or simpler model."

## Additional Considerations

- How to handle model warmup (cold starts)?
- How to manage GPU memory efficiently?
- How to support multi-tenant scenarios?
- How to handle model updates without downtime?

