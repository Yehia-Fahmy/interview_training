# Data/ML Coding - Hard Exercises

These exercises focus on advanced production ML scenarios: LLM evaluation, distributed training, real-time serving, and complex pipelines.

---

## Exercise 1: LLM Evaluation Framework

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** LLM evaluation, production ML for language models (critical for Improvement Engineer role)

### Problem

Build a comprehensive evaluation framework for Large Language Models. The role involves deploying LLMs, so understanding how to evaluate them is crucial.

### Requirements

1. Implement evaluation for:
   - **Automated metrics**: BLEU, ROUGE, perplexity
   - **Semantic similarity**: Embedding-based evaluation
   - **Task-specific metrics**: Accuracy for classification tasks
   - **Custom evaluators**: Extensible framework for new metrics

2. Support batch and streaming evaluation

3. Create a framework that can:
   - Compare multiple LLMs
   - Track evaluation over time
   - Generate comprehensive reports

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 2: Model Serving Infrastructure

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** Production model serving, API design, performance optimization

### Problem

Design and implement a model serving system that can:
1. Serve multiple models concurrently
2. Handle batching for efficiency
3. Support different model types (sklearn, PyTorch, etc.)
4. Provide health checks and monitoring
5. Handle versioning and A/B testing

### Requirements

1. Create a `ModelServer` class with:
   - `load_model()` - Load a model from registry
   - `predict()` - Single prediction
   - `predict_batch()` - Batch prediction with batching
   - `health_check()` - Server health status
   - `get_metrics()` - Performance metrics

2. Implement:
   - Request queuing for batching
   - Async prediction handling
   - Model caching and loading
   - Simple API endpoint (Flask/FastAPI)

3. Add monitoring:
   - Request latency
   - Throughput
   - Error rates
   - Queue depth

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 3: Distributed Training Pipeline

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** Distributed ML, parallel training, data parallelism

### Problem

Implement a distributed training system that can:
1. Split data across multiple workers
2. Coordinate gradient updates
3. Handle worker failures
4. Support both data and model parallelism

### Requirements

1. Create a distributed training coordinator
2. Implement parameter server or all-reduce pattern
3. Handle worker synchronization
4. Support checkpointing and recovery

Note: This is a simplified version - in practice you'd use frameworks like PyTorch DDP or Horovod.

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

---

## Exercise 4: Real-time Feature Pipeline

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** Stream processing, feature engineering, low-latency systems

### Problem

Build a real-time feature engineering pipeline that:
1. Processes streaming data
2. Computes features on-the-fly
3. Maintains feature stores for lookups
4. Handles windowed aggregations

### Requirements

1. Stream processing framework
2. Feature computation (windowed stats, joins, etc.)
3. Feature store for serving
4. Low-latency requirements

2. Support for:
   - Windowed aggregations (sliding windows)
   - Feature lookups (historical data)
   - Real-time transformations

See the full exercise details in `EXERCISES.md` for complete solution template and implementation details.

