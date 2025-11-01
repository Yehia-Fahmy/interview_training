# Exercise 3: Design a Distributed ML Training Infrastructure

**Difficulty:** Hard  
**Time Limit:** 60 minutes  
**Focus:** Training infrastructure, resource management, data pipelines

## Problem

Design a training infrastructure that:
- Supports multiple ML teams and projects
- Handles distributed training (data/model parallelism)
- Manages GPU/TPU resources efficiently
- Supports hyperparameter tuning
- Tracks experiments and results
- Handles large datasets (petabyte-scale)

## Requirements to Discuss

1. **Resource Management**
   - Job scheduling (Slurm, Kubernetes)?
   - GPU allocation?
   - Fair sharing across teams?
   - Spot instances for cost optimization?

2. **Distributed Training**
   - Data parallelism (PyTorch DDP, Horovod)?
   - Model parallelism (large models)?
   - Pipeline parallelism?
   - Parameter server vs all-reduce?

3. **Data Pipeline**
   - Data storage (object store, distributed filesystem)?
   - Data versioning?
   - Preprocessing pipeline?
   - Data streaming for training?

4. **Experiment Management**
   - Experiment tracking (MLflow)?
   - Hyperparameter tuning (Ray Tune, Optuna)?
   - Artifact storage?
   - Reproducibility?

5. **Monitoring**
   - Training job monitoring?
   - GPU utilization?
   - Data pipeline monitoring?
   - Cost tracking?

6. **Scalability**
   - How to handle hundreds of concurrent jobs?
   - Queue management?
   - Priority scheduling?

## Key Topics to Cover

- **Distributed Training**: Data/model/pipeline parallelism
- **Job Orchestration**: Kubernetes, Slurm, custom scheduler
- **Storage**: HDFS, S3, distributed filesystems
- **Experiment Tracking**: MLflow, Weights & Biases
- **Hyperparameter Tuning**: Bayesian optimization, grid search

## Sample Discussion Points

1. "I'd use Kubernetes for orchestration with a custom job scheduler. Jobs are submitted as Kubernetes jobs with resource requests (GPUs, memory, CPU)."

2. "For distributed training, I'd support PyTorch DDP for data parallelism. Large models would use model parallelism or pipeline parallelism across multiple GPUs."

3. "Data would be stored in a distributed object store (S3-compatible). Datasets are versioned. Training jobs mount datasets or stream from the store."

4. "Experiment tracking integrated with MLflow. Each training run logs hyperparameters, metrics, and artifacts. Supports comparison across runs."

5. "Hyperparameter tuning would use Ray Tune with Bayesian optimization. Parallel trials share GPU cluster, early stopping for unpromising trials."

6. "Monitoring tracks: GPU utilization, training throughput (samples/sec), job queue length, cluster utilization. Set alerts for stuck jobs or low GPU usage."

## Additional Considerations

- How to handle dataset caching for frequently used data?
- How to support interactive development (Jupyter)?
- How to manage dependencies and environments?
- How to handle preemption for high-priority jobs?

