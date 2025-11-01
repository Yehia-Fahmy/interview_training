# Exercise 3: Distributed Training Pipeline

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** Distributed ML, parallel training, data parallelism

## Problem

Implement a distributed training system that can:
1. Split data across multiple workers
2. Coordinate gradient updates
3. Handle worker failures
4. Support both data and model parallelism

## Requirements

1. Create a distributed training coordinator
2. Implement parameter server or all-reduce pattern
3. Handle worker synchronization
4. Support checkpointing and recovery

Note: This is a simplified version - in practice you'd use frameworks like PyTorch DDP or Horovod.

## Solution Template

```python
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import time
import pickle

@dataclass
class WorkerConfig:
    """Configuration for a training worker"""
    worker_id: int
    data_indices: List[int]
    model_state: Dict[str, Any]
    learning_rate: float

class ParameterServer:
    """Central parameter server for distributed training"""
    
    def __init__(self, initial_params: Dict[str, Any]):
        self.params = initial_params
        self.gradient_buffer = {k: np.zeros_like(v) for k, v in initial_params.items()}
        self.num_workers = 0
        self.gradients_received = 0
    
    def push_gradients(self, worker_id: int, gradients: Dict[str, Any]):
        """Receive gradients from worker and accumulate"""
        # Your implementation
        pass
    
    def pull_parameters(self) -> Dict[str, Any]:
        """Return current parameters"""
        return self.params
    
    def update_parameters(self, learning_rate: float):
        """Update parameters from accumulated gradients"""
        # Your implementation
        pass

def worker_process(config: WorkerConfig, 
                  data: np.ndarray,
                  labels: np.ndarray,
                  shared_params: mp.Manager):
    """Training worker process"""
    # Initialize local model with shared parameters
    # Train on assigned data slice
    # Compute gradients
    # Send gradients to parameter server
    # Pull updated parameters
    # Repeat for multiple epochs
    pass

class DistributedTrainer:
    """Distributed training coordinator"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.workers = []
    
    def train(self, X: np.ndarray, y: np.ndarray,
             model_config: Dict, num_epochs: int = 10):
        """Coordinate distributed training"""
        # Split data across workers
        # Initialize parameter server
        # Start worker processes
        # Coordinate training loop
        # Handle synchronization
        pass

# This is a simplified framework
# Real implementations use optimized libraries
```

## Key Learning Points

1. **Distributed Systems:** Parameter servers, all-reduce
2. **Parallelism Strategies:** Data vs model parallelism
3. **Fault Tolerance:** Handling worker failures

