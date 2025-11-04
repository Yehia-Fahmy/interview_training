"""
Solution for Exercise 3: Distributed Training Pipeline

This file contains a simplified reference solution.
Note: This is educational - real implementations use optimized libraries.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import time


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
        for key in self.gradient_buffer:
            self.gradient_buffer[key] += gradients[key]
        self.gradients_received += 1
    
    def pull_parameters(self) -> Dict[str, Any]:
        """Return current parameters"""
        return self.params.copy()
    
    def update_parameters(self, learning_rate: float):
        """Update parameters from accumulated gradients"""
        if self.gradients_received == 0:
            return
        
        # Average gradients
        for key in self.params:
            avg_gradient = self.gradient_buffer[key] / self.gradients_received
            self.params[key] -= learning_rate * avg_gradient
        
        # Reset buffer
        self.gradient_buffer = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.gradients_received = 0


class DistributedTrainer:
    """Distributed training coordinator (simplified)"""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
    
    def train(self, X: np.ndarray, y: np.ndarray,
             model_config: Dict, num_epochs: int = 10):
        """Coordinate distributed training (simplified)"""
        # Simplified implementation
        # In practice, this would use multiprocessing or distributed frameworks
        print(f"Distributed training with {self.num_workers} workers")
        print(f"Note: This is a simplified educational implementation")
        print(f"Real implementations use PyTorch DDP, Horovod, or similar frameworks")


if __name__ == "__main__":
    print("Distributed Training Framework")
    print("Note: This is a simplified framework")
    print("Real implementations use optimized libraries like PyTorch DDP or Horovod")

