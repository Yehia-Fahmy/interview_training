"""
Exercise 2: Model Serving Infrastructure

Design and implement a model serving system that can:
1. Serve multiple models concurrently
2. Handle batching for efficiency
3. Support different model types
4. Provide health checks and monitoring
"""

import time
import threading
from queue import Queue
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import pickle
from pathlib import Path
import numpy as np


@dataclass
class PredictionRequest:
    """Request for model prediction"""
    request_id: str
    features: Any
    callback: Optional[callable] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ModelMetrics:
    """Metrics for model serving"""
    total_requests: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    average_latency: float = 0.0
    p95_latency: float = 0.0
    latency_history: List[float] = None
    
    def __post_init__(self):
        if self.latency_history is None:
            self.latency_history = []
    
    def record_prediction(self, latency: float, success: bool):
        """Record a prediction"""
        # TODO: Update metrics
        pass


class ModelServer:
    """
    Model serving infrastructure with batching and async handling.
    """
    
    def __init__(self, batch_size: int = 32, 
                 max_wait_time: float = 0.1,
                 max_queue_size: int = 1000):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_queue_size = max_queue_size
        
        self.models: Dict[str, Any] = {}
        self.request_queue: Queue = Queue(maxsize=max_queue_size)
        self.metrics: Dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        
        self.running = False
        self.worker_thread: Optional[threading.Thread] = None
    
    def load_model(self, model_id: str, model_path: Path, model_type: str = 'sklearn'):
        """Load a model into the server"""
        # TODO: Load model based on type (sklearn, pytorch, etc.)
        pass
    
    def predict(self, model_id: str, features: Any, 
                request_id: Optional[str] = None) -> Any:
        """Synchronous prediction"""
        # TODO: Implement synchronous prediction
        pass
    
    def predict_batch_async(self, model_id: str, features: Any,
                           callback: Optional[callable] = None) -> str:
        """Asynchronous batch prediction"""
        # TODO: Add request to queue
        pass
    
    def _batch_worker(self):
        """Worker thread that processes batches"""
        # TODO: Implement batch processing worker
        pass
    
    def _process_batch(self, batch: List[tuple]):
        """Process a batch of requests"""
        # TODO: Group by model, predict, call callbacks
        pass
    
    def start(self):
        """Start the model server"""
        # TODO: Start worker thread
        pass
    
    def stop(self):
        """Stop the model server"""
        # TODO: Stop worker thread
        pass
    
    def health_check(self) -> Dict:
        """Get server health status"""
        # TODO: Return health status with metrics
        pass


if __name__ == "__main__":
    server = ModelServer(batch_size=32, max_wait_time=0.1)
    # server.load_model("model_v1", Path("model.pkl"))
    server.start()
    print(server.health_check())
    server.stop()

