"""
Solution for Exercise 2: Model Serving Infrastructure

This file contains the reference solution.
"""

import time
import threading
from queue import Queue, Empty
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
        self.total_requests += 1
        if success:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1
        
        self.latency_history.append(latency)
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        
        # Update averages
        if len(self.latency_history) > 0:
            self.average_latency = np.mean(self.latency_history)
            self.p95_latency = np.percentile(self.latency_history, 95)


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
        if model_type == 'sklearn':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.models[model_id] = {
            'model': model,
            'type': model_type,
            'loaded_at': time.time()
        }
    
    def predict(self, model_id: str, features: Any, 
                request_id: Optional[str] = None) -> Any:
        """Synchronous prediction"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")
        
        model_info = self.models[model_id]
        model = model_info['model']
        model_type = model_info['type']
        
        start_time = time.time()
        try:
            if model_type == 'sklearn':
                prediction = model.predict([features])[0]
            else:
                prediction = None
            
            latency = time.time() - start_time
            self.metrics[model_id].record_prediction(latency, True)
            
            return prediction
        except Exception as e:
            latency = time.time() - start_time
            self.metrics[model_id].record_prediction(latency, False)
            raise
    
    def predict_batch_async(self, model_id: str, features: Any,
                           callback: Optional[callable] = None) -> str:
        """Asynchronous batch prediction"""
        request_id = f"req_{int(time.time() * 1000000)}"
        
        request = PredictionRequest(
            request_id=request_id,
            features=features,
            callback=callback
        )
        
        try:
            self.request_queue.put((model_id, request), timeout=1.0)
        except:
            raise RuntimeError("Request queue full")
        
        return request_id
    
    def _batch_worker(self):
        """Worker thread that processes batches"""
        batch = []
        last_batch_time = time.time()
        
        while self.running:
            try:
                try:
                    model_id, request = self.request_queue.get(timeout=self.max_wait_time)
                    batch.append((model_id, request))
                except Empty:
                    # Timeout - process current batch if exists
                    if batch and time.time() - last_batch_time >= self.max_wait_time:
                        self._process_batch(batch)
                        batch = []
                        last_batch_time = time.time()
                    continue
                
                # Process batch if full
                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
            
            except Exception as e:
                print(f"Error in batch worker: {e}")
    
    def _process_batch(self, batch: List[tuple]):
        """Process a batch of requests"""
        by_model = defaultdict(list)
        for model_id, request in batch:
            by_model[model_id].append(request)
        
        for model_id, requests in by_model.items():
            if model_id not in self.models:
                for req in requests:
                    if req.callback:
                        req.callback(None, ValueError(f"Model {model_id} not found"))
                continue
            
            features_batch = [req.features for req in requests]
            
            start_time = time.time()
            try:
                model = self.models[model_id]['model']
                model_type = self.models[model_id]['type']
                
                if model_type == 'sklearn':
                    predictions = model.predict(features_batch)
                else:
                    predictions = None
                
                latency = time.time() - start_time
                self.metrics[model_id].record_prediction(latency, True)
                
                for req, pred in zip(requests, predictions):
                    if req.callback:
                        req.callback(pred, None)
            
            except Exception as e:
                latency = time.time() - start_time
                self.metrics[model_id].record_prediction(latency, False)
                for req in requests:
                    if req.callback:
                        req.callback(None, e)
    
    def start(self):
        """Start the model server"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the model server"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def health_check(self) -> Dict:
        """Get server health status"""
        return {
            'status': 'healthy' if self.running else 'stopped',
            'models_loaded': len(self.models),
            'queue_size': self.request_queue.qsize(),
            'metrics': {model_id: {
                'total_requests': m.total_requests,
                'average_latency': m.average_latency,
                'success_rate': m.successful_predictions / max(m.total_requests, 1)
            } for model_id, m in self.metrics.items()}
        }

