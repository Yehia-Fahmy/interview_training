"""
Solution for Exercise 4: Real-time Feature Pipeline

This file contains the reference solution.
"""

from collections import deque
from typing import Dict, List, Any, Callable
import time
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureWindow:
    """Sliding window for feature computation"""
    window_size: int  # Number of records
    window_duration: float  # Time duration in seconds
    values: deque
    timestamps: deque


class RealTimeFeaturePipeline:
    """Real-time feature engineering pipeline"""
    
    def __init__(self):
        self.feature_store: Dict[str, Any] = {}
        self.windows: Dict[str, FeatureWindow] = {}
        self.transformations: Dict[str, Callable] = {}
    
    def register_window(self, feature_name: str, window_size: int = 100,
                       window_duration: float = 60.0):
        """Register a sliding window feature"""
        self.windows[feature_name] = FeatureWindow(
            window_size=window_size,
            window_duration=window_duration,
            values=deque(maxlen=window_size),
            timestamps=deque(maxlen=window_size)
        )
    
    def process_event(self, event: Dict[str, Any]):
        """Process a streaming event"""
        entity_id = event.get('entity_id')
        timestamp = event.get('timestamp', time.time())
        
        if entity_id not in self.feature_store:
            self.feature_store[entity_id] = {}
        
        # Update windows
        for feature_name, window in self.windows.items():
            value = event.get(feature_name)
            if value is not None:
                window.values.append(value)
                window.timestamps.append(timestamp)
                
                # Compute windowed features
                if len(window.values) > 0:
                    # Remove old values outside time window
                    current_time = timestamp
                    while (len(window.timestamps) > 0 and 
                           current_time - window.timestamps[0] > window.window_duration):
                        window.values.popleft()
                        window.timestamps.popleft()
                    
                    # Compute aggregations
                    if len(window.values) > 0:
                        self.feature_store[entity_id][f"{feature_name}_mean"] = np.mean(window.values)
                        self.feature_store[entity_id][f"{feature_name}_sum"] = np.sum(window.values)
                        self.feature_store[entity_id][f"{feature_name}_count"] = len(window.values)
    
    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict:
        """Retrieve computed features"""
        entity_features = self.feature_store.get(entity_id, {})
        return {name: entity_features.get(name, None) for name in feature_names}


if __name__ == "__main__":
    pipeline = RealTimeFeaturePipeline()
    pipeline.register_window('amount', window_size=100, window_duration=60.0)
    
    # Simulate events
    for i in range(10):
        event = {
            'entity_id': 'user_123',
            'amount': i * 10,
            'timestamp': time.time()
        }
        pipeline.process_event(event)
    
    features = pipeline.get_features('user_123', ['amount_mean', 'amount_sum', 'amount_count'])
    print("Features:", features)

