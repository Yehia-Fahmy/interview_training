"""
Exercise 4: Real-time Feature Pipeline

Build a real-time feature engineering pipeline that:
1. Processes streaming data
2. Computes features on-the-fly
3. Maintains feature stores for lookups
4. Handles windowed aggregations
"""

from collections import deque
from typing import Dict, List, Any, Callable
import time
from dataclasses import dataclass


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
        # TODO: Create FeatureWindow and add to self.windows
        pass
    
    def process_event(self, event: Dict[str, Any]):
        """Process a streaming event"""
        # TODO: Update windows with new event data
        # Compute windowed features
        # Store features in feature_store
        pass
    
    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict:
        """Retrieve computed features"""
        # TODO: Return requested features for entity
        pass


if __name__ == "__main__":
    pipeline = RealTimeFeaturePipeline()
    print("Real-time Feature Pipeline Framework")

