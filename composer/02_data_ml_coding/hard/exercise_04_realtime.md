# Exercise 4: Real-time Feature Pipeline

**Difficulty:** Hard  
**Time Limit:** 90 minutes  
**Focus:** Stream processing, feature engineering, low-latency systems

## Problem

Build a real-time feature engineering pipeline that:
1. Processes streaming data
2. Computes features on-the-fly
3. Maintains feature stores for lookups
4. Handles windowed aggregations

## Requirements

1. Stream processing framework
2. Feature computation (windowed stats, joins, etc.)
3. Feature store for serving
4. Low-latency requirements

2. Support for:
   - Windowed aggregations (sliding windows)
   - Feature lookups (historical data)
   - Real-time transformations

## Solution Template

```python
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
        self.windows[feature_name] = FeatureWindow(
            window_size=window_size,
            window_duration=window_duration,
            values=deque(maxlen=window_size),
            timestamps=deque(maxlen=window_size)
        )
    
    def process_event(self, event: Dict[str, Any]):
        """Process a streaming event"""
        # Update windows
        # Compute features
        # Store features
        pass
    
    def get_features(self, entity_id: str, feature_names: List[str]) -> Dict:
        """Retrieve computed features"""
        pass
```

## Key Learning Points

1. **Stream Processing:** Handling continuous data
2. **Feature Engineering:** Real-time computation
3. **Low Latency:** Optimizing for speed

