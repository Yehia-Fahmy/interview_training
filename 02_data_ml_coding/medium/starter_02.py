"""
Exercise 2: Model Monitoring and Drift Detection

Implement a monitoring system that detects:
1. Data drift - Distribution shift in input features
2. Prediction drift - Change in prediction distribution
3. Concept drift - Relationship between features and target changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    feature: str
    drift_type: str  # 'data_drift', 'concept_drift', 'prediction_drift'
    statistic: float
    p_value: float
    severity: str  # 'low', 'medium', 'high'
    detected_at: datetime


class ModelMonitor:
    """
    Monitor model performance and detect drift.
    
    Tracks baseline statistics and compares against incoming data.
    """
    
    def __init__(self, baseline_data: pd.DataFrame,
                 baseline_predictions: Optional[np.ndarray] = None,
                 drift_threshold: float = 0.05):
        """
        Initialize monitor with baseline data.
        
        Args:
            baseline_data: Training/reference data
            baseline_predictions: Baseline predictions (if available)
            drift_threshold: P-value threshold for drift detection
        """
        self.baseline_data = baseline_data
        self.baseline_predictions = baseline_predictions
        self.drift_threshold = drift_threshold
        self.baseline_stats = self._compute_baseline_stats()
        self.alerts: List[DriftAlert] = []
    
    def _compute_baseline_stats(self) -> Dict:
        """Compute baseline statistics for each feature"""
        # TODO: Implement baseline statistics computation
        # For numerical features: mean, std, min, max, distribution
        # For categorical features: value_counts, distribution
        pass
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """
        Detect data drift in input features.
        
        Args:
            current_data: Current data to check
        
        Returns:
            List of drift alerts
        """
        alerts = []
        
        # TODO: Implement data drift detection
        # For each feature:
        #   - If numerical: Use Kolmogorov-Smirnov test
        #   - If categorical: Use Chi-square test
        #   - Create DriftAlert if p-value < threshold
        #   - Determine severity based on p-value
        
        return alerts
    
    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Optional[DriftAlert]:
        """
        Detect drift in prediction distribution.
        
        Args:
            current_predictions: Current predictions
        
        Returns:
            DriftAlert if drift detected, None otherwise
        """
        # TODO: Implement prediction drift detection
        # Use Kolmogorov-Smirnov test to compare distributions
        # Return DriftAlert if significant drift detected
        pass
    
    def _determine_severity(self, statistic: float, p_value: float) -> str:
        """Determine severity of drift"""
        # TODO: Implement severity determination
        # 'high' if p < 0.01, 'medium' if p < 0.05, 'low' otherwise
        pass
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        # TODO: Implement report generation
        # Include summary of alerts, statistics, etc.
        pass


# Usage example
if __name__ == "__main__":
    # Generate baseline data
    baseline = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.normal(5, 2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    baseline_preds = np.random.normal(0.5, 0.1, 1000)
    
    # Initialize monitor
    monitor = ModelMonitor(baseline, baseline_preds)
    
    # Simulate drift (shifted distribution)
    current = pd.DataFrame({
        'feature1': np.random.normal(2, 1, 100),  # Drifted!
        'feature2': np.random.normal(5, 2, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Detect drift
    alerts = monitor.detect_data_drift(current)
    print(f"Detected {len(alerts)} drift alerts")
    
    # Generate report
    print(monitor.generate_report())

