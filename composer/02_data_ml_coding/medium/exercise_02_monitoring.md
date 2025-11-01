# Exercise 2: Model Monitoring and Drift Detection

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Production ML monitoring, detecting model degradation

## Problem

Implement a monitoring system that detects:
1. **Data drift** - Distribution shift in input features
2. **Prediction drift** - Change in prediction distribution
3. **Concept drift** - Relationship between features and target changes

This is critical for an Improvement Engineer role that involves ensuring model reliability.

## Requirements

1. Create a `ModelMonitor` class that:
   - Tracks baseline statistics (from training data)
   - Compares current data to baseline
   - Detects drift using statistical tests
   - Generates alerts when drift detected

2. Implement drift detection methods:
   - Kolmogorov-Smirnov test for distributions
   - Chi-square test for categorical features
   - Monitoring prediction distributions

3. Create a simple dashboard/reporting system

## Solution Template

```python
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
        stats_dict = {}
        
        for col in self.baseline_data.columns:
            if pd.api.types.is_numeric_dtype(self.baseline_data[col]):
                stats_dict[col] = {
                    'mean': self.baseline_data[col].mean(),
                    'std': self.baseline_data[col].std(),
                    'min': self.baseline_data[col].min(),
                    'max': self.baseline_data[col].max(),
                    'distribution': self.baseline_data[col].values
                }
            else:
                # Categorical
                stats_dict[col] = {
                    'value_counts': self.baseline_data[col].value_counts().to_dict(),
                    'distribution': self.baseline_data[col].values
                }
        
        return stats_dict
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """
        Detect data drift in input features.
        
        Args:
            current_data: Current data to check
        
        Returns:
            List of drift alerts
        """
        alerts = []
        
        for col in self.baseline_data.columns:
            if col not in current_data.columns:
                continue
            
            baseline_dist = self.baseline_stats[col]['distribution']
            current_dist = current_data[col].dropna().values
            
            if len(current_dist) == 0:
                continue
            
            # Detect drift based on feature type
            if pd.api.types.is_numeric_dtype(self.baseline_data[col]):
                # Use Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline_dist, current_dist)
            else:
                # Chi-square test for categorical
                # Your implementation
                statistic, p_value = 0.0, 1.0
            
            if p_value < self.drift_threshold:
                severity = self._determine_severity(statistic, p_value)
                alert = DriftAlert(
                    feature=col,
                    drift_type='data_drift',
                    statistic=statistic,
                    p_value=p_value,
                    severity=severity,
                    detected_at=datetime.now()
                )
                alerts.append(alert)
        
        self.alerts.extend(alerts)
        return alerts
    
    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Optional[DriftAlert]:
        """Detect drift in prediction distribution"""
        if self.baseline_predictions is None:
            return None
        
        statistic, p_value = stats.ks_2samp(
            self.baseline_predictions, 
            current_predictions
        )
        
        if p_value < self.drift_threshold:
            severity = self._determine_severity(statistic, p_value)
            return DriftAlert(
                feature='predictions',
                drift_type='prediction_drift',
                statistic=statistic,
                p_value=p_value,
                severity=severity,
                detected_at=datetime.now()
            )
        return None
    
    def _determine_severity(self, statistic: float, p_value: float) -> str:
        """Determine severity of drift"""
        if p_value < 0.01:
            return 'high'
        elif p_value < 0.05:
            return 'medium'
        else:
            return 'low'
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        report = f"Model Monitoring Report\n"
        report += f"Generated at: {datetime.now()}\n"
        report += f"=" * 50 + "\n\n"
        
        if not self.alerts:
            report += "No drift detected.\n"
        else:
            report += f"Drift Alerts: {len(self.alerts)}\n\n"
            for alert in self.alerts:
                report += f"Feature: {alert.feature}\n"
                report += f"Type: {alert.drift_type}\n"
                report += f"Severity: {alert.severity}\n"
                report += f"P-value: {alert.p_value:.4f}\n"
                report += "-" * 30 + "\n"
        
        return report

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
```

## Key Learning Points

1. **Statistical Testing:** Understanding KS test, chi-square, etc.
2. **Production Monitoring:** What to track and alert on
3. **Drift Types:** Data vs concept vs prediction drift

## Design Considerations

- How often should monitoring run?
- What's an appropriate threshold?
- How to handle missing data?
- Should you track drift over time (trends)?

