# Exercise 3: A/B Testing Framework for ML Models

**Difficulty:** Medium  
**Time Limit:** 60 minutes  
**Focus:** Experimentation, statistical testing, production ML

## Problem

Build an A/B testing framework for comparing ML models in production. This is essential for an Improvement Engineer who needs to validate model improvements.

## Requirements

1. Create a framework that:
   - Splits traffic between models (A/B or A/B/C)
   - Tracks metrics for each variant
   - Performs statistical significance testing
   - Determines winner based on criteria

2. Support:
   - Different traffic splits (50/50, 80/20, etc.)
   - Multiple metrics (accuracy, F1, latency, etc.)
   - Statistical tests (t-test, Mann-Whitney, etc.)
   - Minimum sample size calculations

3. Generate reports with:
   - Metric comparisons
   - Statistical significance
   - Recommendation (which model to use)

## Solution Template

```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from datetime import datetime
import random

@dataclass
class ExperimentConfig:
    """A/B test configuration"""
    variant_a_name: str = "control"
    variant_b_name: str = "treatment"
    traffic_split: float = 0.5  # 50/50 split
    minimum_sample_size: int = 1000
    significance_level: float = 0.05
    primary_metric: str = "accuracy"

@dataclass
class ExperimentResult:
    """Results of A/B test"""
    variant_a_metrics: Dict[str, List[float]]
    variant_b_metrics: Dict[str, List[float]]
    sample_size_a: int
    sample_size_b: int
    significance_tests: Dict[str, Dict]
    recommendation: str
    conclusion: str

class ABTestFramework:
    """
    A/B testing framework for ML models.
    
    Manages traffic splitting, metric collection, and statistical analysis.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.variant_a_results = []
        self.variant_b_results = []
        self.variant_a_metrics = {self.config.primary_metric: []}
        self.variant_b_metrics = {self.config.primary_metric: []}
    
    def assign_variant(self) -> str:
        """
        Assign request to variant A or B based on traffic split.
        
        Returns:
            'A' or 'B'
        """
        # Your implementation
        pass
    
    def record_result(self, variant: str, metrics: Dict[str, float]):
        """Record results for a variant"""
        # Your implementation
        pass
    
    def calculate_sample_size(self, effect_size: float, power: float = 0.8) -> int:
        """
        Calculate minimum sample size needed.
        
        Args:
            effect_size: Minimum detectable effect size (e.g., 0.05 for 5% improvement)
            power: Statistical power (1 - probability of Type II error)
        
        Returns:
            Minimum sample size per variant
        """
        # Use power analysis
        # Your implementation using statsmodels or scipy
        pass
    
    def analyze_results(self) -> ExperimentResult:
        """
        Perform statistical analysis and determine winner.
        
        Returns:
            ExperimentResult with analysis and recommendation
        """
        # Check if we have enough samples
        if (len(self.variant_a_metrics[self.config.primary_metric]) < 
            self.config.minimum_sample_size):
            return ExperimentResult(
                variant_a_metrics=self.variant_a_metrics,
                variant_b_metrics=self.variant_b_metrics,
                sample_size_a=len(self.variant_a_metrics[self.config.primary_metric]),
                sample_size_b=len(self.variant_b_metrics[self.config.primary_metric]),
                significance_tests={},
                recommendation="continue_experiment",
                conclusion="Not enough samples collected yet"
            )
        
        # Perform statistical tests
        significance_tests = {}
        
        for metric in self.variant_a_metrics.keys():
            values_a = self.variant_a_metrics[metric]
            values_b = self.variant_b_metrics[metric]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Mann-Whitney U test (non-parametric alternative)
            u_stat, u_pvalue = stats.mannwhitneyu(values_a, values_b, 
                                                   alternative='two-sided')
            
            significance_tests[metric] = {
                't_test': {'statistic': t_stat, 'p_value': p_value},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
                'mean_a': np.mean(values_a),
                'mean_b': np.mean(values_b),
                'improvement': ((np.mean(values_b) - np.mean(values_a)) / 
                               np.mean(values_a)) * 100
            }
        
        # Determine winner
        primary_test = significance_tests[self.config.primary_metric]
        is_significant = (primary_test['t_test']['p_value'] < 
                         self.config.significance_level)
        
        if is_significant:
            if primary_test['mean_b'] > primary_test['mean_a']:
                recommendation = "switch_to_b"
                conclusion = f"Variant B is significantly better ({primary_test['improvement']:.2f}% improvement)"
            else:
                recommendation = "keep_a"
                conclusion = "Variant A is significantly better"
        else:
            recommendation = "no_difference"
            conclusion = "No significant difference detected"
        
        return ExperimentResult(
            variant_a_metrics=self.variant_a_metrics,
            variant_b_metrics=self.variant_b_metrics,
            sample_size_a=len(self.variant_a_metrics[self.config.primary_metric]),
            sample_size_b=len(self.variant_b_metrics[self.config.primary_metric]),
            significance_tests=significance_tests,
            recommendation=recommendation,
            conclusion=conclusion
        )
    
    def generate_report(self, result: ExperimentResult) -> str:
        """Generate human-readable report"""
        # Your implementation
        pass

# Usage example
if __name__ == "__main__":
    config = ExperimentConfig(
        variant_a_name="current_model",
        variant_b_name="improved_model",
        traffic_split=0.5,
        minimum_sample_size=1000,
        primary_metric="accuracy"
    )
    
    framework = ABTestFramework(config)
    
    # Simulate experiment
    np.random.seed(42)
    for i in range(2000):
        variant = framework.assign_variant()
        
        if variant == 'A':
            # Simulate current model (lower performance)
            metrics = {'accuracy': np.random.normal(0.85, 0.02)}
        else:
            # Simulate improved model (higher performance)
            metrics = {'accuracy': np.random.normal(0.87, 0.02)}
        
        framework.record_result(variant, metrics)
    
    # Analyze
    result = framework.analyze_results()
    print(f"Recommendation: {result.recommendation}")
    print(f"Conclusion: {result.conclusion}")
    print(f"\nPrimary Metric Analysis:")
    primary = result.significance_tests['accuracy']
    print(f"Mean A: {primary['mean_a']:.4f}")
    print(f"Mean B: {primary['mean_b']:.4f}")
    print(f"P-value: {primary['t_test']['p_value']:.4f}")
```

## Key Learning Points

1. **Statistical Testing:** Understanding significance testing
2. **Experimental Design:** Traffic splitting, sample sizes
3. **Production Deployment:** Gradual rollouts

## Design Considerations

- How to handle early stopping?
- Should you track multiple metrics simultaneously?
- How to prevent selection bias in traffic splitting?
- What about sequential testing vs fixed sample size?

