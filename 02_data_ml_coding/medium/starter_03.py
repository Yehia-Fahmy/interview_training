"""
Exercise 3: A/B Testing Framework for ML Models

Build an A/B testing framework for comparing ML models in production.
"""

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
        # TODO: Implement variant assignment
        # Use random.random() and traffic_split to assign
        pass
    
    def record_result(self, variant: str, metrics: Dict[str, float]):
        """Record results for a variant"""
        # TODO: Implement result recording
        # Store metrics in appropriate variant's metrics dict
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
        # TODO: Implement sample size calculation
        # Use power analysis (can use statsmodels.stats.power for this)
        # For now, return a simple approximation
        pass
    
    def analyze_results(self) -> ExperimentResult:
        """
        Perform statistical analysis and determine winner.
        
        Returns:
            ExperimentResult with analysis and recommendation
        """
        # TODO: Implement result analysis
        # Check if we have enough samples
        # Perform statistical tests (t-test, Mann-Whitney)
        # Determine winner based on significance and improvement
        # Return ExperimentResult
        pass
    
    def generate_report(self, result: ExperimentResult) -> str:
        """Generate human-readable report"""
        # TODO: Implement report generation
        # Include metrics, statistical tests, recommendation
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

