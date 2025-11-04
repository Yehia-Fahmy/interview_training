"""
Solution for Exercise 3: A/B Testing Framework for ML Models

This file contains the reference solution.
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
        if random.random() < self.config.traffic_split:
            return 'A'
        else:
            return 'B'
    
    def record_result(self, variant: str, metrics: Dict[str, float]):
        """Record results for a variant"""
        if variant == 'A':
            for metric_name, value in metrics.items():
                if metric_name not in self.variant_a_metrics:
                    self.variant_a_metrics[metric_name] = []
                self.variant_a_metrics[metric_name].append(value)
        elif variant == 'B':
            for metric_name, value in metrics.items():
                if metric_name not in self.variant_b_metrics:
                    self.variant_b_metrics[metric_name] = []
                self.variant_b_metrics[metric_name].append(value)
    
    def calculate_sample_size(self, effect_size: float, power: float = 0.8) -> int:
        """
        Calculate minimum sample size needed.
        
        Args:
            effect_size: Minimum detectable effect size (e.g., 0.05 for 5% improvement)
            power: Statistical power (1 - probability of Type II error)
        
        Returns:
            Minimum sample size per variant
        """
        # Simplified calculation - in practice, use statsmodels.stats.power
        z_alpha = 1.96  # For alpha=0.05
        z_beta = 0.84  # For power=0.8
        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        return int(n)
    
    def analyze_results(self) -> ExperimentResult:
        """
        Perform statistical analysis and determine winner.
        
        Returns:
            ExperimentResult with analysis and recommendation
        """
        # Check if we have enough samples
        primary_a = self.variant_a_metrics.get(self.config.primary_metric, [])
        primary_b = self.variant_b_metrics.get(self.config.primary_metric, [])
        
        if (len(primary_a) < self.config.minimum_sample_size or 
            len(primary_b) < self.config.minimum_sample_size):
            return ExperimentResult(
                variant_a_metrics=self.variant_a_metrics,
                variant_b_metrics=self.variant_b_metrics,
                sample_size_a=len(primary_a),
                sample_size_b=len(primary_b),
                significance_tests={},
                recommendation="continue_experiment",
                conclusion="Not enough samples collected yet"
            )
        
        # Perform statistical tests
        significance_tests = {}
        
        for metric in set(list(self.variant_a_metrics.keys()) + list(self.variant_b_metrics.keys())):
            values_a = self.variant_a_metrics.get(metric, [])
            values_b = self.variant_b_metrics.get(metric, [])
            
            if len(values_a) == 0 or len(values_b) == 0:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Mann-Whitney U test (non-parametric alternative)
            u_stat, u_pvalue = stats.mannwhitneyu(values_a, values_b, 
                                                   alternative='two-sided')
            
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            
            significance_tests[metric] = {
                't_test': {'statistic': t_stat, 'p_value': p_value},
                'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
                'mean_a': mean_a,
                'mean_b': mean_b,
                'improvement': ((mean_b - mean_a) / mean_a) * 100 if mean_a > 0 else 0
            }
        
        # Determine winner
        if self.config.primary_metric in significance_tests:
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
        else:
            recommendation = "insufficient_data"
            conclusion = "Insufficient data for primary metric"
        
        return ExperimentResult(
            variant_a_metrics=self.variant_a_metrics,
            variant_b_metrics=self.variant_b_metrics,
            sample_size_a=len(primary_a),
            sample_size_b=len(primary_b),
            significance_tests=significance_tests,
            recommendation=recommendation,
            conclusion=conclusion
        )
    
    def generate_report(self, result: ExperimentResult) -> str:
        """Generate human-readable report"""
        report = f"A/B Test Report\n"
        report += f"=" * 50 + "\n\n"
        report += f"Variant A ({self.config.variant_a_name}): {result.sample_size_a} samples\n"
        report += f"Variant B ({self.config.variant_b_name}): {result.sample_size_b} samples\n\n"
        
        if result.significance_tests:
            for metric, test in result.significance_tests.items():
                report += f"{metric}:\n"
                report += f"  Mean A: {test['mean_a']:.4f}\n"
                report += f"  Mean B: {test['mean_b']:.4f}\n"
                report += f"  Improvement: {test['improvement']:.2f}%\n"
                report += f"  T-test p-value: {test['t_test']['p_value']:.4f}\n"
                report += f"  Significant: {test['t_test']['p_value'] < self.config.significance_level}\n\n"
        
        report += f"Recommendation: {result.recommendation}\n"
        report += f"Conclusion: {result.conclusion}\n"
        
        return report

