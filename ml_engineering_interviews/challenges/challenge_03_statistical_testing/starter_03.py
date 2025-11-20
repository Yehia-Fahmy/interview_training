"""
Challenge 3: Statistical Hypothesis Testing
Starter Code

Implement statistical hypothesis testing from scratch to compare model performance.
This is explicitly tested in EvenUp's coding interview.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from scipy import stats
import math


def one_sample_t_test(
    sample: np.ndarray,
    population_mean: float,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform one-sample t-test.
    
    Tests if sample mean is significantly different from population mean.
    
    Args:
        sample: Sample data (1D array)
        population_mean: Hypothesized population mean
        alpha: Significance level (default 0.05)
    
    Returns:
        Dictionary with:
            - t_statistic: t-value
            - p_value: p-value
            - degrees_of_freedom: df
            - reject_null: Whether to reject H0
            - confidence_interval: 95% CI for mean
    """
    # Steps:
    # 1. Calculate sample mean and standard deviation
    sample_mean = np.mean(sample)
    std_dev = np.std(sample, ddof=1)  # Use sample std (Bessel's correction: divide by n-1)
    # 2. Calculate standard error
    std_err = std_dev / np.sqrt(len(sample))
    # 3. Calculate t-statistic: t = (sample_mean - population_mean) / standard_error
    t = (sample_mean - population_mean) / std_err
    # 4. Calculate degrees of freedom: n - 1
    degrees_of_freedom = len(sample) - 1
    # 5. Calculate p-value using t-distribution
    p = 2 * stats.t.sf(abs(t), degrees_of_freedom)
    # 6. Calculate confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    margin_of_error = t_critical * std_err
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    # 7. Determine if we reject H0 (p < alpha)
    reject_h0 = p < alpha
    
    # Return results as dictionary
    return {
        't_statistic': t,
        'p_value': p,
        'degrees_of_freedom': degrees_of_freedom,
        'reject_null': reject_h0,
        'confidence_interval': confidence_interval
    }


def two_sample_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True
) -> Dict[str, float]:
    """
    Perform two-sample t-test (independent samples).
    
    Tests if means of two samples are significantly different.
    
    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level
        equal_var: Whether to assume equal variances (default True)
    
    Returns:
        Dictionary with t-statistic, p-value, degrees of freedom, etc.
    """
    # Steps:
    # 1. Calculate means and variances for both samples
    m1 = np.mean(sample1)
    var_1 = np.var(sample1, ddof=1)
    n1 = len(sample1)
    m2 = np.mean(sample2)
    var_2 = np.var(sample2, ddof=1)
    n2 = len(sample2)
    
    # 2. Calculate standard error (pooled if equal_var, separate if not)
    if equal_var:
        # Pooled variance t-test (assumes equal variances)
        pooled_var = ((n1 - 1) * var_1 + (n2 - 1) * var_2) / (n1 + n2 - 2)
        std_err = np.sqrt(pooled_var * (1/n1 + 1/n2))
        degrees_of_freedom = n1 + n2 - 2
    else:
        # Welch's t-test (unequal variances)
        std_err = np.sqrt(var_1/n1 + var_2/n2)
        # Welch-Satterthwaite degrees of freedom
        df_numerator = (var_1/n1 + var_2/n2)**2
        df_denominator = (var_1/n1)**2/(n1-1) + (var_2/n2)**2/(n2-1)
        degrees_of_freedom = df_numerator / df_denominator
    
    # 3. Calculate t-statistic
    t = (m1 - m2) / std_err
    
    # 4. Calculate p-value (two-tailed)
    p_value = 2 * stats.t.sf(abs(t), degrees_of_freedom)
    
    # 5. Calculate confidence interval for difference in means
    t_critical = stats.t.ppf(1 - alpha/2, degrees_of_freedom)
    margin_of_error = t_critical * std_err
    mean_diff = m1 - m2
    confidence_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)
    
    # 6. Determine if we reject H0
    reject_null = p_value < alpha

    # Return results as dictionary
    return {
        't_statistic': t,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom,
        'reject_null': reject_null,
        'confidence_interval': confidence_interval
    }


def paired_t_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Perform paired t-test.
    
    Tests if mean difference between paired samples is significantly different from zero.
    Useful for comparing same subjects under different conditions.
    
    Args:
        sample1: First set of paired measurements
        sample2: Second set of paired measurements (same length as sample1)
        alpha: Significance level
    
    Returns:
        Dictionary with test results
    """
    # TODO: Implement paired t-test
    # Steps:
    # 1. Calculate differences: d = sample1 - sample2
    # 2. Calculate mean and std of differences
    # 3. Calculate standard error of mean difference
    # 4. Calculate t-statistic: t = mean_diff / standard_error
    # 5. Calculate p-value and CI
    
    # Hint: This is essentially a one-sample t-test on the differences
    
    pass


def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Perform chi-square test for independence or goodness of fit.
    
    Args:
        observed: Observed frequencies (2D array for contingency table)
        expected: Expected frequencies (optional, calculated from observed if not provided)
    
    Returns:
        Dictionary with chi-square statistic, p-value, degrees of freedom
    """
    # TODO: Implement chi-square test
    # Steps:
    # 1. If expected not provided, calculate from observed (assuming independence)
    # 2. Calculate chi-square statistic: sum((observed - expected)^2 / expected)
    # 3. Calculate degrees of freedom: (rows - 1) * (cols - 1)
    # 4. Calculate p-value using chi-square distribution
    
    # Hint: Use scipy.stats.chi2 for chi-square distribution
    
    pass


def calculate_statistical_power(
    effect_size: float,
    sample_size: int,
    alpha: float = 0.05,
    test_type: str = 'two_sample'
) -> float:
    """
    Calculate statistical power for a given effect size and sample size.
    
    Statistical power = 1 - P(Type II error) = Probability of detecting an effect
    
    Args:
        effect_size: Cohen's d (standardized effect size)
        sample_size: Sample size per group (for two-sample) or total (for one-sample)
        alpha: Significance level
        test_type: 'one_sample', 'two_sample', or 'paired'
    
    Returns:
        Statistical power (0 to 1)
    """
    # TODO: Implement power calculation
    # Steps:
    # 1. Calculate non-centrality parameter based on effect size and sample size
    # 2. Find critical value for given alpha
    # 3. Calculate power as probability of exceeding critical value under H1
    
    # Hint: Use non-central t-distribution
    # For two-sample: ncp = effect_size * sqrt(n/2)
    # Power = 1 - CDF(non-central t, critical_value)
    
    pass


def calculate_required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test_type: str = 'two_sample'
) -> int:
    """
    Calculate required sample size to achieve desired power.
    
    Args:
        effect_size: Minimum effect size we want to detect (Cohen's d)
        power: Desired statistical power (default 0.8)
        alpha: Significance level
        test_type: Type of test
    
    Returns:
        Required sample size per group (or total for one-sample)
    """
    # TODO: Implement sample size calculation
    # This is the inverse of power calculation
    # Use iterative approach: try different sample sizes until power is achieved
    
    # Hint: Start with small n, increase until power >= desired_power
    
    pass


def compare_model_performance(
    model_a_scores: np.ndarray,
    model_b_scores: np.ndarray,
    metric_name: str = "accuracy",
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Compare performance of two models using appropriate statistical test.
    
    Args:
        model_a_scores: Performance scores for model A (e.g., accuracies on test sets)
        model_b_scores: Performance scores for model B
        metric_name: Name of the metric being compared
        alpha: Significance level
    
    Returns:
        Dictionary with:
            - test_used: Which test was used
            - statistic: Test statistic value
            - p_value: p-value
            - significant: Whether difference is significant
            - effect_size: Cohen's d
            - interpretation: Human-readable interpretation
    """
    # TODO: Implement model comparison
    # Steps:
    # 1. Determine appropriate test (t-test for continuous, chi-square for categorical)
    # 2. Check assumptions (normality, equal variance)
    # 3. Perform test
    # 4. Calculate effect size (Cohen's d)
    # 5. Provide interpretation
    
    # Hint: For accuracy scores, use t-test
    # For confusion matrix comparisons, use chi-square
    
    pass


def main():
    """Main function to test statistical testing implementation"""
    print("=" * 70)
    print("Challenge 3: Statistical Hypothesis Testing")
    print("=" * 70)
    
    # Example 1: One-sample t-test
    print("\n1. One-sample t-test example:")
    print("   Testing if sample mean is significantly different from 0.5")
    sample = np.random.normal(0.6, 0.1, 30)  # Mean=0.6, std=0.1, n=30
    # result = one_sample_t_test(sample, population_mean=0.5)
    # print(f"   t-statistic: {result['t_statistic']:.3f}")
    # print(f"   p-value: {result['p_value']:.4f}")
    # print(f"   Reject H0: {result['reject_null']}")
    
    # Example 2: Two-sample t-test
    print("\n2. Two-sample t-test example:")
    print("   Comparing two model accuracies")
    model_a = np.random.normal(0.85, 0.05, 50)  # Model A: mean=0.85
    model_b = np.random.normal(0.82, 0.05, 50)  # Model B: mean=0.82
    # result = two_sample_t_test(model_a, model_b)
    # print(f"   t-statistic: {result['t_statistic']:.3f}")
    # print(f"   p-value: {result['p_value']:.4f}")
    # print(f"   Significant difference: {result['reject_null']}")
    
    # Example 3: Paired t-test
    print("\n3. Paired t-test example:")
    print("   Comparing same models on same test sets")
    # result = paired_t_test(model_a, model_b)
    # print(f"   t-statistic: {result['t_statistic']:.3f}")
    # print(f"   p-value: {result['p_value']:.4f}")
    
    # Example 4: Chi-square test
    print("\n4. Chi-square test example:")
    print("   Testing independence in confusion matrices")
    # Create 2x2 contingency table
    observed = np.array([[45, 5], [10, 40]])  # Example confusion matrices
    # result = chi_square_test(observed)
    # print(f"   chi-square: {result['chi_square']:.3f}")
    # print(f"   p-value: {result['p_value']:.4f}")
    
    # Example 5: Power analysis
    print("\n5. Power analysis example:")
    effect_size = 0.5  # Medium effect
    # power = calculate_statistical_power(effect_size, sample_size=30)
    # print(f"   Power for effect_size={effect_size}, n=30: {power:.3f}")
    # required_n = calculate_required_sample_size(effect_size, power=0.8)
    # print(f"   Required sample size for power=0.8: {required_n}")
    
    # Example 6: Model comparison
    print("\n6. Model performance comparison:")
    # comparison = compare_model_performance(model_a, model_b)
    # print(f"   Test used: {comparison['test_used']}")
    # print(f"   p-value: {comparison['p_value']:.4f}")
    # print(f"   Effect size (Cohen's d): {comparison['effect_size']:.3f}")
    # print(f"   Interpretation: {comparison['interpretation']}")
    
    print("\n" + "=" * 70)
    print("TODO: Implement the functions above to complete the challenge")
    print("=" * 70)
    print("\nTips:")
    print("- Review t-test and chi-square theory")
    print("- Use scipy.stats for distributions, but calculate statistics yourself")
    print("- Handle edge cases (small samples, zero variance, etc.)")
    print("- Provide clear interpretations of results")


if __name__ == "__main__":
    main()

