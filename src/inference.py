"""Module M4: Statistical Inference

Performs hypothesis testing, confidence intervals, and statistical inference.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, ttest_1samp, ttest_ind, f_oneway


class StatisticalInference:
    """Statistical inference operations."""

    def __init__(self, df):
        """Initialize with dataframe."""
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.confidence_level = 0.95
        self.alpha = 1 - self.confidence_level

    def confidence_intervals(self, confidence=0.95):
        """Calculate confidence intervals for means."""
        print("\n" + "="*60)
        print("M4: STATISTICAL INFERENCE")
        print("="*60)
        print(f"\n=== CONFIDENCE INTERVALS (95%) ===")
        
        ci_results = {}
        
        for col in self.numeric_cols:
            n = len(self.df[col])
            mean = self.df[col].mean()
            std_err = self.df[col].sem()  # Standard error of mean
            
            # t-score for 95% CI
            t_score = t.ppf(1 - (1 - confidence) / 2, n - 1)
            
            margin_of_error = t_score * std_err
            ci_lower = mean - margin_of_error
            ci_upper = mean + margin_of_error
            
            ci_results[col] = {
                'Mean': mean,
                'Std Error': std_err,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'Margin of Error': margin_of_error
            }
            
            print(f"\n{col}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  Margin of Error: ±{margin_of_error:.4f}")
        
        return ci_results

    def hypothesis_testing_one_sample(self, col, hypothesized_mean=None):
        """One-sample t-test."""
        print(f"\n=== ONE-SAMPLE T-TEST: {col} ===")
        
        if hypothesized_mean is None:
            hypothesized_mean = self.df[col].mean()
        
        t_stat, p_value = ttest_1samp(self.df[col], hypothesized_mean)
        
        print(f"H0: μ = {hypothesized_mean}")
        print(f"H1: μ ≠ {hypothesized_mean}")
        print(f"t-statistic: {t_stat:.6f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        return {'t_stat': t_stat, 'p_value': p_value}

    def correlation_significance_test(self):
        """Test significance of correlations with Sales."""
        print("\n=== CORRELATION SIGNIFICANCE TEST ===")
        
        if 'Sales' not in self.df.columns:
            print("Sales column not found")
            return None
        
        sales = self.df['Sales']
        results = {}
        
        for col in self.numeric_cols:
            if col == 'Sales':
                continue
            
            # Pearson correlation and p-value
            corr_coeff, p_value = stats.pearsonr(self.df[col], sales)
            
            results[col] = {
                'Correlation': corr_coeff,
                'P-Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            }
            
            print(f"\n{col} vs Sales:")
            print(f"  Correlation: {corr_coeff:.4f}")
            print(f"  P-Value: {p_value:.6f}")
            print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
        
        return results

    def mean_comparison_test(self, col, group_by=None):
        """Independent samples t-test or ANOVA."""
        print(f"\n=== MEAN COMPARISON TEST ===")
        print(f"Variable: {col}")
        
        # For this dataset, we'll split by median for demonstration
        if group_by is None:
            median = self.df[col].median()
            group1 = self.df[self.df[col] <= median][col]
            group2 = self.df[self.df[col] > median][col]
            
            t_stat, p_value = ttest_ind(group1, group2)
            
            print(f"\nGroup 1 (≤ median): Mean = {group1.mean():.4f}, N = {len(group1)}")
            print(f"Group 2 (> median): Mean = {group2.mean():.4f}, N = {len(group2)}")
            print(f"t-statistic: {t_stat:.6f}")
            print(f"p-value: {p_value:.6f}")
            print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
            
            return {'t_stat': t_stat, 'p_value': p_value}

    def run_inference(self):
        """Run complete statistical inference."""
        ci = self.confidence_intervals()
        corr_sig = self.correlation_significance_test()
        
        print("\n" + "="*60)
        print("✓ STATISTICAL INFERENCE COMPLETE")
        print("="*60)
        
        return {
            'confidence_intervals': ci,
            'correlation_significance': corr_sig
        }


def perform_inference(df):
    """Convenience function for statistical inference."""
    inference = StatisticalInference(df)
    return inference.run_inference()
