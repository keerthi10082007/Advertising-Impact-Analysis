"""Module M3: Probability Analysis

Analyzes probability distributions and performs probability calculations.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, shapiro, kstest


class ProbabilityAnalysis:
    """Probability distribution analysis."""

    def __init__(self, df):
        """Initialize with dataframe."""
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns

    def normality_test(self):
        """Test for normality using Shapiro-Wilk test."""
        print("\n" + "="*60)
        print("M3: PROBABILITY ANALYSIS")
        print("="*60)
        print("\n=== NORMALITY TESTS (Shapiro-Wilk) ===")
        
        normality_results = {}
        
        for col in self.numeric_cols:
            stat, p_value = shapiro(self.df[col])
            is_normal = "Yes" if p_value > 0.05 else "No"
            
            normality_results[col] = {
                'Statistic': stat,
                'P-Value': p_value,
                'Normal (α=0.05)': is_normal
            }
            
            print(f"\n{col}:")
            print(f"  Statistic: {stat:.6f}")
            print(f"  P-Value: {p_value:.6f}")
            print(f"  Normal: {is_normal}")
        
        return normality_results

    def fit_distributions(self):
        """Fit normal distribution to each column."""
        print("\n=== DISTRIBUTION FITTING ===")
        
        fit_results = {}
        
        for col in self.numeric_cols:
            # Fit normal distribution
            mu, sigma = stats.norm.fit(self.df[col])
            
            fit_results[col] = {
                'Mean (μ)': mu,
                'Std Dev (σ)': sigma,
                'Distribution': 'Normal'
            }
            
            print(f"\n{col} - Normal Distribution:")
            print(f"  μ (mean) = {mu:.4f}")
            print(f"  σ (std dev) = {sigma:.4f}")
        
        return fit_results

    def calculate_probabilities(self):
        """Calculate probabilities for key statistics."""
        print("\n=== PROBABILITY CALCULATIONS ===")
        
        prob_results = {}
        
        for col in self.numeric_cols:
            mu = self.df[col].mean()
            sigma = self.df[col].std()
            
            # P(X > mean)
            p_above_mean = 1 - norm.cdf(mu, mu, sigma)
            
            # P(X < mean)
            p_below_mean = norm.cdf(mu, mu, sigma)
            
            # P(mean - σ < X < mean + σ)
            p_within_1sigma = norm.cdf(mu + sigma, mu, sigma) - norm.cdf(mu - sigma, mu, sigma)
            
            prob_results[col] = {
                'P(X > μ)': p_above_mean,
                'P(X < μ)': p_below_mean,
                'P(μ-σ < X < μ+σ)': p_within_1sigma
            }
            
            print(f"\n{col}:")
            print(f"  P(X > mean) = {p_above_mean:.4f}")
            print(f"  P(X < mean) = {p_below_mean:.4f}")
            print(f"  P(within 1σ) = {p_within_1sigma:.4f}")
        
        return prob_results

    def quantile_analysis(self):
        """Analyze quantiles (percentiles)."""
        print("\n=== QUANTILE ANALYSIS ===")
        
        quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        quantile_results = {}
        
        for col in self.numeric_cols:
            quantile_values = [self.df[col].quantile(q) for q in quantiles]
            quantile_results[col] = dict(zip([f"{int(q*100)}%" for q in quantiles], quantile_values))
            
            print(f"\n{col}:")
            for q, val in zip(quantiles, quantile_values):
                print(f"  {int(q*100)}th percentile: {val:.4f}")
        
        return quantile_results

    def run_analysis(self):
        """Run complete probability analysis."""
        normality = self.normality_test()
        distributions = self.fit_distributions()
        probabilities = self.calculate_probabilities()
        quantiles = self.quantile_analysis()
        
        print("\n" + "="*60)
        print("✓ PROBABILITY ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'normality': normality,
            'distributions': distributions,
            'probabilities': probabilities,
            'quantiles': quantiles
        }


def analyze_probability(df):
    """Convenience function for probability analysis."""
    prob_analysis = ProbabilityAnalysis(df)
    return prob_analysis.run_analysis()
