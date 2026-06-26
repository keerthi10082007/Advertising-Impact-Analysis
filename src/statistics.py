"""Module M2: Descriptive Statistics

Computes and visualizes descriptive statistics and correlations.
"""

import pandas as pd
import numpy as np
from scipy import stats


class DescriptiveStatistics:
    """Compute descriptive statistics."""

    def __init__(self, df):
        """Initialize with dataframe."""
        self.df = df
        self.stats_summary = None

    def compute_basic_stats(self):
        """Compute mean, median, std, min, max, quartiles."""
        print("\n" + "="*60)
        print("M2: DESCRIPTIVE STATISTICS")
        print("="*60)
        
        stats_data = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats_data[col] = {
                'Count': self.df[col].count(),
                'Mean': self.df[col].mean(),
                'Median': self.df[col].median(),
                'Std Dev': self.df[col].std(),
                'Min': self.df[col].min(),
                'Q1 (25%)': self.df[col].quantile(0.25),
                'Q2 (50%)': self.df[col].quantile(0.50),
                'Q3 (75%)': self.df[col].quantile(0.75),
                'Max': self.df[col].max(),
                'Range': self.df[col].max() - self.df[col].min(),
                'IQR': self.df[col].quantile(0.75) - self.df[col].quantile(0.25)
            }
        
        self.stats_summary = pd.DataFrame(stats_data).T
        print("\n" + self.stats_summary.to_string())
        return self.stats_summary

    def compute_skewness_kurtosis(self):
        """Compute skewness and kurtosis."""
        print("\n=== SKEWNESS & KURTOSIS ===")
        
        sk_data = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            skewness = stats.skew(self.df[col])
            kurtosis = stats.kurtosis(self.df[col])
            
            sk_data[col] = {
                'Skewness': skewness,
                'Kurtosis': kurtosis,
                'Distribution': 'Symmetric' if abs(skewness) < 0.5 else 
                               'Right-skewed' if skewness > 0.5 else 'Left-skewed'
            }
            print(f"\n{col}:")
            print(f"  Skewness: {skewness:.4f}")
            print(f"  Kurtosis: {kurtosis:.4f}")
            print(f"  Type: {sk_data[col]['Distribution']}")
        
        return sk_data

    def correlation_analysis(self):
        """Compute correlation matrix."""
        print("\n=== CORRELATION MATRIX ===")
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        print("\n" + corr_matrix.to_string())
        return corr_matrix

    def correlation_with_target(self, target='Sales'):
        """Compute correlation with target variable."""
        print(f"\n=== CORRELATION WITH TARGET ('{target}') ===")
        
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if target in numeric_df.columns:
            target_corr = numeric_df.corr()[target].sort_values(ascending=False)
            print("\n" + target_corr.to_string())
            return target_corr
        else:
            print(f"✗ Target '{target}' not found in columns")
            return None

    def generate_report(self, output_file='output/statistical_summary.txt'):
        """Generate comprehensive statistics report."""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ADVERTISING IMPACT ANALYSIS - STATISTICAL SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(self.stats_summary.to_string())
            
            f.write("\n\nCORRELATION MATRIX\n")
            f.write("-"*70 + "\n")
            corr = self.correlation_analysis()
            f.write(corr.to_string())
            
            f.write("\n\nCORRELATION WITH SALES\n")
            f.write("-"*70 + "\n")
            corr_sales = self.correlation_with_target('Sales')
            f.write(corr_sales.to_string())
        
        print(f"\n✓ Report saved: {output_file}")


def compute_statistics(df):
    """Convenience function for statistics computation."""
    stats_obj = DescriptiveStatistics(df)
    stats_obj.compute_basic_stats()
    stats_obj.compute_skewness_kurtosis()
    stats_obj.correlation_analysis()
    stats_obj.correlation_with_target('Sales')
    stats_obj.generate_report()
    return stats_obj.stats_summary
