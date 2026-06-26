"""Visualization utilities for exploratory data analysis and model results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


class Visualizer:
    """Handles all data visualizations."""

    def __init__(self, df, figsize=(12, 8), style='whitegrid'):
        """Initialize visualizer."""
        self.df = df
        self.figsize = figsize
        sns.set_style(style)
        sns.set_palette("husl")
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        os.makedirs('images', exist_ok=True)

    def histograms(self):
        """Create histograms for all numeric variables."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Distribution of Variables', fontsize=16, fontweight='bold')
        
        axes = axes.ravel()
        for idx, col in enumerate(self.numeric_cols):
            axes[idx].hist(self.df[col], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{col} Distribution', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/01_histograms.png', dpi=300, bbox_inches='tight')
        print("✓ Histograms saved: images/01_histograms.png")
        plt.close()

    def boxplots(self):
        """Create boxplots for outlier detection."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle('Boxplots - Outlier Detection', fontsize=14, fontweight='bold')
        
        for idx, col in enumerate(self.numeric_cols):
            axes[idx].boxplot(self.df[col], vert=True)
            axes[idx].set_title(col, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/02_boxplots.png', dpi=300, bbox_inches='tight')
        print("✓ Boxplots saved: images/02_boxplots.png")
        plt.close()

    def scatter_plots(self):
        """Create scatter plots for relationships with Sales."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle('Relationship with Sales', fontsize=14, fontweight='bold')
        
        predictors = [col for col in self.numeric_cols if col != 'Sales']
        
        for idx, col in enumerate(predictors):
            axes[idx].scatter(self.df[col], self.df['Sales'], alpha=0.6, s=50, color='steelblue')
            axes[idx].set_xlabel(col, fontweight='bold')
            axes[idx].set_ylabel('Sales', fontweight='bold')
            axes[idx].set_title(f'{col} vs Sales')
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/03_scatter_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Scatter plots saved: images/03_scatter_plots.png")
        plt.close()

    def correlation_heatmap(self):
        """Create correlation heatmap."""
        plt.figure(figsize=(10, 8))
        corr_matrix = self.df[self.numeric_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved: images/04_correlation_heatmap.png")
        plt.close()

    def pairplot(self):
        """Create pairplot."""
        fig = sns.pairplot(self.df[self.numeric_cols], diag_kind='hist', 
                           plot_kws={'alpha': 0.6}, diag_kws={'bins': 30, 'edgecolor': 'black'})
        fig.fig.suptitle('Pairplot - Variable Relationships', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig('images/05_pairplot.png', dpi=300, bbox_inches='tight')
        print("✓ Pairplot saved: images/05_pairplot.png")
        plt.close()

    def regression_plot(self, y_actual, y_pred, title='Regression Results'):
        """Create regression actual vs predicted plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_actual, y_pred, alpha=0.6, color='steelblue')
        axes[0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Sales', fontweight='bold')
        axes[0].set_ylabel('Predicted Sales', fontweight='bold')
        axes[0].set_title('Actual vs Predicted', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Residuals
        residuals = y_actual - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color='coral')
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Sales', fontweight='bold')
        axes[1].set_ylabel('Residuals', fontweight='bold')
        axes[1].set_title('Residual Plot', fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/06_regression_diagnostics.png', dpi=300, bbox_inches='tight')
        print("✓ Regression diagnostics saved: images/06_regression_diagnostics.png")
        plt.close()

    def generate_all(self, regression_results=None):
        """Generate all visualizations."""
        print("\n=== GENERATING VISUALIZATIONS ===")
        self.histograms()
        self.boxplots()
        self.scatter_plots()
        self.correlation_heatmap()
        self.pairplot()
        
        if regression_results is not None:
            y_pred = regression_results['Y_pred']
            y_actual = self.df['Sales'].values
            self.regression_plot(y_actual, y_pred)


def generate_visualizations(df, regression_results=None):
    """Convenience function to generate all visualizations."""
    visualizer = Visualizer(df)
    visualizer.generate_all(regression_results)
    print("\n✓ All visualizations saved to images/ folder")
