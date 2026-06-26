"""Main analysis pipeline: M1-M6 Advertising Impact Analysis"""

import pandas as pd
import numpy as np
from data_cleaning import load_and_clean_data
from statistics import compute_statistics
from probability import analyze_probability
from inference import perform_inference
from regression import build_models, RegressionAnalysis
from visualization import generate_visualizations
import os


def main():
    """Execute complete analysis pipeline."""
    
    print("\n" + "="*70)
    print("ADVERTISING IMPACT ANALYSIS - COMPLETE PIPELINE")
    print("M1-M6: Data Cleaning → Interpretation")
    print("="*70)
    
    # Dataset path
    data_path = 'data/advertising.csv'
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"\n✗ Dataset not found at {data_path}")
        print("Please download advertising.csv from:")
        print("https://www.kaggle.com/datasets/ashydv/advertising-dataset")
        return
    
    # M1: DATA CLEANING
    print("\n" + "-"*70)
    print("Starting M1: DATA CLEANING")
    print("-"*70)
    df_cleaned = load_and_clean_data(data_path)
    
    # M2: DESCRIPTIVE STATISTICS
    print("\n" + "-"*70)
    print("Starting M2: DESCRIPTIVE STATISTICS")
    print("-"*70)
    stats_summary = compute_statistics(df_cleaned)
    
    # M3: PROBABILITY ANALYSIS
    print("\n" + "-"*70)
    print("Starting M3: PROBABILITY ANALYSIS")
    print("-"*70)
    prob_analysis = analyze_probability(df_cleaned)
    
    # M4: STATISTICAL INFERENCE
    print("\n" + "-"*70)
    print("Starting M4: STATISTICAL INFERENCE")
    print("-"*70)
    inference_results = perform_inference(df_cleaned)
    
    # M5 & M6: REGRESSION ANALYSIS & INTERPRETATION
    print("\n" + "-"*70)
    print("Starting M5: REGRESSION ANALYSIS")
    print("-"*70)
    regression_obj = RegressionAnalysis(df_cleaned, target='Sales')
    regression_obj.run_regression_analysis()
    
    # VISUALIZATIONS
    print("\n" + "-"*70)
    print("Starting Visualization Generation")
    print("-"*70)
    generate_visualizations(df_cleaned, regression_obj.models['multiple'])
    
    # M6: INTERPRETATION & CONCLUSIONS
    print("\n" + "="*70)
    print("M6: INTERPRETATION & RESULTS SUMMARY")
    print("="*70)
    print_conclusions(regression_obj, stats_summary)
    
    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated Files:")
    print("  - Cleaned Data: output/cleaned_data.csv")
    print("  - Statistics Report: output/statistical_summary.txt")
    print("  - Regression Results: output/regression_results.txt")
    print("  - Visualizations: images/ (6 PNG files)")
    print("\n")


def print_conclusions(regression_obj, stats_summary):
    """Print key findings and conclusions."""
    
    multiple = regression_obj.models['multiple']
    
    print("\n=== KEY FINDINGS ===")
    print(f"\n1. Model Performance:")
    print(f"   - R² = {multiple['R²']:.4f} (Model explains {multiple['R²']*100:.2f}% of sales variance)")
    print(f"   - RMSE = {multiple['RMSE']:.4f}K units")
    print(f"   - MAE = {multiple['MAE']:.4f}K units")
    
    print(f"\n2. Regression Equation:")
    print(f"   {multiple['Equation']}")
    
    print(f"\n3. Coefficient Interpretation:")
    for col, coef in multiple['Coefficients'].items():
        print(f"   - {col}: {coef:.4f} (${coef:.4f}K sales per $1K ad spend)")
    
    print(f"\n4. Business Insights:")
    coeffs = multiple['Coefficients']
    max_coeff = max(coeffs.items(), key=lambda x: x[1])
    print(f"   - {max_coeff[0]} has the strongest impact on sales")
    print(f"   - Invest more in {max_coeff[0]} for better ROI")
    print(f"   - Model can predict sales within ±{multiple['MAE']:.2f}K units on average")
    
    print(f"\n5. Recommendations:")
    print(f"   ✓ Multiple linear regression model is recommended")
    print(f"   ✓ Focus advertising budget on high-impact channels")
    print(f"   ✓ Monitor actual vs predicted sales for model validation")


if __name__ == '__main__':
    main()
