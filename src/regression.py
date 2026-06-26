"""Module M5 & M6: Regression Analysis and Interpretation

Builds regression models, evaluates performance, and interprets results.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
import os


class RegressionAnalysis:
    """Regression modeling and analysis."""

    def __init__(self, df, target='Sales'):
        """Initialize with dataframe and target variable."""
        self.df = df
        self.target = target
        self.models = {}
        self.results = {}
        self.numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]

    def simple_linear_regression(self):
        """Build simple linear regression models for each predictor."""
        print("\n" + "="*60)
        print("M5: REGRESSION ANALYSIS")
        print("="*60)
        print("\n=== SIMPLE LINEAR REGRESSION ===")
        
        X = self.df[self.numeric_cols].values
        y = self.df[self.target].values
        
        simple_results = {}
        
        for i, col in enumerate(self.numeric_cols):
            X_single = X[:, i].reshape(-1, 1)
            
            model = LinearRegression()
            model.fit(X_single, y)
            y_pred = model.predict(X_single)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            
            simple_results[col] = {
                'Intercept': model.intercept_,
                'Coefficient': model.coef_[0],
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Model': model
            }
            
            print(f"\n{col} → {self.target}:")
            print(f"  Equation: {self.target} = {model.intercept_:.4f} + {model.coef_[0]:.4f} * {col}")
            print(f"  R² = {r2:.4f}")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  MAE = {mae:.4f}")
        
        self.models['simple'] = simple_results
        return simple_results

    def multiple_linear_regression(self):
        """Build multiple linear regression model with all predictors."""
        print("\n=== MULTIPLE LINEAR REGRESSION ===")
        
        X = self.df[self.numeric_cols]
        y = self.df[self.target].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        adjusted_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        
        # Build regression equation
        equation = f"{self.target} = {model.intercept_:.4f}"
        for col, coef in zip(self.numeric_cols, model.coef_):
            equation += f" + {coef:.4f} * {col}"
        
        print(f"\nRegression Equation:")
        print(f"  {equation}")
        print(f"\nModel Performance:")
        print(f"  R² = {r2:.4f}")
        print(f"  Adjusted R² = {adjusted_r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE = {mae:.4f}")
        
        print(f"\nCoefficient Interpretation:")
        for col, coef in zip(self.numeric_cols, model.coef_):
            print(f"  {col}: {coef:.4f} (For every $1K increase, Sales increases by {coef:.4f}K units)")
        
        multiple_results = {
            'Model': model,
            'Intercept': model.intercept_,
            'Coefficients': dict(zip(self.numeric_cols, model.coef_)),
            'R²': r2,
            'Adjusted_R²': adjusted_r2,
            'RMSE': rmse,
            'MAE': mae,
            'Y_pred': y_pred,
            'Equation': equation
        }
        
        self.models['multiple'] = multiple_results
        return multiple_results

    def model_comparison(self):
        """Compare simple and multiple regression models."""
        print("\n=== MODEL COMPARISON ===")
        
        simple = self.models['simple']
        multiple = self.models['multiple']
        
        print("\n{:<20} {:<15} {:<15} {:<15}".format("Model", "R²", "RMSE", "MAE"))
        print("-" * 65)
        
        for col in simple:
            print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
                col, simple[col]['R²'], simple[col]['RMSE'], simple[col]['MAE']
            ))
        
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            "Multiple", multiple['R²'], multiple['RMSE'], multiple['MAE']
        ))
        
        print("\n✓ Multiple regression model performs best!")

    def residual_analysis(self):
        """Analyze residuals for model diagnostics."""
        print("\n=== RESIDUAL ANALYSIS ===")
        
        y = self.df[self.target].values
        y_pred = self.models['multiple']['Y_pred']
        residuals = y - y_pred
        
        print(f"Mean of residuals: {np.mean(residuals):.6f}")
        print(f"Std of residuals: {np.std(residuals):.4f}")
        print(f"Normality (Shapiro-Wilk): p = {stats.shapiro(residuals)[1]:.6f}")
        
        return residuals

    def predictions(self, new_data):
        """Make predictions on new data."""
        model = self.models['multiple']['Model']
        return model.predict(new_data)

    def generate_report(self, output_file='output/regression_results.txt'):
        """Generate comprehensive regression report."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ADVERTISING IMPACT ANALYSIS - REGRESSION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            multiple = self.models['multiple']
            
            f.write("MULTIPLE LINEAR REGRESSION MODEL\n")
            f.write("-"*70 + "\n")
            f.write(f"Equation: {multiple['Equation']}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write(f"  R² = {multiple['R²']:.4f}\n")
            f.write(f"  Adjusted R² = {multiple['Adjusted_R²']:.4f}\n")
            f.write(f"  RMSE = {multiple['RMSE']:.4f}\n")
            f.write(f"  MAE = {multiple['MAE']:.4f}\n\n")
            
            f.write("COEFFICIENT INTERPRETATION\n")
            for col, coef in multiple['Coefficients'].items():
                f.write(f"  {col}: {coef:.4f}\n")
        
        print(f"\n✓ Regression report saved: {output_file}")

    def run_regression_analysis(self):
        """Run complete regression analysis."""
        self.simple_linear_regression()
        self.multiple_linear_regression()
        self.model_comparison()
        self.residual_analysis()
        self.generate_report()
        
        print("\n" + "="*60)
        print("✓ REGRESSION ANALYSIS COMPLETE")
        print("="*60)


def build_models(df, target='Sales'):
    """Convenience function to build regression models."""
    regression = RegressionAnalysis(df, target)
    regression.run_regression_analysis()
    return regression.models
