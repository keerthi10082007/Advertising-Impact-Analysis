"""Module M1: Data Cleaning

Handles data loading, validation, missing values, outliers, and preprocessing.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os


class DataCleaner:
    """Handles data cleaning operations."""

    def __init__(self, filepath):
        """Initialize with dataset path."""
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None
        self.report = {}

    def load_data(self):
        """Load CSV dataset."""
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            print(f"✗ File not found: {self.filepath}")
            raise

    def inspect_data(self):
        """Initial data inspection."""
        print("\n=== DATA INSPECTION ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nFirst 5 rows:\n{self.df.head()}")
        print(f"\nBasic Info:\n{self.df.info()}")
        self.report['inspection'] = True

    def check_missing_values(self):
        """Check for missing values."""
        missing = self.df.isnull().sum()
        print(f"\n=== MISSING VALUES ===")
        if missing.sum() == 0:
            print("✓ No missing values found")
        else:
            print(f"Missing values:\n{missing}")
        self.report['missing_values'] = missing.to_dict()
        return missing

    def detect_outliers(self, method='iqr', threshold=1.5):
        """Detect outliers using IQR method."""
        print(f"\n=== OUTLIER DETECTION ({method.upper()}) ===")
        outliers = {}
        
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                print(f"Column '{col}': {outlier_count} outliers ({outliers[col]['percentage']:.2f}%)")
        
        if not outliers:
            print("✓ No significant outliers detected")
        
        self.report['outliers'] = outliers
        return outliers

    def validate_data_types(self):
        """Validate and convert data types."""
        print("\n=== DATA TYPE VALIDATION ===")
        self.df_cleaned = self.df.copy()
        
        # Ensure all columns are numeric
        for col in self.df_cleaned.columns:
            try:
                self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col])
                print(f"✓ {col}: numeric")
            except ValueError:
                print(f"✗ {col}: non-numeric")
        
        self.report['data_types'] = self.df_cleaned.dtypes.to_dict()
        return self.df_cleaned

    def handle_duplicates(self):
        """Check and remove duplicates."""
        print("\n=== DUPLICATE CHECK ===")
        dup_count = self.df_cleaned.duplicated().sum()
        
        if dup_count > 0:
            print(f"Found {dup_count} duplicate rows. Removing...")
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            print(f"✓ Duplicates removed. New shape: {self.df_cleaned.shape}")
        else:
            print("✓ No duplicates found")
        
        self.report['duplicates'] = dup_count
        return self.df_cleaned

    def normalize_data(self):
        """Normalize numerical features (optional)."""
        print("\n=== DATA NORMALIZATION ===")
        df_normalized = self.df_cleaned.copy()
        
        for col in df_normalized.select_dtypes(include=[np.number]).columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        print("✓ Data normalized (Min-Max scaling)")
        return df_normalized

    def clean_pipeline(self):
        """Execute complete cleaning pipeline."""
        print("\n" + "="*50)
        print("M1: DATA CLEANING PIPELINE")
        print("="*50)
        
        self.load_data()
        self.inspect_data()
        self.check_missing_values()
        self.detect_outliers()
        self.validate_data_types()
        self.handle_duplicates()
        
        print("\n" + "="*50)
        print("✓ CLEANING COMPLETE")
        print("="*50)
        
        return self.df_cleaned

    def save_cleaned_data(self, output_path='output/cleaned_data.csv'):
        """Save cleaned dataset."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.df_cleaned.to_csv(output_path, index=False)
        print(f"✓ Cleaned data saved: {output_path}")


def load_and_clean_data(filepath):
    """Convenience function to load and clean data."""
    cleaner = DataCleaner(filepath)
    df_cleaned = cleaner.clean_pipeline()
    return df_cleaned
