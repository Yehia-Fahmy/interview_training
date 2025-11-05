"""
Solution for Exercise 5: Data Cleaning and Preprocessing with Pandas

This file contains the reference solution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime


class DataCleaner:
    """
    Comprehensive data cleaning utility for pandas DataFrames.
    
    Handles missing values, duplicates, data types, outliers, and string normalization.
    """
    
    def __init__(self, 
                 handle_missing: str = 'drop',
                 handle_duplicates: bool = True,
                 detect_outliers: bool = True,
                 normalize_strings: bool = True):
        """
        Initialize data cleaner.
        
        Args:
            handle_missing: Strategy for missing values ('drop', 'forward_fill', 'backward_fill', 'mean', 'median', 'mode')
            handle_duplicates: Whether to remove duplicate rows
            detect_outliers: Whether to detect and handle outliers
            normalize_strings: Whether to normalize string columns
        """
        self.handle_missing = handle_missing
        self.handle_duplicates = handle_duplicates
        self.detect_outliers = detect_outliers
        self.normalize_strings = normalize_strings
        self.report = {}
    
    def clean(self, df: pd.DataFrame, 
              date_columns: Optional[List[str]] = None,
              numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean the dataframe.
        
        Args:
            df: Input dataframe to clean
            date_columns: List of column names that should be dates
            numeric_columns: List of column names that should be numeric
        
        Returns:
            Cleaned dataframe
        """
        df_cleaned = df.copy()
        self.report = {'original_shape': df.shape}
        date_columns = date_columns or []
        numeric_columns = numeric_columns or []
        
        # Step 1: Normalize missing value representations
        missing_values = ['N/A', 'n/a', 'NA', 'na', 'NULL', 'null', '', ' ']
        for col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].replace(missing_values, np.nan)
        
        self.report['missing_before'] = df_cleaned.isnull().sum().to_dict()
        
        # Step 2: Handle missing values
        if self.handle_missing == 'drop':
            df_cleaned = df_cleaned.dropna()
        elif self.handle_missing == 'forward_fill':
            df_cleaned = df_cleaned.fillna(method='ffill')
        elif self.handle_missing == 'backward_fill':
            df_cleaned = df_cleaned.fillna(method='bfill')
        elif self.handle_missing in ['mean', 'median', 'mode']:
            for col in df_cleaned.select_dtypes(include=[np.number]).columns:
                if self.handle_missing == 'mean':
                    df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                elif self.handle_missing == 'median':
                    df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                elif self.handle_missing == 'mode':
                    mode_val = df_cleaned[col].mode()
                    if len(mode_val) > 0:
                        df_cleaned[col].fillna(mode_val[0], inplace=True)
            
            # For non-numeric columns, use mode
            for col in df_cleaned.select_dtypes(exclude=[np.number]).columns:
                mode_val = df_cleaned[col].mode()
                if len(mode_val) > 0:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)
        
        self.report['missing_after'] = df_cleaned.isnull().sum().to_dict()
        self.report['rows_dropped'] = self.report['original_shape'][0] - df_cleaned.shape[0]
        
        # Step 3: Remove duplicates
        if self.handle_duplicates:
            duplicates_before = df_cleaned.duplicated().sum()
            df_cleaned = df_cleaned.drop_duplicates()
            self.report['duplicates_removed'] = duplicates_before
        else:
            self.report['duplicates_removed'] = 0
        
        # Step 4: Fix data types
        # Convert date columns
        for col in date_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
        
        # Convert numeric columns
        for col in numeric_columns:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        
        # Step 5: Detect and handle outliers (IQR method)
        if self.detect_outliers:
            outliers_removed = {}
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                if outliers_count > 0:
                    # Cap outliers instead of removing
                    df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                    df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                    outliers_removed[col] = outliers_count
            
            self.report['outliers_capped'] = outliers_removed
        
        # Step 6: Normalize strings
        if self.normalize_strings:
            string_cols = df_cleaned.select_dtypes(include=['object']).columns
            for col in string_cols:
                if col not in date_columns:  # Don't normalize date columns
                    df_cleaned[col] = df_cleaned[col].astype(str).str.lower().str.strip()
                    df_cleaned[col] = df_cleaned[col].replace('nan', np.nan)
        
        self.report['final_shape'] = df_cleaned.shape
        return df_cleaned
    
    def get_report(self) -> Dict:
        """Return summary report of cleaning operations."""
        return self.report


# Test
if __name__ == "__main__":
    # Create messy sample data
    messy_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, None, 5],
        'name': ['Alice', 'Bob', 'bob', 'Charlie', 'DAVE', '', 'Eve'],
        'age': [25, 30, 30, None, 150, 28, 35],
        'salary': [50000, '60000', 60000, None, 80000, 55000, 'invalid'],
        'date_joined': ['2020-01-15', '2020/02/20', 'invalid', '2020-03-10', None, '2020-04-01', '2020-05-15']
    })
    
    cleaner = DataCleaner(handle_missing='mean', handle_duplicates=True)
    cleaned = cleaner.clean(messy_data, 
                            date_columns=['date_joined'],
                            numeric_columns=['age', 'salary'])
    
    print("Cleaned DataFrame:")
    print(cleaned)
    print("\nCleaning Report:")
    print(cleaner.get_report())

