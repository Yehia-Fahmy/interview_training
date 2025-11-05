"""
Exercise 5: Data Cleaning and Preprocessing with Pandas

Given a messy dataset with various data quality issues, create a robust data cleaning 
function that handles missing values, duplicates, data types, outliers, and string normalization.
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
        
        # TODO: Implement cleaning steps
        # 1. Normalize missing value representations (None, 'N/A', '', etc. -> NaN)
        # 2. Handle missing values according to strategy
        # 3. Remove duplicates if enabled
        # 4. Fix data types (dates, numeric)
        # 5. Detect and handle outliers (IQR method for numeric columns)
        # 6. Normalize strings (lowercase, strip whitespace)
        
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

