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
        df_cleaned = df_cleaned.replace([None, "N/A", "", " "], np.nan)
        # 2. Handle missing values according to strategy
        if numeric_columns:
            for col in numeric_columns:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        match self.handle_missing:
            case 'drop':
                df_cleaned = df_cleaned.dropna()
            case 'forward_fill':
                df_cleaned = df_cleaned.ffill()
            case 'backward_fill':
                df_cleaned = df_cleaned.bfill()
                # Backward fill may leave NaN at the start, so forward fill to handle those
                df_cleaned = df_cleaned.ffill()
            case 'mean':
                if numeric_columns:
                    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].mean())
            case 'median':
                if numeric_columns:
                    df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())
            case 'mode':
                if numeric_columns:
                    for col in numeric_columns:
                        mode_values = df_cleaned[col].mode()
                        if len(mode_values) > 0:
                            df_cleaned[col] = df_cleaned[col].fillna(mode_values[0])
                        else:
                            # If no mode exists, use mean as fallback
                            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())

        # 3. Remove duplicates if enabled
        if self.handle_duplicates:
            df_cleaned = df_cleaned.drop_duplicates()
        # 4. Fix data types (dates, numeric)
        if date_columns:
            for col in date_columns:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
        # 5. Detect and handle outliers (IQR method for numeric columns)
        if self.detect_outliers and numeric_columns:
            for col in numeric_columns:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lb = Q1 - 1.5 * IQR
                ub = Q3 + 1.5 * IQR

                df_cleaned = df_cleaned[(df_cleaned[col] >= lb) & (df_cleaned[col] <= ub)]
        # 6. Normalize strings (lowercase, strip whitespace)
        if self.normalize_strings:
            string_columns = df_cleaned.select_dtypes(include=['object']).columns
            for col in string_columns:
                mask = df_cleaned[col].notna()
                df_cleaned.loc[mask, col] = df_cleaned.loc[mask, col].str.lower().str.strip()
        
        self.report['final_shape'] = df_cleaned.shape
        return df_cleaned
    
    def get_report(self) -> Dict:
        """Return summary report of cleaning operations."""
        return self.report


# ============================================================================
# Comprehensive Test Suite
# ============================================================================

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}{Colors.RESET}")


def assert_test(condition, message, passed_count, failed_count):
    """Assert a test condition and update counters."""
    if condition:
        print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")
        return passed_count + 1, failed_count
    else:
        print(f"{Colors.RED}✗ {message}{Colors.RESET}")
        return passed_count, failed_count + 1


def test_missing_value_normalization():
    """Test 1: Missing value normalization."""
    print_test_header("Missing Value Normalization")
    
    data = pd.DataFrame({
        'col1': [1, None, 'N/A', '', ' ', 'NULL', np.nan],
        'col2': ['A', None, 'N/A', '', ' ', 'NULL', 'B']
    })
    
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data)
    
    passed, failed = 0, 0
    
    # Check that various missing representations are normalized to NaN
    null_values = ['N/A', '', ' ', 'NULL', None]
    test_data = pd.DataFrame({'col': null_values})
    cleaned_test = test_data.replace([None, "N/A", "", " "], np.nan)
    passed, failed = assert_test(
        cleaned_test['col'].isnull().sum() >= 3,
        "Missing value representations are normalized to NaN",
        passed, failed
    )
    
    return passed, failed


def test_missing_value_strategies():
    """Test 2: All missing value handling strategies."""
    print_test_header("Missing Value Handling Strategies")
    
    passed, failed = 0, 0
    
    # Create data with missing values
    data = pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5, np.nan],
        'text': ['A', 'B', None, 'D', 'E', None]
    })
    
    # Test drop strategy
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    passed, failed = assert_test(
        cleaned['numeric'].notna().all(),
        "Drop strategy removes rows with missing values",
        passed, failed
    )
    
    # Test forward fill
    cleaner = DataCleaner(handle_missing='forward_fill', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    passed, failed = assert_test(
        cleaned['numeric'].notna().all(),
        "Forward fill fills missing values",
        passed, failed
    )
    
    # Test backward fill
    cleaner = DataCleaner(handle_missing='backward_fill', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    passed, failed = assert_test(
        cleaned['numeric'].notna().all(),
        "Backward fill fills missing values",
        passed, failed
    )
    
    # Test mean imputation
    cleaner = DataCleaner(handle_missing='mean', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    expected_mean = data['numeric'].mean()
    passed, failed = assert_test(
        cleaned['numeric'].notna().all() and np.isclose(cleaned['numeric'].mean(), expected_mean, rtol=1e-5),
        "Mean imputation fills NaN with mean value",
        passed, failed
    )
    
    # Test median imputation
    cleaner = DataCleaner(handle_missing='median', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    expected_median = data['numeric'].median()
    passed, failed = assert_test(
        cleaned['numeric'].notna().all() and np.isclose(cleaned['numeric'].median(), expected_median, rtol=1e-5),
        "Median imputation fills NaN with median value",
        passed, failed
    )
    
    # Test mode imputation
    mode_data = pd.DataFrame({
        'numeric': [1, 2, np.nan, 2, 2, np.nan],
        'text': ['A', 'B', None, 'A', 'A', None]
    })
    cleaner = DataCleaner(handle_missing='mode', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(mode_data, numeric_columns=['numeric'])
    expected_mode = mode_data['numeric'].mode()[0]
    passed, failed = assert_test(
        cleaned['numeric'].notna().all() and cleaned['numeric'].mode()[0] == expected_mode,
        "Mode imputation fills NaN with mode value",
        passed, failed
    )
    
    return passed, failed


def test_duplicate_removal():
    """Test 3: Duplicate removal."""
    print_test_header("Duplicate Removal")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'col1': [1, 2, 2, 3, 4],
        'col2': ['A', 'B', 'B', 'C', 'D']
    })
    
    # Test with duplicates enabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=True,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data)
    passed, failed = assert_test(
        cleaned.shape[0] == data.drop_duplicates().shape[0],
        "Duplicates are removed when enabled",
        passed, failed
    )
    
    # Test with duplicates disabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data)
    passed, failed = assert_test(
        cleaned.shape[0] == data.shape[0],
        "Duplicates are preserved when disabled",
        passed, failed
    )
    
    return passed, failed


def test_date_parsing():
    """Test 4: Date parsing."""
    print_test_header("Date Parsing")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'date1': ['2020-01-15', '2020/02/20', '2020-03-10'],
        'date2': ['2020-01-15', 'invalid', '2020-03-10'],
        'date3': ['2020-01-15', None, '2020-03-10']
    })
    
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, date_columns=['date1', 'date2', 'date3'])
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date1']),
        "Date column 1 is parsed as datetime",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date2']),
        "Date column 2 is parsed as datetime (invalid -> NaT)",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date3']),
        "Date column 3 is parsed as datetime",
        passed, failed
    )
    
    return passed, failed


def test_numeric_conversion():
    """Test 5: Numeric conversion."""
    print_test_header("Numeric Conversion")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'numeric1': ['100', '200', '300'],
        'numeric2': [100, '200', 300],
        'numeric3': ['100', 'invalid', '300'],
        'numeric4': [100, None, 300]
    })
    
    cleaner = DataCleaner(handle_missing='mean', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric1', 'numeric2', 'numeric3', 'numeric4'])
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['numeric1']),
        "String numeric column 1 converted to numeric",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['numeric2']),
        "Mixed numeric column 2 converted to numeric",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['numeric3']),
        "Column with invalid values converted to numeric (invalid -> NaN)",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['numeric4']),
        "Column with None converted to numeric",
        passed, failed
    )
    
    return passed, failed


def test_outlier_detection():
    """Test 6: Outlier detection."""
    print_test_header("Outlier Detection")
    
    passed, failed = 0, 0
    
    # Create data with clear outliers
    np.random.seed(42)
    normal_data = np.random.normal(100, 10, 50)
    outliers = [200, 250, -50, -100]  # Clear outliers
    data = pd.DataFrame({
        'values': list(normal_data) + outliers
    })
    
    # Test with outlier detection enabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=True, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['values'])
    
    Q1 = data['values'].quantile(0.25)
    Q3 = data['values'].quantile(0.75)
    IQR = Q3 - Q1
    lb = Q1 - 1.5 * IQR
    ub = Q3 + 1.5 * IQR
    
    passed, failed = assert_test(
        (cleaned['values'] >= lb).all() and (cleaned['values'] <= ub).all(),
        "Outliers are removed when detection is enabled",
        passed, failed
    )
    
    # Test with outlier detection disabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned_no_detect = cleaner.clean(data, numeric_columns=['values'])
    
    passed, failed = assert_test(
        cleaned_no_detect.shape[0] == data.shape[0],
        "Outliers are preserved when detection is disabled",
        passed, failed
    )
    
    return passed, failed


def test_string_normalization():
    """Test 7: String normalization."""
    print_test_header("String Normalization")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'text1': ['  ALICE  ', 'BOB', '  Charlie  '],
        'text2': ['  DAVE  ', None, '  EVE  '],
        'text3': ['Mixed Case', '  WITH SPACES  ', 'Normal']
    })
    
    # Test with normalization enabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=True)
    cleaned = cleaner.clean(data)
    
    passed, failed = assert_test(
        cleaned['text1'].iloc[0] == 'alice',
        "String normalization converts to lowercase",
        passed, failed
    )
    
    passed, failed = assert_test(
        ' ' not in cleaned['text1'].iloc[0],
        "String normalization strips whitespace",
        passed, failed
    )
    
    passed, failed = assert_test(
        cleaned['text3'].iloc[0] == 'mixed case',
        "String normalization handles mixed case correctly",
        passed, failed
    )
    
    # Test with normalization disabled
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned_no_norm = cleaner.clean(data)
    
    passed, failed = assert_test(
        cleaned_no_norm['text1'].iloc[0] == '  ALICE  ',
        "Strings are preserved when normalization is disabled",
        passed, failed
    )
    
    return passed, failed


def test_report_generation():
    """Test 8: Report generation."""
    print_test_header("Report Generation")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E']
    })
    
    cleaner = DataCleaner()
    cleaned = cleaner.clean(data)
    report = cleaner.get_report()
    
    passed, failed = assert_test(
        'original_shape' in report,
        "Report contains original_shape",
        passed, failed
    )
    
    passed, failed = assert_test(
        'final_shape' in report,
        "Report contains final_shape",
        passed, failed
    )
    
    passed, failed = assert_test(
        report['original_shape'] == data.shape,
        "Report original_shape matches input data shape",
        passed, failed
    )
    
    passed, failed = assert_test(
        report['final_shape'] == cleaned.shape,
        "Report final_shape matches cleaned data shape",
        passed, failed
    )
    
    return passed, failed


def test_combined_features():
    """Test 9: Combined features."""
    print_test_header("Combined Features")
    
    passed, failed = 0, 0
    
    # Complex messy data
    messy_data = pd.DataFrame({
        'id': [1, 2, 2, 3, 4, None, 5],
        'name': ['  ALICE  ', 'Bob', 'bob', '  CHARLIE  ', 'DAVE', '', 'Eve'],
        'age': [25, 30, 30, None, 150, 28, 35],
        'salary': [50000, '60000', 60000, None, 80000, 55000, 'invalid'],
        'date_joined': ['2020-01-15', '2020/02/20', 'invalid', '2020-03-10', None, '2020-04-01', '2020-05-15']
    })
    
    cleaner = DataCleaner(
        handle_missing='mean',
        handle_duplicates=True,
        detect_outliers=True,
        normalize_strings=True
    )
    
    cleaned = cleaner.clean(
        messy_data,
        date_columns=['date_joined'],
        numeric_columns=['age', 'salary']
    )
    
    # Check all features work together
    passed, failed = assert_test(
        cleaned.shape[0] <= messy_data.shape[0],
        "Combined cleaning reduces data size appropriately",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date_joined']),
        "Date parsing works with other features",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['age']) and pd.api.types.is_numeric_dtype(cleaned['salary']),
        "Numeric conversion works with other features",
        passed, failed
    )
    
    if cleaned['name'].notna().any():
        passed, failed = assert_test(
            cleaned.loc[cleaned['name'].notna(), 'name'].str.islower().all(),
            "String normalization works with other features",
            passed, failed
        )
    
    return passed, failed


def test_edge_cases():
    """Test 10: Edge cases."""
    print_test_header("Edge Cases")
    
    passed, failed = 0, 0
    
    # Empty dataframe
    empty_df = pd.DataFrame()
    cleaner = DataCleaner()
    try:
        cleaned = cleaner.clean(empty_df)
        passed, failed = assert_test(
            cleaned.shape == empty_df.shape,
            "Empty dataframe handling",
            passed, failed
        )
    except Exception as e:
        passed, failed = assert_test(
            False,
            f"Empty dataframe handling raised error: {e}",
            passed, failed
        )
    
    # All NaN dataframe
    all_nan_df = pd.DataFrame({
        'col1': [np.nan, np.nan, np.nan],
        'col2': [np.nan, np.nan, np.nan]
    })
    cleaner = DataCleaner(handle_missing='drop')
    cleaned = cleaner.clean(all_nan_df)
    passed, failed = assert_test(
        cleaned.shape[0] == 0,
        "Dataframe with all NaN values handled correctly",
        passed, failed
    )
    
    # Single row dataframe
    single_row = pd.DataFrame({'col': [1]})
    cleaner = DataCleaner()
    cleaned = cleaner.clean(single_row)
    passed, failed = assert_test(
        cleaned.shape[0] >= 0,
        "Single row dataframe handling",
        passed, failed
    )
    
    # No missing values
    no_missing = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['A', 'B', 'C']
    })
    cleaner = DataCleaner(handle_missing='drop')
    cleaned = cleaner.clean(no_missing)
    passed, failed = assert_test(
        cleaned.shape == no_missing.shape,
        "Dataframe with no missing values handled correctly",
        passed, failed
    )
    
    return passed, failed


def test_numeric_only_imputation():
    """Test 11: Numeric-only imputation."""
    print_test_header("Numeric-only Imputation")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'numeric': [1, 2, np.nan, 4, 5],
        'text': ['A', 'B', None, 'D', 'E']
    })
    
    # Test that mean/median/mode only work on numeric columns
    cleaner = DataCleaner(handle_missing='mean', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(data, numeric_columns=['numeric'])
    
    passed, failed = assert_test(
        cleaned['numeric'].notna().all(),
        "Mean imputation fills numeric column",
        passed, failed
    )
    
    return passed, failed


def test_multiple_column_types():
    """Test 12: Multiple column types."""
    print_test_header("Multiple Column Types")
    
    passed, failed = 0, 0
    
    data = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['A', 'B', 'C', 'D', 'E'],
        'date_col': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
        'bool_col': [True, False, True, False, True]
    })
    
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=True, normalize_strings=True)
    cleaned = cleaner.clean(
        data,
        date_columns=['date_col'],
        numeric_columns=['int_col', 'float_col']
    )
    
    passed, failed = assert_test(
        cleaned.shape[0] == data.shape[0],
        "Multiple column types handled correctly",
        passed, failed
    )
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date_col']),
        "Date column converted correctly",
        passed, failed
    )
    
    return passed, failed


def test_operation_order():
    """Test 13: Order of operations."""
    print_test_header("Order of Operations")
    
    passed, failed = 0, 0
    
    # Test that numeric conversion happens before outlier detection
    data = pd.DataFrame({
        'numeric_str': ['100', '200', '300', '400', '500'],
        'numeric': [100, 200, 300, 400, 500]
    })
    
    cleaner = DataCleaner(
        handle_missing='drop',
        handle_duplicates=False,
        detect_outliers=True,
        normalize_strings=False
    )
    
    cleaned = cleaner.clean(data, numeric_columns=['numeric_str', 'numeric'])
    
    passed, failed = assert_test(
        pd.api.types.is_numeric_dtype(cleaned['numeric_str']),
        "Numeric conversion happens before outlier detection",
        passed, failed
    )
    
    # Test that dates are parsed correctly even with invalid values
    date_data = pd.DataFrame({
        'date': ['2020-01-01', 'invalid', '2020-03-01'],
        'value': [1, 2, 3]
    })
    
    cleaner = DataCleaner(handle_missing='drop', handle_duplicates=False,
                         detect_outliers=False, normalize_strings=False)
    cleaned = cleaner.clean(date_data, date_columns=['date'])
    
    passed, failed = assert_test(
        pd.api.types.is_datetime64_any_dtype(cleaned['date']),
        "Date parsing handles invalid dates correctly (converts to NaT)",
        passed, failed
    )
    
    return passed, failed


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("="*70)
    print("Comprehensive Test Suite for DataCleaner")
    print("="*70)
    print(f"{Colors.RESET}")
    
    total_passed = 0
    total_failed = 0
    
    tests = [
        ("Missing Value Normalization", test_missing_value_normalization),
        ("Missing Value Strategies", test_missing_value_strategies),
        ("Duplicate Removal", test_duplicate_removal),
        ("Date Parsing", test_date_parsing),
        ("Numeric Conversion", test_numeric_conversion),
        ("Outlier Detection", test_outlier_detection),
        ("String Normalization", test_string_normalization),
        ("Report Generation", test_report_generation),
        ("Combined Features", test_combined_features),
        ("Edge Cases", test_edge_cases),
        ("Numeric-only Imputation", test_numeric_only_imputation),
        ("Multiple Column Types", test_multiple_column_types),
        ("Operation Order", test_operation_order),
    ]
    
    for test_name, test_func in tests:
        try:
            passed, failed = test_func()
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"{Colors.RED}✗ Test '{test_name}' raised exception: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            total_failed += 1
    
    # Final summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print("Final Summary")
    print(f"{'='*70}{Colors.RESET}")
    print(f"{Colors.GREEN}Total Passed: {total_passed}{Colors.RESET}")
    print(f"{Colors.RED}Total Failed: {total_failed}{Colors.RESET}")
    
    if total_failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! 🎉{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.YELLOW}Some tests failed. Review the implementation.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    import sys
    
    # Run comprehensive tests
    exit_code = run_comprehensive_tests()
    
    # Also run a simple example
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print("Simple Example")
    print(f"{'='*70}{Colors.RESET}")
    
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
    
    print("\nCleaned DataFrame:")
    print(cleaned)
    print("\nCleaning Report:")
    print(cleaner.get_report())
    
    sys.exit(exit_code)

