"""
Test: Ultra Memory-Efficient Matrix Multiplication
(When Matrix B cannot fit in memory)
"""

import csv
from matrix_multiplication_memory_efficient import MemoryEfficientMatrixMultiplier


def create_small_test_matrices():
    """Create small test matrices for demonstration."""
    import random
    
    print("Creating small test matrices for demonstration...")
    
    # Small test: A (10 × 5), B (5 × 8), Result (10 × 8)
    rows_a, cols_a = 10, 5
    with open('test_matrix_a.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(rows_a):
            row = [random.random() for _ in range(cols_a)]
            writer.writerow(row)
    
    rows_b, cols_b = 5, 8
    with open('test_matrix_b.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(rows_b):
            row = [random.random() for _ in range(cols_b)]
            writer.writerow(row)
    
    print(f"Created test_matrix_a.csv ({rows_a} × {cols_a})")
    print(f"Created test_matrix_b.csv ({rows_b} × {cols_b})")


def verify_result(matrix_a_file: str, matrix_b_file: str, result_file: str):
    """Verify result using numpy (for small matrices only)."""
    try:
        import numpy as np
        
        # Load matrices
        a = np.loadtxt(matrix_a_file, delimiter=',')
        b = np.loadtxt(matrix_b_file, delimiter=',')
        result = np.loadtxt(result_file, delimiter=',')
        
        # Compute expected result
        expected = np.dot(a, b)
        
        # Check if results match (within floating point precision)
        if np.allclose(result, expected, rtol=1e-5):
            print("✓ Result verification passed! Output matches numpy.dot()")
            return True
        else:
            print("✗ Result verification failed!")
            print(f"Max difference: {np.max(np.abs(result - expected))}")
            return False
    except ImportError:
        print("NumPy not available, skipping verification")
        return None


if __name__ == "__main__":
    print("="*70)
    print("ULTRA MEMORY-EFFICIENT MATRIX MULTIPLICATION TEST")
    print("(When Matrix B cannot fit in memory)")
    print("="*70)
    
    # Create test matrices
    create_small_test_matrices()
    
    # Initialize multiplier
    multiplier = MemoryEfficientMatrixMultiplier(memory_limit_mb=500.0)
    
    # Show memory estimates for all methods
    print("\nMemory Usage Estimates:")
    estimates = multiplier.estimate_memory_usage('test_matrix_a.csv', 'test_matrix_b.csv', 
                                                   block_size=5, block_cols_b=4)
    
    print(f"\nNaive (all at once): {estimates['naive_all_at_once']['total_mb']:.6f} MB")
    print(f"Row-by-row (B fits in memory): {estimates['row_by_row']['total_mb']:.6f} MB")
    print(f"Ultra-efficient (B doesn't fit): {estimates['ultra_efficient']['total_mb']:.6f} MB")
    
    # Perform multiplication using ultra-efficient method
    print("\n" + "="*70)
    print("Performing matrix multiplication (ultra-efficient method)...")
    print("="*70)
    
    multiplier.multiply_matrices_ultra_efficient(
        matrix_a_file='test_matrix_a.csv',
        matrix_b_file='test_matrix_b.csv',
        output_file='test_matrix_result_ultra.csv',
        block_rows_a=5,  # Process 5 rows of A at a time
        block_cols_b=4   # Process 4 columns of B at a time
    )
    
    # Verify result
    print("\nVerifying result...")
    verify_result('test_matrix_a.csv', 'test_matrix_b.csv', 'test_matrix_result_ultra.csv')
    
    print("\n" + "="*70)
    print("Comparison: When to use which method?")
    print("="*70)
    print("\n1. Row-by-row method (multiply_matrices):")
    print("   - Use when: Transposed B can fit in memory (~120 MB)")
    print("   - Memory: ~120 MB")
    print("   - Speed: Faster (less I/O)")
    
    print("\n2. Ultra-efficient method (multiply_matrices_ultra_efficient):")
    print("   - Use when: Even transposed B cannot fit in memory")
    print("   - Memory: Configurable via block sizes")
    print("   - Speed: Slower (more I/O, but handles any size)")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

