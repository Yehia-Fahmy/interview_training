"""
Quick Test/Demo: Memory-Efficient Matrix Multiplication

Smaller test case to demonstrate the solution without generating huge matrices.
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
    print("SMALL TEST CASE - Memory-Efficient Matrix Multiplication")
    print("="*70)
    
    # Create test matrices
    create_small_test_matrices()
    
    # Initialize multiplier
    multiplier = MemoryEfficientMatrixMultiplier(memory_limit_mb=500.0)
    
    # Show memory estimates
    print("\nMemory Usage Estimates:")
    estimates = multiplier.estimate_memory_usage('test_matrix_a.csv', 'test_matrix_b.csv')
    print(f"Row-by-row approach: {estimates['row_by_row']['total_mb']:.6f} MB")
    
    # Perform multiplication
    print("\nPerforming matrix multiplication...")
    multiplier.multiply_matrices(
        matrix_a_file='test_matrix_a.csv',
        matrix_b_file='test_matrix_b.csv',
        output_file='test_matrix_result.csv',
        use_block_method=False
    )
    
    # Verify result
    print("\nVerifying result...")
    verify_result('test_matrix_a.csv', 'test_matrix_b.csv', 'test_matrix_result.csv')
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

