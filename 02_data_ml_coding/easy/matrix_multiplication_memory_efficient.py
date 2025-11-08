"""
Memory-Efficient Matrix Multiplication

Problem: Multiply two large matrices from CSV files with memory constraint < 500MB
- Matrix A: 8000 × 5000
- Matrix B: 5000 × 3000
- Result C: 8000 × 3000
- Each element is a float (8 bytes)

Memory Analysis:
- Matrix A: 8000 × 5000 × 8 bytes = 320 MB
- Matrix B: 5000 × 3000 × 8 bytes = 120 MB
- Result C: 8000 × 3000 × 8 bytes = 192 MB
- Total if loaded all at once: ~632 MB (exceeds 500MB limit)

Solution: Block-based matrix multiplication with streaming CSV reading
"""

import csv
import os
from typing import Iterator, List
import sys


class MemoryEfficientMatrixMultiplier:
    """
    Performs matrix multiplication while respecting memory constraints.
    
    Strategy:
    1. Process matrices in blocks/chunks
    2. Stream data from CSV files row-by-row or block-by-block
    3. Compute partial results incrementally
    4. Write results to output CSV incrementally
    
    Memory efficiency:
    - Only loads small chunks of data at a time
    - Processes one row of A with all columns of B at a time
    - Or processes blocks of A with corresponding blocks of B
    """
    
    def __init__(self, memory_limit_mb: float = 500.0):
        """
        Initialize the matrix multiplier.
        
        Args:
            memory_limit_mb: Maximum memory usage in MB
        """
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
    
    def get_matrix_dimensions(self, csv_file: str) -> tuple:
        """
        Get dimensions of a matrix stored in CSV without loading it all.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            (rows, cols) tuple
        """
        rows = 0
        cols = 0
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            first_row = next(reader)
            cols = len(first_row)
            rows = 1
            
            for _ in reader:
                rows += 1
        
        return rows, cols
    
    def read_matrix_row(self, csv_file: str, row_index: int) -> List[float]:
        """
        Read a specific row from CSV file.
        
        Args:
            csv_file: Path to CSV file
            row_index: Zero-based row index
            
        Returns:
            List of floats representing the row
        """
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == row_index:
                    return [float(x) for x in row]
        raise IndexError(f"Row {row_index} not found")
    
    def read_matrix_block(self, csv_file: str, start_row: int, num_rows: int) -> List[List[float]]:
        """
        Read a block of rows from CSV file.
        
        Args:
            csv_file: Path to CSV file
            start_row: Starting row index (zero-based)
            num_rows: Number of rows to read
            
        Returns:
            List of rows (each row is a list of floats)
        """
        block = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i < start_row:
                    continue
                if i >= start_row + num_rows:
                    break
                block.append([float(x) for x in row])
        return block
    
    def transpose_and_cache_b(self, matrix_b_file: str, output_file: str):
        """
        Transpose matrix B and cache it to disk for efficient column access.
        
        For C = A × B, we need columns of B.
        Transposing B allows us to read rows (which are columns of original B).
        
        Args:
            matrix_b_file: Path to original matrix B CSV
            output_file: Path to save transposed matrix
        """
        print("Transposing matrix B for efficient column access...")
        
        # Read B row by row and write transposed
        with open(matrix_b_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            
            # Read all rows and transpose
            rows = []
            for row in reader:
                rows.append([float(x) for x in row])
            
            # Write transposed (columns become rows)
            num_cols = len(rows[0])
            for col_idx in range(num_cols):
                transposed_row = [rows[row_idx][col_idx] for row_idx in range(len(rows))]
                writer.writerow(transposed_row)
        
        print(f"Transposed matrix B cached to {output_file}")
    
    def multiply_row_by_matrix(self, row_a: List[float], matrix_b_transposed_file: str) -> List[float]:
        """
        Multiply a single row of A by entire matrix B.
        
        For C[i][j] = Σ(A[i][k] * B[k][j]), we compute:
        - For each column j of B, compute dot product of row A[i] with column B[:,j]
        - Since B is transposed, column j becomes row j in transposed file
        
        Args:
            row_a: Single row from matrix A
            matrix_b_transposed_file: Path to transposed matrix B CSV
            
        Returns:
            Result row (one row of output matrix C)
        """
        result_row = []
        
        # Read each row of transposed B (which is a column of original B)
        with open(matrix_b_transposed_file, 'r') as f:
            reader = csv.reader(f)
            for transposed_row in reader:
                col_b = [float(x) for x in transposed_row]
                
                # Compute dot product: row_a · col_b
                dot_product = sum(row_a[k] * col_b[k] for k in range(len(row_a)))
                result_row.append(dot_product)
        
        return result_row
    
    def multiply_matrices(self, matrix_a_file: str, matrix_b_file: str, 
                          output_file: str, use_block_method: bool = False,
                          block_size: int = 100):
        """
        Multiply two matrices stored in CSV files.
        
        Args:
            matrix_a_file: Path to matrix A CSV (8000 × 5000)
            matrix_b_file: Path to matrix B CSV (5000 × 3000)
            output_file: Path to save result CSV (8000 × 3000)
            use_block_method: If True, use block-based multiplication
            block_size: Number of rows to process at a time (for block method)
        """
        print("Starting memory-efficient matrix multiplication...")
        
        # Get dimensions
        rows_a, cols_a = self.get_matrix_dimensions(matrix_a_file)
        rows_b, cols_b = self.get_matrix_dimensions(matrix_b_file)
        
        print(f"Matrix A: {rows_a} × {cols_a}")
        print(f"Matrix B: {rows_b} × {cols_b}")
        
        if cols_a != rows_b:
            raise ValueError(f"Matrix dimensions incompatible: A has {cols_a} cols, B has {rows_b} rows")
        
        result_rows = rows_a
        result_cols = cols_b
        print(f"Result matrix C: {result_rows} × {result_cols}")
        
        # Transpose B for efficient column access
        transposed_b_file = matrix_b_file.replace('.csv', '_transposed.csv')
        if not os.path.exists(transposed_b_file):
            self.transpose_and_cache_b(matrix_b_file, transposed_b_file)
        
        # Open output file for writing
        with open(output_file, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            
            if use_block_method:
                # Block-based method: process multiple rows at a time
                print(f"Using block method with block size: {block_size}")
                
                for start_row in range(0, rows_a, block_size):
                    end_row = min(start_row + block_size, rows_a)
                    print(f"Processing rows {start_row} to {end_row-1}...")
                    
                    # Read block of A
                    block_a = self.read_matrix_block(matrix_a_file, start_row, end_row - start_row)
                    
                    # Compute result for this block
                    for row_a in block_a:
                        result_row = self.multiply_row_by_matrix(row_a, transposed_b_file)
                        writer.writerow(result_row)
            else:
                # Row-by-row method: process one row at a time (most memory efficient)
                print("Using row-by-row method (most memory efficient)...")
                
                for i in range(rows_a):
                    if i % 100 == 0:
                        print(f"Processing row {i}/{rows_a}...")
                    
                    # Read one row of A
                    row_a = self.read_matrix_row(matrix_a_file, i)
                    
                    # Multiply by entire B matrix
                    result_row = self.multiply_row_by_matrix(row_a, transposed_b_file)
                    
                    # Write result row immediately
                    writer.writerow(result_row)
        
        print(f"Matrix multiplication complete! Result saved to {output_file}")
        
        # Clean up transposed file if desired
        # os.remove(transposed_b_file)
    
    def read_matrix_column_block(self, csv_file: str, start_col: int, num_cols: int) -> List[List[float]]:
        """
        Read a block of columns from CSV file.
        
        This reads all rows but only specified columns, which is memory efficient
        for accessing column blocks of B.
        
        Args:
            csv_file: Path to CSV file
            start_col: Starting column index (zero-based)
            num_cols: Number of columns to read
            
        Returns:
            List of rows, where each row contains only the specified columns
        """
        block = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                row_floats = [float(x) for x in row]
                # Extract only the columns we need
                col_block = row_floats[start_col:start_col + num_cols]
                block.append(col_block)
        return block
    
    def multiply_matrices_ultra_efficient(self, matrix_a_file: str, matrix_b_file: str,
                                         output_file: str, 
                                         block_rows_a: int = 100,
                                         block_cols_b: int = 500):
        """
        Ultra memory-efficient matrix multiplication when B cannot fit in memory.
        
        Strategy: Block-based multiplication
        - Process A in row blocks
        - Process B in column blocks
        - For each combination, compute partial results
        - Accumulate results incrementally
        
        Memory usage:
        - Block of A: block_rows_a × cols_a × 8 bytes
        - Block of B: rows_b × block_cols_b × 8 bytes
        - Partial result: block_rows_a × block_cols_b × 8 bytes
        - Total: (block_rows_a × cols_a + rows_b × block_cols_b + block_rows_a × block_cols_b) × 8 bytes
        
        Args:
            matrix_a_file: Path to matrix A CSV
            matrix_b_file: Path to matrix B CSV
            output_file: Path to save result CSV
            block_rows_a: Number of rows of A to process at once
            block_cols_b: Number of columns of B to process at once
        """
        print("Starting ultra memory-efficient matrix multiplication...")
        print("(Matrix B cannot fit entirely in memory)")
        
        # Get dimensions
        rows_a, cols_a = self.get_matrix_dimensions(matrix_a_file)
        rows_b, cols_b = self.get_matrix_dimensions(matrix_b_file)
        
        print(f"Matrix A: {rows_a} × {cols_a}")
        print(f"Matrix B: {rows_b} × {cols_b}")
        
        if cols_a != rows_b:
            raise ValueError(f"Matrix dimensions incompatible: A has {cols_a} cols, B has {rows_b} rows")
        
        result_rows = rows_a
        result_cols = cols_b
        print(f"Result matrix C: {result_rows} × {result_cols}")
        print(f"Block sizes: A rows={block_rows_a}, B cols={block_cols_b}")
        
        # Estimate memory usage
        float_size = 8
        memory_mb = (block_rows_a * cols_a + rows_b * block_cols_b + block_rows_a * block_cols_b) * float_size / (1024 * 1024)
        print(f"Estimated memory usage: {memory_mb:.2f} MB")
        
        if memory_mb > self.memory_limit_mb:
            print(f"WARNING: Estimated memory ({memory_mb:.2f} MB) exceeds limit ({self.memory_limit_mb} MB)")
            print("Consider reducing block_rows_a or block_cols_b")
        
        # Initialize result matrix file (we'll accumulate results)
        # First, create empty result file with zeros
        print("Initializing result matrix...")
        with open(output_file, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            for i in range(result_rows):
                writer.writerow([0.0] * result_cols)
        
        # Process A in row blocks
        for start_row_a in range(0, rows_a, block_rows_a):
            end_row_a = min(start_row_a + block_rows_a, rows_a)
            actual_block_rows = end_row_a - start_row_a
            
            print(f"\nProcessing A rows {start_row_a} to {end_row_a-1}...")
            
            # Read block of A
            block_a = self.read_matrix_block(matrix_a_file, start_row_a, actual_block_rows)
            
            # Process B in column blocks
            for start_col_b in range(0, cols_b, block_cols_b):
                end_col_b = min(start_col_b + block_cols_b, cols_b)
                actual_block_cols = end_col_b - start_col_b
                
                print(f"  Processing B columns {start_col_b} to {end_col_b-1}...")
                
                # Read block of B (all rows, but only specified columns)
                block_b = self.read_matrix_column_block(matrix_b_file, start_col_b, actual_block_cols)
                
                # Compute partial result: block_a × block_b
                # This gives us a partial result for C[start_row_a:end_row_a, start_col_b:end_col_b]
                partial_result = []
                for row_a in block_a:
                    result_row = []
                    # For each column in block_b
                    for col_idx in range(actual_block_cols):
                        # Compute dot product: row_a · column col_idx of block_b
                        dot_product = sum(row_a[k] * block_b[k][col_idx] for k in range(cols_a))
                        result_row.append(dot_product)
                    partial_result.append(result_row)
                
                # Accumulate partial result into output file
                # Read current values, add partial result, write back
                self._accumulate_partial_result(
                    output_file, 
                    partial_result,
                    start_row_a, 
                    start_col_b,
                    actual_block_rows,
                    actual_block_cols
                )
        
        print(f"\nMatrix multiplication complete! Result saved to {output_file}")
    
    def _accumulate_partial_result(self, output_file: str, partial_result: List[List[float]],
                                   start_row: int, start_col: int,
                                   num_rows: int, num_cols: int):
        """
        Accumulate partial result into the output file efficiently.
        
        Only reads and updates the rows that need to be modified.
        
        Args:
            output_file: Path to result CSV file
            partial_result: Partial result to add (list of rows)
            start_row: Starting row index in result matrix
            start_col: Starting column index in result matrix
            num_rows: Number of rows in partial result
            num_cols: Number of columns in partial result
        """
        # Read only the rows we need to update
        rows_to_update = []
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if start_row <= i < start_row + num_rows:
                    rows_to_update.append([float(x) for x in row])
                elif i >= start_row + num_rows:
                    break
        
        # Accumulate partial result into the rows we're updating
        for i in range(num_rows):
            for j in range(num_cols):
                rows_to_update[i][start_col + j] += partial_result[i][j]
        
        # Write back only the updated rows using a temporary file approach
        # Read all rows, update specific ones, write all back
        all_rows = []
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                all_rows.append([float(x) for x in row])
        
        # Update the specific rows
        for i in range(num_rows):
            all_rows[start_row + i] = rows_to_update[i]
        
        # Write all rows back
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in all_rows:
                writer.writerow(row)
    
    def estimate_memory_usage(self, matrix_a_file: str, matrix_b_file: str, 
                              block_size: int = 100, block_cols_b: int = 500) -> dict:
        """
        Estimate memory usage for different approaches.
        
        Returns:
            Dictionary with memory estimates
        """
        rows_a, cols_a = self.get_matrix_dimensions(matrix_a_file)
        rows_b, cols_b = self.get_matrix_dimensions(matrix_b_file)
        
        float_size = 8  # bytes
        
        estimates = {
            'naive_all_at_once': {
                'matrix_a_mb': (rows_a * cols_a * float_size) / (1024 * 1024),
                'matrix_b_mb': (rows_b * cols_b * float_size) / (1024 * 1024),
                'result_mb': (rows_a * cols_b * float_size) / (1024 * 1024),
                'total_mb': (rows_a * cols_a + rows_b * cols_b + rows_a * cols_b) * float_size / (1024 * 1024)
            },
            'row_by_row': {
                'one_row_a_mb': (cols_a * float_size) / (1024 * 1024),
                'transposed_b_mb': (rows_b * cols_b * float_size) / (1024 * 1024),
                'one_result_row_mb': (cols_b * float_size) / (1024 * 1024),
                'total_mb': (cols_a + rows_b * cols_b + cols_b) * float_size / (1024 * 1024)
            },
            'block_method': {
                'block_a_mb': (block_size * cols_a * float_size) / (1024 * 1024),
                'transposed_b_mb': (rows_b * cols_b * float_size) / (1024 * 1024),
                'block_result_mb': (block_size * cols_b * float_size) / (1024 * 1024),
                'total_mb': (block_size * cols_a + rows_b * cols_b + block_size * cols_b) * float_size / (1024 * 1024)
            },
            'ultra_efficient': {
                'block_a_mb': (block_size * cols_a * float_size) / (1024 * 1024),
                'block_b_mb': (rows_b * block_cols_b * float_size) / (1024 * 1024),
                'partial_result_mb': (block_size * block_cols_b * float_size) / (1024 * 1024),
                'total_mb': (block_size * cols_a + rows_b * block_cols_b + block_size * block_cols_b) * float_size / (1024 * 1024)
            }
        }
        
        return estimates


def create_sample_matrices():
    """Create sample CSV files for testing."""
    import random
    
    print("Creating sample matrices...")
    
    # Matrix A: 8000 × 5000
    rows_a, cols_a = 8000, 5000
    with open('matrix_a.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(rows_a):
            row = [random.random() for _ in range(cols_a)]
            writer.writerow(row)
            if (i + 1) % 1000 == 0:
                print(f"Written {i+1}/{rows_a} rows of matrix A")
    
    print(f"Created matrix_a.csv ({rows_a} × {cols_a})")
    
    # Matrix B: 5000 × 3000
    rows_b, cols_b = 5000, 3000
    with open('matrix_b.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i in range(rows_b):
            row = [random.random() for _ in range(cols_b)]
            writer.writerow(row)
            if (i + 1) % 1000 == 0:
                print(f"Written {i+1}/{rows_b} rows of matrix B")
    
    print(f"Created matrix_b.csv ({rows_b} × {cols_b})")


if __name__ == "__main__":
    # Analysis and explanation
    print("="*70)
    print("MEMORY-EFFICIENT MATRIX MULTIPLICATION - ANALYSIS")
    print("="*70)
    print("\n1. PROBLEM DIFFICULTY ANALYSIS:")
    print("-" * 70)
    print("Difficulty Level: Medium-Hard")
    print("\nWhy it's challenging:")
    print("  • Requires understanding of matrix multiplication algorithm")
    print("  • Need to manage memory constraints effectively")
    print("  • Must handle streaming/chunked data processing")
    print("  • CSV I/O optimization is important")
    print("\nTime Estimates:")
    print("  • Junior Engineer: 2-3 hours")
    print("  • Mid-level Engineer: 1-2 hours")
    print("  • Senior Engineer: 30-60 minutes")
    print("\nKey Concepts Tested:")
    print("  • Algorithm design (block matrix multiplication)")
    print("  • Memory management")
    print("  • File I/O optimization")
    print("  • Trade-offs between time and space complexity")
    
    print("\n" + "="*70)
    print("2. SOLUTION DESIGN")
    print("="*70)
    print("\nMemory Constraint Analysis:")
    print("  • Matrix A: 8000 × 5000 × 8 bytes = 320 MB")
    print("  • Matrix B: 5000 × 3000 × 8 bytes = 120 MB")
    print("  • Result C: 8000 × 3000 × 8 bytes = 192 MB")
    print("  • Total (naive): ~632 MB (EXCEEDS 500MB limit)")
    
    print("\nSolution Strategy:")
    print("  1. Transpose matrix B and cache to disk")
    print("     - Allows efficient column access (read rows of transposed B)")
    print("     - One-time cost, reused for all rows of A")
    print("  2. Process matrix A row-by-row")
    print("     - Load only one row of A at a time (~40 KB)")
    print("     - Multiply with entire transposed B (~114 MB)")
    print("     - Write result row immediately")
    print("  3. Memory usage: ~114 MB (well under 500MB limit)")
    
    print("\nEfficiency Explanation:")
    print("  • CSV library: Efficient streaming, no need to load entire file")
    print("  • Row-by-row processing: Minimal memory footprint")
    print("  • Transposed B cache: Enables column access without full matrix load")
    print("  • Incremental writing: No need to store full result in memory")
    print("  • Time complexity: O(rows_a × cols_a × cols_b) - same as standard")
    print("  • Space complexity: O(cols_a + rows_b × cols_b + cols_b) - much better!")
    
    print("\n" + "="*70)
    print("3. USAGE EXAMPLE")
    print("="*70)
    
    # Check if sample files exist
    if not (os.path.exists('matrix_a.csv') and os.path.exists('matrix_b.csv')):
        print("\nSample matrices not found. Creating them...")
        print("(This may take a few minutes for large matrices)")
        create_sample_matrices()
    else:
        print("\nUsing existing matrix files...")
    
    # Initialize multiplier
    multiplier = MemoryEfficientMatrixMultiplier(memory_limit_mb=500.0)
    
    # Estimate memory usage
    print("\nMemory Usage Estimates:")
    estimates = multiplier.estimate_memory_usage('matrix_a.csv', 'matrix_b.csv')
    
    print(f"\nNaive approach (load all): {estimates['naive_all_at_once']['total_mb']:.2f} MB")
    print(f"Row-by-row approach: {estimates['row_by_row']['total_mb']:.2f} MB")
    print(f"Block approach (100 rows): {estimates['block_method']['total_mb']:.2f} MB")
    
    # Perform multiplication
    print("\n" + "="*70)
    print("Performing matrix multiplication...")
    print("="*70)
    
    multiplier.multiply_matrices(
        matrix_a_file='matrix_a.csv',
        matrix_b_file='matrix_b.csv',
        output_file='matrix_result.csv',
        use_block_method=False  # Use row-by-row for maximum memory efficiency
    )
    
    print("\n" + "="*70)
    print("Verification:")
    print("="*70)
    
    # Verify result dimensions
    rows_result, cols_result = multiplier.get_matrix_dimensions('matrix_result.csv')
    print(f"Result matrix dimensions: {rows_result} × {cols_result}")
    print(f"Expected: 8000 × 3000")
    
    if rows_result == 8000 and cols_result == 3000:
        print("✓ Dimensions match expected result!")
    else:
        print("✗ Dimensions don't match!")

