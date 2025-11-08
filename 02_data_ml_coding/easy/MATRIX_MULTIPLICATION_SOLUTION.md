# Memory-Efficient Matrix Multiplication - Interview Solution

## Problem Analysis

### Question Difficulty: **Medium-Hard**

**Why it's challenging:**
- Requires understanding of matrix multiplication algorithm
- Need to manage memory constraints effectively
- Must handle streaming/chunked data processing
- CSV I/O optimization is important

**Time Estimates:**
- **Junior Engineer**: 2-3 hours
- **Mid-level Engineer**: 1-2 hours  
- **Senior Engineer**: 30-60 minutes

**Key Concepts Tested:**
- Algorithm design (block matrix multiplication)
- Memory management
- File I/O optimization
- Trade-offs between time and space complexity

## Memory Constraint Analysis

Given:
- Matrix A: 8000 × 5000
- Matrix B: 5000 × 3000
- Result C: 8000 × 3000
- Each element: float (8 bytes)
- Memory limit: < 500 MB

**Memory calculations:**
- Matrix A: 8000 × 5000 × 8 bytes = **320 MB**
- Matrix B: 5000 × 3000 × 8 bytes = **120 MB**
- Result C: 8000 × 3000 × 8 bytes = **192 MB**
- **Total (naive approach): ~632 MB** ❌ **EXCEEDS 500MB limit**

## Solution Design

### Strategy: Row-by-Row Processing with Transposed B Cache

**Key Insight:** 
For matrix multiplication C = A × B, where C[i][j] = Σ(A[i][k] × B[k][j]):
- We can compute each row of C independently
- Each row of C requires one row of A and all columns of B
- We can transpose B once and cache it, then read it row-by-row (which gives us columns of original B)

### Algorithm Steps:

1. **Transpose and cache matrix B**
   - Read B row-by-row, write transposed version
   - One-time cost: O(rows_b × cols_b)
   - Enables efficient column access (read rows of transposed B)

2. **Process matrix A row-by-row**
   - For each row i in A:
     - Load row A[i] (~40 KB)
     - For each row j in transposed B (which is column j of original B):
       - Compute dot product: A[i] · B[:,j]
     - Write result row C[i] immediately

3. **Memory usage breakdown:**
   - One row of A: 5000 × 8 bytes = **40 KB**
   - Transposed B (cached): 5000 × 3000 × 8 bytes = **120 MB**
   - One result row: 3000 × 8 bytes = **24 KB**
   - **Total: ~120 MB** ✅ **Well under 500MB limit**

### Why This is Efficient:

1. **CSV Library Benefits:**
   - `csv.reader()` streams data, doesn't load entire file
   - Memory-efficient for large files
   - Can seek/read specific rows without loading all

2. **Row-by-Row Processing:**
   - Minimal memory footprint
   - Only one row of A in memory at a time
   - Result written incrementally

3. **Transposed B Cache:**
   - Enables column access without full matrix load
   - One-time transpose cost amortized over all rows
   - Can be reused if processing multiple matrices

4. **Time Complexity:**
   - Same as standard matrix multiplication: O(rows_a × cols_a × cols_b)
   - No asymptotic time penalty

5. **Space Complexity:**
   - O(cols_a + rows_b × cols_b + cols_b) instead of O(rows_a × cols_a + rows_b × cols_b + rows_a × cols_b)
   - Massive improvement for large matrices

## What If Matrix B Cannot Fit in Memory?

**Scenario:** Even the transposed matrix B (~120 MB) exceeds available memory.

**Solution:** Ultra-efficient block-based multiplication

### Strategy: Block-Based Multiplication for Both Matrices

When B cannot fit in memory, we need to process both matrices in blocks:

1. **Process A in row blocks** (e.g., 100 rows at a time)
2. **Process B in column blocks** (e.g., 500 columns at a time)
3. **For each combination**, compute partial results
4. **Accumulate results** incrementally into output file

### Memory Usage:

- Block of A: `block_rows_a × cols_a × 8 bytes`
- Block of B: `rows_b × block_cols_b × 8 bytes`
- Partial result: `block_rows_a × block_cols_b × 8 bytes`
- **Total: Configurable via block sizes**

**Example:** With `block_rows_a=100` and `block_cols_b=500`:
- Block A: 100 × 5000 × 8 = 3.8 MB
- Block B: 5000 × 500 × 8 = 19.1 MB
- Partial result: 100 × 500 × 8 = 0.4 MB
- **Total: ~23.3 MB** ✅ **Well under any reasonable limit**

### Algorithm:

```
For each block of A rows:
    For each block of B columns:
        Compute: block_A × block_B → partial_result
        Accumulate partial_result into output file
```

### When to Use Which Method:

| Method | When to Use | Memory Usage | Speed |
|--------|-------------|--------------|-------|
| **Row-by-row** | Transposed B fits in memory (~120 MB) | ~120 MB | Faster (less I/O) |
| **Ultra-efficient** | Even transposed B doesn't fit | Configurable (~20-50 MB) | Slower (more I/O) |

## Alternative: Block-Based Method (B Fits in Memory)

For better performance when B fits in memory:
- Process multiple rows of A at once (e.g., 100 rows)
- Reduces file I/O overhead
- Still respects memory constraints
- Memory: ~120 MB + (block_size × cols_a × 8 bytes)

## Implementation Highlights

```python
# Key functions:
1. get_matrix_dimensions() - Get size without loading full matrix
2. read_matrix_row() - Read single row efficiently
3. transpose_and_cache_b() - One-time transpose operation
4. multiply_row_by_matrix() - Compute one result row
5. multiply_matrices() - Main orchestration function
```

## Usage

### Method 1: Row-by-Row (B fits in memory)

```python
multiplier = MemoryEfficientMatrixMultiplier(memory_limit_mb=500.0)
multiplier.multiply_matrices(
    matrix_a_file='matrix_a.csv',
    matrix_b_file='matrix_b.csv',
    output_file='matrix_result.csv',
    use_block_method=False  # Row-by-row for max efficiency
)
```

### Method 2: Ultra-Efficient (B doesn't fit in memory)

```python
multiplier = MemoryEfficientMatrixMultiplier(memory_limit_mb=50.0)  # Stricter limit
multiplier.multiply_matrices_ultra_efficient(
    matrix_a_file='matrix_a.csv',
    matrix_b_file='matrix_b.csv',
    output_file='matrix_result.csv',
    block_rows_a=100,   # Process 100 rows of A at a time
    block_cols_b=500    # Process 500 columns of B at a time
)
```

## Verification

The solution:
- ✅ Respects 500MB memory limit
- ✅ Handles matrices of any size (limited by disk, not RAM)
- ✅ Uses standard CSV library (no special dependencies)
- ✅ Maintains O(n³) time complexity
- ✅ Can be extended for distributed processing

## Extensions

For even larger matrices or distributed systems:
- Process blocks in parallel across multiple machines
- Use memory-mapped files for very large matrices
- Implement chunked reading with configurable block sizes
- Add progress tracking and resumable processing

