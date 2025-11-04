# Testing Guide

This directory contains starter scripts, solution files, and an automated test runner for the easy code challenge exercises.

## Files

- **`starter_01.py` through `starter_04.py`**: Your starter scripts with minimal hints
- **`solution_01.py` through `solution_04.py`**: Reference solutions (don't peek while working!)
- **`test_all.py`**: Automated test runner that compares your solutions

## Quick Start

### Run All Tests
```bash
python test_all.py
```

### Run a Specific Exercise
```bash
python test_all.py --exercise 1
python test_all.py --exercise 2
python test_all.py --exercise 3
python test_all.py --exercise 4
```

## What the Test Runner Does

The `test_all.py` script:

1. **Tests your implementation** against expected outputs
2. **Compares with inefficient implementations** mentioned in the problem descriptions
3. **Compares with reference solutions** to verify correctness
4. **Measures performance** on larger datasets to show improvements
5. **Provides detailed feedback** with color-coded output:
   - 🟢 Green: Passed tests
   - 🔴 Red: Failed tests or errors
   - 🟡 Yellow: Inefficient but correct implementations
   - 🔵 Blue: Test case information

## Workflow

1. Read the exercise markdown file (e.g., `exercise_01_memory_efficient_list.md`)
2. Open and work on the corresponding starter file (e.g., `starter_01.py`)
3. Implement your solution
4. Run `python test_all.py --exercise 1` to test your work
5. Fix any issues and re-run tests until all pass

## Tips

- Don't look at the solution files until you've solved the problem yourself!
- The test runner will show you if your solution is correct even if it's not optimal
- Performance comparisons help you understand the improvements
- Test with small inputs first, then verify with larger datasets

## Troubleshooting

If you get import errors, make sure you're running the test from the `easy` directory:
```bash
cd composer/01_code_challenge/easy
python test_all.py
```

If your solution passes all tests but you want to see the reference implementation, check the corresponding `solution_*.py` file.
