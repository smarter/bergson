# EKFAC Tests

This directory contains pytest tests for EKFAC (Eigenvalue-corrected Kronecker-Factored Approximate Curvature) computation and application.

## Overview

The tests verify the correctness of EKFAC computations by comparing against ground truth data. Tests are organized into three main categories:

1. **Smoke Tests** (`test_smoke.py`): Fast unit tests that run on CPU without requiring test data or GPU
2. **Compute Tests** (`test_compute_ekfac.py`): Verify EKFAC factor computation including covariances, eigenvectors, and eigenvalue corrections
3. **Apply Tests** (`test_apply_ekfac.py`): Verify EKFAC transformation applied to gradients

## Running Tests

### Quick Start (Smoke Tests)

Run fast smoke tests on CPU without any test data:

```bash
cd tests/ekfac_tests
pytest -m "smoke"
```

These tests validate basic functionality and are perfect for CI environments.

### Running All Tests

Run all tests (will skip tests requiring GPU/test data):

```bash
cd tests/ekfac_tests
pytest
```

### Running with Test Data

If you have ground truth test data available:

```bash
pytest --test-dir /path/to/test/data
```

### Test Markers

Tests are marked with the following markers:

- `requires_test_data`: Tests requiring ground truth data files
- `requires_gpu`: Tests requiring GPU hardware
- `slow`: Tests that take a long time to run

To exclude specific test categories:

```bash
# Run only fast tests
pytest -m "not slow"

# Run only tests that don't require GPU
pytest -m "not requires_gpu"

# Run only tests that don't require test data
pytest -m "not requires_test_data"

# Combine markers
pytest -m "not slow and not requires_gpu"
```

### Command-Line Options

The following custom options are available:

- `--test-dir`: Directory containing test files with ground truth data
- `--overwrite`: Overwrite existing run directory
- `--use-fsdp`: Use Fully Sharded Data Parallel (FSDP)
- `--world-size`: World size for distributed training (default: 8)
- `--gradient-path`: Path to gradient files (for apply tests)
- `--gradient-batch-size`: Batch size for gradient computation (default: 1)

Example with options:

```bash
pytest --test-dir /root/bergson/test_files/pile_100_examples \
       --world-size 8 \
       --use-fsdp \
       --overwrite
```

### Environment Variables

You can also use the `EKFAC_TEST_DIR` environment variable to set the test directory:

```bash
export EKFAC_TEST_DIR=/path/to/test/data
pytest
```

## Test Structure

### Fixtures

Shared fixtures are defined in `conftest.py`:

- `test_dir`: Path to test data directory
- `ground_truth_path`: Path to ground truth data
- `run_path`: Path to test run outputs
- `use_fsdp`, `world_size`, etc.: Configuration options

### Helper Functions

- `test_utils.py`: Common utilities like `set_all_seeds()` for reproducibility
- `test_covariance.py`: Functions to test covariance matrices
- `test_eigenvectors.py`: Functions to test eigenvector computations
- `test_eigenvalue_correction.py`: Functions to test eigenvalue corrections

## CI Integration

Tests are automatically run in CI on the `ekfac` branch. The CI runs smoke tests that work on CPU without requiring test data:

```bash
pytest -m "smoke"
```

These smoke tests validate:
- Import functionality
- Basic tensor operations
- Covariance computation logic
- Eigendecomposition
- Test utilities

## Generating Minimal Test Data

For local testing with a tiny model on CPU, you can generate minimal test data:

```bash
cd tests/ekfac_tests
python generate_minimal_test_data.py --output-dir fixtures/minimal --num-samples 10
```

This uses a very small model (sdobson/nanochat) and generates ground truth data that can be used for integration tests on CPU.

## Legacy Scripts

The following shell scripts are kept for reference but have been superseded by pytest:

- `run_test_compute_ekfac.sh`: Old script for running compute tests
- `run_apply_compute_ekfac.sh`: Old script for running apply tests

These can be deleted once the pytest migration is confirmed working.

## Adding New Tests

To add new tests:

1. Create test functions prefixed with `test_`
2. Add appropriate markers (e.g., `@pytest.mark.requires_gpu`)
3. Use fixtures from `conftest.py` for configuration
4. Document any new command-line options in `conftest.py`

Example:

```python
import pytest

@pytest.mark.requires_gpu
@pytest.mark.slow
def test_my_new_feature(ground_truth_path, run_path):
    """Test description."""
    # Test implementation
    assert result == expected
```
