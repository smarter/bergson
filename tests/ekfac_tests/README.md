# EKFAC Tests

Pytest-based tests for EKFAC (Eigenvalue-corrected Kronecker-Factored Approximate Curvature) computation.

## Quick Start

### With existing test data

If you have test data available:

```bash
cd tests/ekfac_tests
pytest --test-dir /path/to/test/data
```

### Configuration Options

- `--test-dir`: Path to existing test data directory (optional)
- `--model-name`: Model to use if generating data (default: gpt2)
- `--num-samples`: Number of samples for generated data (default: 5)
- `--max-length`: Maximum sequence length (default: 32)
- `--world-size`: World size for distributed training (default: 1)
- `--overwrite`: Overwrite existing run results

## Test Structure

### Main Tests

- `test_compute_ekfac.py`: Tests EKFAC factor computation
  - Covariances (activation & gradient)
  - Eigenvectors (activation & gradient)
  - Eigenvalue corrections
  - Total processed examples

### Helper Modules

- `test_covariance.py`: Covariance comparison utilities
- `test_eigenvectors.py`: Eigenvector comparison utilities
- `test_eigenvalue_correction.py`: Eigenvalue correction utilities
- `test_utils.py`: Common test utilities (seed setting, etc.)

### Test Data Generation

The `generate_test_data.py` script can create minimal test data using small models:

```bash
python generate_test_data.py --output-dir /tmp/test_data --model-name gpt2 --num-samples 5
```

**Note**: The data generation script is currently a work in progress and needs refinement.

## CI Integration

Tests run in CI using GitHub Actions. The CI workflow:

1. Installs dependencies
2. Runs type checking
3. Runs EKFAC tests with auto-generated test data

See `.github/workflows/build.yml` for details.

## Running Without Test Data

If no test directory is provided, tests will attempt to generate minimal test data automatically.
This requires downloading a small model (like gpt2) and may take a few minutes on first run.

## Development Notes

### Converting from Old Format

The tests have been converted from argparse-based scripts to proper pytest format:

- Old: `python test_compute_ekfac.py --test_dir ... --world_size 8`
- New: `pytest --test-dir ... --world-size 8`

### Using Larger Models

For more comprehensive testing with larger models and datasets:

```bash
pytest --test-dir /path/to/large/test/data --model-name meta-llama/Llama-2-7b-hf
```

## Known Issues

1. Test data generation script needs updates to work seamlessly with all models
2. Currently requires existing test data for full functionality
3. See TODOs in code for additional improvements needed
