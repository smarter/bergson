"""Pytest configuration and fixtures for EKFAC tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for EKFAC tests."""
    parser.addoption(
        "--test-dir",
        action="store",
        default=None,
        help="Directory containing test files with ground truth data",
    )
    parser.addoption(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing run directory",
    )
    parser.addoption(
        "--use-fsdp",
        action="store_true",
        default=False,
        help="Use Fully Sharded Data Parallel (FSDP)",
    )
    parser.addoption(
        "--world-size",
        action="store",
        type=int,
        default=8,
        help="World size for distributed training",
    )
    parser.addoption(
        "--gradient-path",
        action="store",
        default=None,
        help="Path to the gradient for apply tests",
    )
    parser.addoption(
        "--gradient-batch-size",
        action="store",
        type=int,
        default=1,
        help="Batch size for gradient computation",
    )


@pytest.fixture(scope="session")
def test_dir(request):
    """Get test directory from command line or environment variable."""
    test_dir = request.config.getoption("--test-dir")
    if test_dir is None:
        test_dir = os.environ.get("EKFAC_TEST_DIR")
    return test_dir


@pytest.fixture(scope="session")
def overwrite(request):
    """Get overwrite flag from command line."""
    return request.config.getoption("--overwrite")


@pytest.fixture(scope="session")
def use_fsdp(request):
    """Get FSDP flag from command line."""
    return request.config.getoption("--use-fsdp")


@pytest.fixture(scope="session")
def world_size(request):
    """Get world size from command line."""
    return request.config.getoption("--world-size")


@pytest.fixture(scope="session")
def gradient_path(request):
    """Get gradient path from command line."""
    return request.config.getoption("--gradient-path")


@pytest.fixture(scope="session")
def gradient_batch_size(request):
    """Get gradient batch size from command line."""
    return request.config.getoption("--gradient-batch-size")


@pytest.fixture(scope="session")
def ground_truth_path(test_dir):
    """Get ground truth path from test directory."""
    if test_dir is None:
        pytest.skip("Test directory not provided. Use --test-dir or EKFAC_TEST_DIR")
    return os.path.join(test_dir, "ground_truth")


@pytest.fixture(scope="session")
def run_path(test_dir):
    """Get run path from test directory."""
    if test_dir is None:
        pytest.skip("Test directory not provided. Use --test-dir or EKFAC_TEST_DIR")
    return os.path.join(test_dir, "run/influence_results")


@pytest.fixture(scope="session")
def minimal_test_data(tmp_path_factory):
    """Generate or use minimal test data for CPU-based smoke tests.

    This fixture generates a small test dataset using a tiny model on CPU.
    The data is cached in a temporary directory for the test session.
    """
    # Check if pre-generated minimal test data exists
    fixtures_dir = Path(__file__).parent / "fixtures" / "minimal"
    if fixtures_dir.exists() and (fixtures_dir / "index_config.json").exists():
        return str(fixtures_dir)

    # Otherwise, generate it on the fly
    tmp_dir = tmp_path_factory.mktemp("minimal_test_data")
    output_dir = str(tmp_dir / "minimal")

    print("\nGenerating minimal test data for smoke tests...")
    script_path = Path(__file__).parent / "generate_minimal_test_data.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--output-dir", output_dir,
                "--num-samples", "5",
                "--max-length", "32",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes timeout
        )
        print(result.stdout)
        return output_dir
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate minimal test data: {e.stderr}")
    except subprocess.TimeoutExpired:
        pytest.skip("Timeout generating minimal test data")
    except Exception as e:
        pytest.skip(f"Error generating minimal test data: {e}")


@pytest.fixture(scope="session")
def smoke_test_mode(request):
    """Check if running in smoke test mode (CPU, minimal data)."""
    return request.config.getoption("--smoke-tests", default=False)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_test_data: mark test as requiring test data files"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as a fast smoke test that runs on CPU"
    )
