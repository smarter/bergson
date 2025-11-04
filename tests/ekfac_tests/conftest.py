"""Pytest configuration and fixtures for EKFAC tests."""

import json
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
        help="Directory containing test data. If not provided, generates test data using --model-name.",
    )
    parser.addoption(
        "--model-name",
        action="store",
        default="calum/tinystories-gpt2-3M",
        help="Model to use for tests (default: calum/tinystories-gpt2-3M)",
    )
    parser.addoption(
        "--num-samples",
        action="store",
        type=int,
        default=5,
        help="Number of samples for generated test data (default: 5)",
    )
    parser.addoption(
        "--max-length",
        action="store",
        type=int,
        default=32,
        help="Maximum sequence length (default: 32)",
    )
    parser.addoption(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing run directory",
    )
    parser.addoption(
        "--world-size",
        action="store",
        type=int,
        default=1,
        help="World size for distributed training (default: 1)",
    )


@pytest.fixture(scope="session")
def test_dir(request, tmp_path_factory):
    """Get or generate test directory."""
    # Check if test directory was provided
    test_dir = request.config.getoption("--test-dir")
    if test_dir is not None:
        return test_dir

    # Check environment variable
    test_dir = os.environ.get("EKFAC_TEST_DIR")
    if test_dir is not None:
        return test_dir

    # Generate test data
    model_name = request.config.getoption("--model-name")
    num_samples = request.config.getoption("--num-samples")
    max_length = request.config.getoption("--max-length")

    print(f"\n{'='*60}")
    print(f"Generating test data with {model_name}")
    print(f"Samples: {num_samples}, Max length: {max_length}")
    print(f"{'='*60}\n")

    tmp_dir = tmp_path_factory.mktemp("ekfac_test_data")
    test_dir = str(tmp_dir)
    script_path = Path(__file__).parent / "generate_test_data.py"

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--output-dir",
                test_dir,
                "--model-name",
                model_name,
                "--num-samples",
                str(num_samples),
                "--max-length",
                str(max_length),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return test_dir
    except subprocess.CalledProcessError as e:
        pytest.skip(f"Failed to generate test data: {e.stderr}")
    except Exception as e:
        pytest.skip(f"Error generating test data: {e}")


@pytest.fixture(scope="session")
def ground_truth_path(test_dir):
    """Get ground truth path."""
    # Support both old format (test_dir/ground_truth) and new format (test_dir directly)
    gt_path = os.path.join(test_dir, "ground_truth")
    if os.path.exists(gt_path):
        return gt_path
    return test_dir


@pytest.fixture(scope="session")
def run_path(test_dir):
    """Get run output path."""
    return os.path.join(test_dir, "run/influence_results")


@pytest.fixture(scope="session")
def overwrite(request):
    """Get overwrite flag."""
    return request.config.getoption("--overwrite")


@pytest.fixture(scope="session")
def world_size(request):
    """Get world size."""
    return request.config.getoption("--world-size")


@pytest.fixture(scope="session")
def model_name(request):
    """Get model name."""
    return request.config.getoption("--model-name")
