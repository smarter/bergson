"""Pytest configuration and fixtures for EKFAC tests."""

import os
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
