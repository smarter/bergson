"""Test EKFAC computation against ground truth."""

import json
import os

import pytest
import torch

from test_covariance import test_covariances
from test_eigenvalue_correction import test_eigenvalue_correction
from test_eigenvectors import test_eigenvectors
from test_utils import set_all_seeds

from bergson.data import DataConfig, IndexConfig
from bergson.distributed import distributed_computing
from bergson.hessians.compute_all import compute_all_factors


@pytest.fixture(scope="module")
def ekfac_config(test_dir, ground_truth_path, use_fsdp, world_size, overwrite, run_path):
    """Create and configure EKFAC computation."""
    set_all_seeds(seed=42)

    # Verify required files exist
    required_files = [
        "covariances",
        "eigenvalue_corrections",
        "eigenvectors",
        "index_config.json",
    ]

    for file_name in required_files:
        file_path = os.path.join(ground_truth_path, file_name)
        if not os.path.exists(file_path):
            pytest.skip(f"Missing required file: {file_name}")

    # Load configuration
    with open(os.path.join(ground_truth_path, "index_config.json"), "r") as f:
        cfg_json = json.load(f)

    cfg = IndexConfig(**cfg_json)
    cfg.data = DataConfig(**(cfg_json["data"]))
    assert isinstance(cfg.fsdp, bool)  # for the type checker
    cfg.run_path = test_dir + "/run"
    cfg.debug = True
    cfg.fsdp = use_fsdp
    cfg.world_size = world_size
    cfg.sample = False  # Default value, can be made configurable if needed

    # Run EKFAC computation if needed
    if not os.path.exists(run_path) or overwrite:
        distributed_computing(
            cfg=cfg,
            worker_fn=compute_all_factors,
        )
        print("EKFAC computation completed successfully.")
    else:
        print("Using existing run directory.")

    return cfg


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_total_processed_examples(ground_truth_path, run_path, ekfac_config):
    """Test that total processed examples match between ground truth and computed values."""
    total_processed_ground_truth_path = os.path.join(
        ground_truth_path, "covariances/stats.json"
    )
    total_processed_run_path = os.path.join(run_path, "total_processed_covariances.pt")

    with open(total_processed_ground_truth_path, "r") as f:
        ground_truth_data = json.load(f)
        total_processed_ground_truth = ground_truth_data["total_processed_global"]

    total_processed_run = torch.load(total_processed_run_path).item()

    print(f"Ground truth: {total_processed_ground_truth}, Run: {total_processed_run}")
    assert total_processed_ground_truth == total_processed_run, (
        f"Total processed examples do not match! "
        f"Ground truth: {total_processed_ground_truth}, Run: {total_processed_run}"
    )
    print("-*" * 50)


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_activation_covariances(run_path, ground_truth_path, ekfac_config):
    """Test activation covariances against ground truth."""
    test_covariances(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        covariance_type="activation",
    )


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_gradient_covariances(run_path, ground_truth_path, ekfac_config):
    """Test gradient covariances against ground truth."""
    test_covariances(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        covariance_type="gradient",
    )


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_activation_eigenvectors(run_path, ground_truth_path, ekfac_config):
    """Test activation eigenvectors against ground truth.

    Note: Currently this tests for close equality, but does not account for
    sign differences in eigenvectors. TODO: fix.
    """
    test_eigenvectors(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        eigenvector_type="activation",
    )


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_gradient_eigenvectors(run_path, ground_truth_path, ekfac_config):
    """Test gradient eigenvectors against ground truth.

    Note: Currently this tests for close equality, but does not account for
    sign differences in eigenvectors. TODO: fix.
    """
    test_eigenvectors(
        run_path=run_path,
        ground_truth_path=ground_truth_path,
        eigenvector_type="gradient",
    )


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_eigenvalue_corrections(ground_truth_path, run_path, ekfac_config):
    """Test eigenvalue corrections against ground truth."""
    test_eigenvalue_correction(ground_truth_path=ground_truth_path, run_path=run_path)
