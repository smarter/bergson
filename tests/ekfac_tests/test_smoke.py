"""Smoke tests for EKFAC functionality.

These tests are designed to run quickly on CPU without GPU or large test datasets.
They validate basic functionality and can run in CI environments.
"""

import pytest
import torch


@pytest.mark.smoke
def test_imports():
    """Test that all required imports work."""
    from bergson.data import DataConfig, IndexConfig
    from bergson.distributed import distributed_computing
    from bergson.hessians.compute_all import compute_all_factors
    from bergson.hessians.utils import TensorDict

    assert DataConfig is not None
    assert IndexConfig is not None
    assert distributed_computing is not None
    assert compute_all_factors is not None
    assert TensorDict is not None


@pytest.mark.smoke
def test_tensor_dict_operations():
    """Test TensorDict operations used in EKFAC."""
    from bergson.hessians.utils import TensorDict

    # Create sample tensors
    data = {
        "layer1": torch.randn(10, 10),
        "layer2": torch.randn(20, 20),
    }

    td = TensorDict(data)

    # Test operations
    assert "layer1" in td
    assert "layer2" in td
    assert len(td) == 2

    # Test allclose
    td2 = TensorDict({
        "layer1": data["layer1"].clone(),
        "layer2": data["layer2"].clone(),
    })
    result = td.allclose(td2)
    assert all(result.values())

    # Test max
    max_vals = td.max()
    assert "layer1" in max_vals
    assert "layer2" in max_vals


@pytest.mark.smoke
def test_covariance_computation():
    """Test basic covariance computation logic."""
    import torch

    # Simulate activations: batch_size x seq_len x hidden_dim
    batch_size, seq_len, hidden_dim = 2, 4, 8
    activations = torch.randn(batch_size, seq_len, hidden_dim)

    # Reshape and compute covariance as done in EKFAC
    a = activations.reshape(-1, hidden_dim)  # [N*S, H]
    cov = a.mT @ a  # [H, H]

    assert cov.shape == (hidden_dim, hidden_dim)
    # Covariance should be symmetric
    assert torch.allclose(cov, cov.T, atol=1e-5)


@pytest.mark.smoke
def test_eigendecomposition():
    """Test eigendecomposition used in EKFAC."""
    import torch

    # Create a symmetric positive definite matrix
    size = 10
    A = torch.randn(size, size)
    cov = A @ A.T + torch.eye(size) * 0.1  # Make it positive definite

    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    assert eigenvalues.shape == (size,)
    assert eigenvectors.shape == (size, size)
    # All eigenvalues should be positive (we made it positive definite)
    assert (eigenvalues > 0).all()


@pytest.mark.smoke
def test_ground_truth_collector():
    """Test ground truth collector instantiation."""
    from ground_truth.collector import GroundTruthCovarianceCollector
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
    )

    activation_covariances = {}
    gradient_covariances = {}

    collector = GroundTruthCovarianceCollector(
        model=model,
        layer_names=["0", "2"],  # The two linear layers
        activation_covariances=activation_covariances,
        gradient_covariances=gradient_covariances,
    )

    assert collector.model is model
    assert len(collector.layer_names) == 2


@pytest.mark.smoke
def test_set_all_seeds():
    """Test that seed setting is deterministic."""
    from test_utils import set_all_seeds
    import torch

    set_all_seeds(42)
    val1 = torch.randn(5)

    set_all_seeds(42)
    val2 = torch.randn(5)

    assert torch.allclose(val1, val2)


@pytest.mark.smoke
@pytest.mark.parametrize("covariance_type", ["activation", "gradient"])
def test_covariance_test_function(covariance_type):
    """Test that covariance comparison function works."""
    from test_covariance import test_covariances
    from bergson.hessians.utils import TensorDict
    from safetensors.torch import save_file
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy ground truth
        gt_dir = os.path.join(tmpdir, "ground_truth")
        run_dir = os.path.join(tmpdir, "run")
        os.makedirs(f"{gt_dir}/covariances", exist_ok=True)
        os.makedirs(f"{run_dir}/{covariance_type}_covariance_sharded", exist_ok=True)

        # Create dummy covariance data
        data = {
            "layer1": torch.randn(10, 10),
            "layer2": torch.randn(20, 20),
        }

        # Save ground truth
        save_file(
            TensorDict(data).cpu(),
            f"{gt_dir}/covariances/{covariance_type}_covariance.safetensors",
        )

        # Save run data (same as ground truth for this test)
        save_file(
            TensorDict(data).cpu(),
            f"{run_dir}/{covariance_type}_covariance_sharded/shard_0.safetensors",
        )

        # This should pass without errors
        test_covariances(
            run_path=run_dir,
            ground_truth_path=gt_dir,
            covariance_type=covariance_type,
        )
