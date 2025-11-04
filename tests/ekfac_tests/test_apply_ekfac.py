"""Test EKFAC application to gradients against ground truth."""

import json
import os

import pytest
import torch
from safetensors.torch import load_file

from bergson.data import DataConfig, IndexConfig, load_gradients
from bergson.distributed import distributed_computing
from bergson.hessians.ekfac_apply import ekfac_apply_worker


@pytest.fixture(scope="module")
def ekfac_apply_config(
    test_dir, ground_truth_path, use_fsdp, world_size, overwrite, run_path,
    gradient_path, gradient_batch_size
):
    """Create and configure EKFAC application."""
    if gradient_path is None:
        pytest.skip("Gradient path not provided. Use --gradient-path")

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

    print(cfg_json)
    cfg = IndexConfig(**cfg_json)
    cfg.data = DataConfig(**(cfg_json["data"]))

    cfg.run_path = test_dir + "/run"
    cfg.debug = True
    cfg.fsdp = use_fsdp
    cfg.world_size = world_size
    cfg.ekfac = True
    cfg.gradient_path = gradient_path
    cfg.gradient_batch_size = gradient_batch_size

    # Run EKFAC application if needed
    if not os.path.exists(run_path) or overwrite:
        distributed_computing(
            cfg=cfg,
            worker_fn=ekfac_apply_worker,
            setup_data=False,
            setup_model=False,
            setup_processor=False,
        )

        print("EKFAC application completed successfully.")
    else:
        print("Using existing run directory.")

    return cfg


@pytest.mark.requires_test_data
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_applied_gradients(test_dir, ekfac_apply_config):
    """Test that EKFAC-transformed gradients match ground truth."""
    gradient_path = ekfac_apply_config.gradient_path
    run_path = gradient_path + "_ekfac"
    ground_truth_path = test_dir + "/test_gradients/gradients_after_ekfac"

    if not os.path.exists(os.path.join(ground_truth_path, "gradients.safetensors")):
        pytest.skip("Ground truth gradients not found")

    ground_truth = load_file(
        os.path.join(ground_truth_path, "gradients.safetensors"), device="cuda"
    )
    computed_mmap = load_gradients(run_path)

    mismatches = []

    for k in ground_truth.keys():
        ground_truth_tensor = ground_truth[k].to(dtype=torch.float32)

        computed_tensor = (
            torch.from_numpy(computed_mmap[k].copy())
            .to(device="cuda")
            .view(-1, *ground_truth_tensor.shape[1:])
        ).to(dtype=torch.float32)

        # Check shapes match
        assert ground_truth_tensor.shape == computed_tensor.shape, (
            f"Shape mismatch for key {k}: "
            f"{ground_truth_tensor.shape} vs {computed_tensor.shape}"
        )

        # Check values match within tolerance
        if not torch.allclose(ground_truth_tensor, computed_tensor, rtol=1e-3, atol=0):
            abs_diff = torch.abs(ground_truth_tensor - computed_tensor)
            rel_diff = abs_diff / (torch.abs(ground_truth_tensor) + 1e-12)

            max_abs_diff = torch.max(abs_diff).item()
            max_rel_diff = torch.max(rel_diff).item()
            argmax_idx = torch.argmax(rel_diff)
            coords = torch.unravel_index(argmax_idx, ground_truth_tensor.shape)

            gt_val = ground_truth_tensor.flatten()[argmax_idx].item()
            comp_val = computed_tensor.flatten()[argmax_idx].item()

            mismatch_msg = (
                f"Mismatch '{k}': max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}\n"
                f"  At {tuple(coords)}: gt={gt_val:.2e}, comp={comp_val:.2e}"
            )
            print(mismatch_msg)
            mismatches.append(mismatch_msg)

    # If there were any mismatches, fail the test
    assert len(mismatches) == 0, f"Found {len(mismatches)} gradient mismatches:\n" + "\n".join(mismatches)
