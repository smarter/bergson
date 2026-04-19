"""Regression tests for the bf16 mixed-precision path in bergson/hessians/.

Two distinct failure modes motivated the ``.float()`` additions in
``sharded_computation.py`` and ``kfac.py``:

1. ``ShardedMul._matmul`` / ``_transpose_matmul`` raised
   ``RuntimeError: expected scalar type Float but found BFloat16`` when the
   stored eigenvectors (loaded as fp32 from disk) were combined with bf16
   activations/gradients from the model.

2. ``CovarianceCollector`` used to compute ``a.mT @ a`` in bf16 before adding
   into its fp32 buffer, which silently lost precision on the accumulator.

These tests lock in the fixes so they don't regress.
"""
import torch
import torch.nn as nn

from bergson.hessians.kfac import CovarianceCollector
from bergson.hessians.sharded_computation import ShardedMul


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_shardedmul_matmul_accepts_bf16_vector_with_fp32_matrix(tmp_path):
    """bf16 vector × fp32 matrix must not raise, and must return fp32."""
    device = _device()
    torch.manual_seed(0)
    vector = torch.randn(2, 4, 8, dtype=torch.bfloat16, device=device)
    matrix = torch.randn(8, 6, dtype=torch.float32, device=device)

    result = ShardedMul()._matmul(vector_nsa=vector, matrix_cb=matrix)

    assert result.dtype == torch.float32
    assert result.shape == (2, 4, 6)
    expected = torch.einsum("n s c, c b -> n s b", vector.float(), matrix)
    torch.testing.assert_close(result, expected)


def test_shardedmul_transpose_matmul_accepts_bf16_vector_with_fp32_matrix(tmp_path):
    device = _device()
    torch.manual_seed(0)
    vector = torch.randn(2, 4, 8, dtype=torch.bfloat16, device=device)
    matrix = torch.randn(6, 8, dtype=torch.float32, device=device)

    result = ShardedMul()._transpose_matmul(vector_nsa=vector, matrix_cb=matrix)

    assert result.dtype == torch.float32
    assert result.shape == (2, 4, 6)
    expected = torch.einsum("n s c, b c -> n s b", vector.float(), matrix)
    torch.testing.assert_close(result, expected)


def test_shardedmul_matmul_preserves_fp32_fast_path(tmp_path):
    """The fp32 fast path must still be bitwise-equivalent to a plain einsum."""
    device = _device()
    torch.manual_seed(0)
    vector = torch.randn(3, 2, 5, dtype=torch.float32, device=device)
    matrix = torch.randn(5, 7, dtype=torch.float32, device=device)

    result = ShardedMul()._matmul(vector_nsa=vector, matrix_cb=matrix)

    assert result.dtype == torch.float32
    expected = torch.einsum("n s c, c b -> n s b", vector, matrix)
    torch.testing.assert_close(result, expected)


def _make_collector(
    in_dim: int, out_dim: int, device: torch.device, path: str
) -> tuple[CovarianceCollector, nn.Linear]:
    linear = nn.Linear(in_dim, out_dim, bias=False, device=device, dtype=torch.bfloat16)
    collector = CovarianceCollector(
        model=linear,
        target_modules={""},
        dtype=torch.float32,
        path=path,
    )
    linear._name = ""  # hooks normally set this inside __enter__
    return collector, linear


def test_covariance_collector_accumulates_activations_in_fp32_for_bf16_model(tmp_path):
    """bf16 activations must be promoted to fp32 before the outer product so the
    fp32 accumulator sees the full-precision result."""
    device = _device()
    torch.manual_seed(0)
    in_dim, out_dim = 8, 4
    collector, linear = _make_collector(in_dim, out_dim, device, str(tmp_path))

    batch, seq = 2, 3
    activations = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16, device=device)
    mask = torch.ones(batch, seq, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        collector.forward_hook(linear, activations)

    assert collector.A_cov_dict[""].dtype == torch.float32
    a_flat = activations.reshape(-1, in_dim).float()
    torch.testing.assert_close(collector.A_cov_dict[""], a_flat.mT @ a_flat)


def test_covariance_collector_accumulates_gradients_in_fp32_for_bf16_model(tmp_path):
    device = _device()
    torch.manual_seed(0)
    in_dim, out_dim = 8, 4
    collector, linear = _make_collector(in_dim, out_dim, device, str(tmp_path))

    batch, seq = 2, 3
    grads = torch.randn(batch, seq, out_dim, dtype=torch.bfloat16, device=device)
    mask = torch.ones(batch, seq, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        collector.backward_hook(linear, grads)

    assert collector.S_cov_dict[""].dtype == torch.float32
    g_flat = grads.reshape(-1, out_dim).float()
    torch.testing.assert_close(collector.S_cov_dict[""], g_flat.mT @ g_flat)


def test_covariance_collector_beats_bf16_matmul_precision(tmp_path):
    """The fix exists because running ``a.mT @ a`` in bf16 leaks ~1e-2 of
    relative error into the fp32 accumulator on realistic inner dimensions.
    The fp32-promoted path should recover all of that precision."""
    device = _device()
    torch.manual_seed(0)
    # Inner dim is large enough that bf16 accumulation error is visible.
    in_dim, out_dim = 256, 128
    collector, linear = _make_collector(in_dim, out_dim, device, str(tmp_path))

    batch, seq = 4, 32
    activations = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16, device=device)
    mask = torch.ones(batch, seq, dtype=torch.bool, device=device)

    with collector.with_batch(mask):
        collector.forward_hook(linear, activations)

    a_flat_fp32 = activations.reshape(-1, in_dim).float()
    expected_fp32 = a_flat_fp32.mT @ a_flat_fp32

    a_flat_bf16 = activations.reshape(-1, in_dim)
    bf16_native = (a_flat_bf16.mT @ a_flat_bf16).float()

    fp32_err = (collector.A_cov_dict[""] - expected_fp32).norm() / expected_fp32.norm()
    bf16_err = (bf16_native - expected_fp32).norm() / expected_fp32.norm()

    torch.testing.assert_close(collector.A_cov_dict[""], expected_fp32)
    assert fp32_err < bf16_err / 100, (
        f"Expected fp32-promoted path to be >=100x tighter than the bf16 matmul, "
        f"got fp32_err={fp32_err.item():.3e} vs bf16_err={bf16_err.item():.3e}"
    )
