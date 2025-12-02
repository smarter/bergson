"""
Unit tests to verify the sliced covariance collection logic matches the original KFAC.

The original collect_kfac_multilayer.py uses [:, :-1] slicing to drop the last
sequence position. This test verifies that the same slicing logic produces
identical covariance matrices.
"""
import tempfile
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Original KFAC implementation (from collect_kfac_multilayer.py)
# ============================================================================
class OriginalKFAC:
    """Original KFAC collector that drops last position with [:, :-1]."""

    def __init__(self, layer: nn.Linear):
        d_out, d_in = layer.weight.shape
        self.A = torch.zeros(
            d_in, d_in, dtype=torch.float32, device=layer.weight.device
        )
        self.G = torch.zeros(
            d_out, d_out, dtype=torch.float32, device=layer.weight.device
        )
        self.n = 0
        self._buf = None
        self._h_fwd = layer.register_forward_pre_hook(self._fwd, prepend=False)
        self._h_bwd = layer.register_full_backward_hook(self._bwd, prepend=False)

    def _fwd(self, _, inp):
        if torch.is_grad_enabled():
            x = inp[0][:, :-1].detach()  # Drop last position
            self._buf = x.reshape(-1, x.size(-1)).float()

    def _bwd(self, _, __, go):
        if go[0] is None or self._buf is None:
            return
        g = go[0][:, :-1].detach().reshape(-1, go[0].size(-1)).float()  # Drop last position
        self.A.add_(self._buf.T @ self._buf)
        self.G.add_(g.T @ g)
        self.n += g.size(0)
        self._buf = None

    def factors(self):
        return self.A / self.n, self.G / self.n

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()
        self._buf = None


# ============================================================================
# Sliced KFAC collector (matching bergson's SlicedCovarianceCollector logic)
# ============================================================================
class SlicedKFAC:
    """KFAC collector with [:, :-1] slicing, matching bergson's SlicedCovarianceCollector."""

    def __init__(self, layer: nn.Linear):
        d_out, d_in = layer.weight.shape
        self.A = torch.zeros(
            d_in, d_in, dtype=torch.float32, device=layer.weight.device
        )
        self.G = torch.zeros(
            d_out, d_out, dtype=torch.float32, device=layer.weight.device
        )
        self.n = 0
        self._h_fwd = layer.register_forward_hook(self._fwd)
        self._h_bwd = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, inp, __):
        """Forward hook matching bergson's SlicedCovarianceCollector.forward_hook."""
        if torch.is_grad_enabled():
            a = inp[0].detach()  # [N, S, I]
            # Drop last position to match original implementation
            a = a[:, :-1, :]
            # Reshape to [N*(S-1), I]
            a_bi = a.reshape(-1, a.shape[-1]).float()
            # Compute covariance: a_bi.mT @ a_bi
            self.A.add_(a_bi.mT @ a_bi)

    def _bwd(self, _, __, go):
        """Backward hook matching bergson's SlicedCovarianceCollector.backward_hook."""
        if go[0] is None:
            return
        g = go[0].detach()  # [N, S, O]
        # Drop last position to match original implementation
        g = g[:, :-1, :]
        # Reshape to [N*(S-1), O]
        g_bo = g.reshape(-1, g.shape[-1]).float()
        # Compute covariance: g_bo.mT @ g_bo
        self.G.add_(g_bo.mT @ g_bo)
        self.n += g_bo.size(0)

    def factors(self):
        return self.A / self.n, self.G / self.n

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ============================================================================
# KFAC collector WITHOUT slicing (for comparison)
# ============================================================================
class UnslicedKFAC:
    """KFAC collector WITHOUT [:, :-1] slicing."""

    def __init__(self, layer: nn.Linear):
        d_out, d_in = layer.weight.shape
        self.A = torch.zeros(
            d_in, d_in, dtype=torch.float32, device=layer.weight.device
        )
        self.G = torch.zeros(
            d_out, d_out, dtype=torch.float32, device=layer.weight.device
        )
        self.n = 0
        self._h_fwd = layer.register_forward_hook(self._fwd)
        self._h_bwd = layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, inp, __):
        if torch.is_grad_enabled():
            a = inp[0].detach()  # [N, S, I] - NO slicing
            a_bi = a.reshape(-1, a.shape[-1]).float()
            self.A.add_(a_bi.mT @ a_bi)

    def _bwd(self, _, __, go):
        if go[0] is None:
            return
        g = go[0].detach()  # [N, S, O] - NO slicing
        g_bo = g.reshape(-1, g.shape[-1]).float()
        self.G.add_(g_bo.mT @ g_bo)
        self.n += g_bo.size(0)

    def factors(self):
        return self.A / self.n, self.G / self.n

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# ============================================================================
# Simple test model
# ============================================================================
class SimpleMLP(nn.Module):
    """Simple MLP for testing covariance collection."""

    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 128):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU-style activation
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden)


class SimpleModel(nn.Module):
    """Simple model with embedding and MLP layers for testing."""

    def __init__(
        self, vocab_size: int = 100, hidden_dim: int = 64, intermediate_dim: int = 128
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.mlp = SimpleMLP(hidden_dim, intermediate_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x = self.mlp(x)
        return self.head(x)


# ============================================================================
# Tests
# ============================================================================
class TestSlicedVsOriginalKFAC:
    """Tests verifying SlicedKFAC matches OriginalKFAC exactly."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device (CPU for consistency)."""
        return torch.device("cpu")

    @pytest.fixture
    def model(self, device: torch.device) -> SimpleModel:
        """Create a simple test model."""
        torch.manual_seed(42)
        return SimpleModel(vocab_size=100, hidden_dim=64, intermediate_dim=128).to(
            device
        )

    @pytest.fixture
    def input_batch(self, device: torch.device) -> Tensor:
        """Create a batch of input tokens."""
        torch.manual_seed(42)
        # [batch_size=4, seq_len=16]
        return torch.randint(0, 100, (4, 16), device=device)

    def test_single_batch_covariances_match(
        self, model: SimpleModel, input_batch: Tensor
    ):
        """Test that SlicedKFAC produces same results as OriginalKFAC for a single batch."""
        # Run original KFAC
        original_collector = OriginalKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        orig_A, orig_G = original_collector.factors()
        orig_n = original_collector.n
        original_collector.close()

        # Run SlicedKFAC (should produce identical results)
        sliced_collector = SlicedKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        sliced_A, sliced_G = sliced_collector.factors()
        sliced_n = sliced_collector.n
        sliced_collector.close()

        # Token counts should match
        assert orig_n == sliced_n, f"Token count mismatch: {orig_n} vs {sliced_n}"

        # Activation covariance should match exactly
        torch.testing.assert_close(
            sliced_A,
            orig_A,
            rtol=1e-5,
            atol=1e-6,
            msg="Activation covariance matrices don't match",
        )

        # Gradient covariance should match exactly
        torch.testing.assert_close(
            sliced_G,
            orig_G,
            rtol=1e-5,
            atol=1e-6,
            msg="Gradient covariance matrices don't match",
        )

    def test_multiple_batches_accumulation(
        self, model: SimpleModel, device: torch.device
    ):
        """Test that covariances accumulate correctly across multiple batches."""
        torch.manual_seed(42)
        batches = [torch.randint(0, 100, (2, 8), device=device) for _ in range(3)]

        # Run original KFAC
        original_collector = OriginalKFAC(model.mlp.down_proj)

        for batch in batches:
            model.zero_grad()
            logits = model(batch)
            loss = logits.sum()
            loss.backward()

        orig_A, orig_G = original_collector.factors()
        orig_n = original_collector.n
        original_collector.close()

        # Run SlicedKFAC
        sliced_collector = SlicedKFAC(model.mlp.down_proj)

        for batch in batches:
            model.zero_grad()
            logits = model(batch)
            loss = logits.sum()
            loss.backward()

        sliced_A, sliced_G = sliced_collector.factors()
        sliced_n = sliced_collector.n
        sliced_collector.close()

        assert orig_n == sliced_n, f"Token count mismatch: {orig_n} vs {sliced_n}"

        torch.testing.assert_close(
            sliced_A,
            orig_A,
            rtol=1e-5,
            atol=1e-6,
            msg="Accumulated activation covariance matrices don't match",
        )

        torch.testing.assert_close(
            sliced_G,
            orig_G,
            rtol=1e-5,
            atol=1e-6,
            msg="Accumulated gradient covariance matrices don't match",
        )

    def test_multiple_layers(self, model: SimpleModel, input_batch: Tensor):
        """Test collecting covariances from multiple layers simultaneously."""
        # Run original KFAC on all MLP layers
        original_collectors = {
            "gate_proj": OriginalKFAC(model.mlp.gate_proj),
            "up_proj": OriginalKFAC(model.mlp.up_proj),
            "down_proj": OriginalKFAC(model.mlp.down_proj),
        }

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        orig_results = {}
        for name, collector in original_collectors.items():
            A, G = collector.factors()
            orig_results[name] = {"A": A, "G": G}
            collector.close()

        # Run SlicedKFAC on all layers
        sliced_collectors = {
            "gate_proj": SlicedKFAC(model.mlp.gate_proj),
            "up_proj": SlicedKFAC(model.mlp.up_proj),
            "down_proj": SlicedKFAC(model.mlp.down_proj),
        }

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        for name, collector in sliced_collectors.items():
            A, G = collector.factors()
            collector.close()

            torch.testing.assert_close(
                A,
                orig_results[name]["A"],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Activation covariance for {name} doesn't match",
            )

            torch.testing.assert_close(
                G,
                orig_results[name]["G"],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Gradient covariance for {name} doesn't match",
            )


class TestSlicingMakesDifference:
    """Sanity tests verifying that slicing actually changes the result."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cpu")

    @pytest.fixture
    def model(self, device: torch.device) -> SimpleModel:
        torch.manual_seed(42)
        return SimpleModel(vocab_size=100, hidden_dim=64, intermediate_dim=128).to(
            device
        )

    @pytest.fixture
    def input_batch(self, device: torch.device) -> Tensor:
        torch.manual_seed(42)
        return torch.randint(0, 100, (4, 16), device=device)

    def test_sliced_vs_unsliced_different(
        self, model: SimpleModel, input_batch: Tensor
    ):
        """Verify that slicing produces different results than not slicing."""
        # Collect WITH slicing
        sliced_collector = SlicedKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        sliced_A, sliced_G = sliced_collector.factors()
        sliced_n = sliced_collector.n
        sliced_collector.close()

        # Collect WITHOUT slicing
        unsliced_collector = UnslicedKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        unsliced_A, unsliced_G = unsliced_collector.factors()
        unsliced_n = unsliced_collector.n
        unsliced_collector.close()

        # Token counts should differ by batch_size (one less position per sequence)
        batch_size, seq_len = input_batch.shape
        expected_sliced_n = batch_size * (seq_len - 1)
        expected_unsliced_n = batch_size * seq_len

        assert sliced_n == expected_sliced_n, f"Sliced n={sliced_n}, expected {expected_sliced_n}"
        assert unsliced_n == expected_unsliced_n, f"Unsliced n={unsliced_n}, expected {expected_unsliced_n}"

        # Covariance matrices should be DIFFERENT
        assert not torch.allclose(
            sliced_A, unsliced_A, rtol=1e-3, atol=1e-6
        ), "Sliced and unsliced activation covariances are identical - slicing has no effect!"

        assert not torch.allclose(
            sliced_G, unsliced_G, rtol=1e-3, atol=1e-6
        ), "Sliced and unsliced gradient covariances are identical - slicing has no effect!"

    def test_token_count_consistency(self, model: SimpleModel, input_batch: Tensor):
        """Verify token counts are correct for sliced collection."""
        original_collector = OriginalKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        original_n = original_collector.n
        original_collector.close()

        # Calculate expected: batch_size * (seq_len - 1)
        batch_size, seq_len = input_batch.shape
        expected_n = batch_size * (seq_len - 1)

        assert (
            original_n == expected_n
        ), f"Original token count {original_n} != expected {expected_n}"


class TestCovarianceProperties:
    """Test that covariance matrices have expected mathematical properties."""

    @pytest.fixture
    def device(self) -> torch.device:
        return torch.device("cpu")

    @pytest.fixture
    def model(self, device: torch.device) -> SimpleModel:
        torch.manual_seed(42)
        return SimpleModel(vocab_size=100, hidden_dim=64, intermediate_dim=128).to(
            device
        )

    def test_covariance_symmetry(self, model: SimpleModel, device: torch.device):
        """Covariance matrices should be symmetric."""
        torch.manual_seed(42)
        input_batch = torch.randint(0, 100, (4, 16), device=device)

        collector = OriginalKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        A, G = collector.factors()
        collector.close()

        # Check symmetry
        torch.testing.assert_close(
            A, A.T, rtol=1e-5, atol=1e-6, msg="Activation covariance not symmetric"
        )
        torch.testing.assert_close(
            G, G.T, rtol=1e-5, atol=1e-6, msg="Gradient covariance not symmetric"
        )

    def test_covariance_positive_semidefinite(
        self, model: SimpleModel, device: torch.device
    ):
        """Covariance matrices should be positive semi-definite."""
        torch.manual_seed(42)
        input_batch = torch.randint(0, 100, (4, 16), device=device)

        collector = OriginalKFAC(model.mlp.gate_proj)

        model.zero_grad()
        logits = model(input_batch)
        loss = logits.sum()
        loss.backward()

        A, G = collector.factors()
        collector.close()

        # Check eigenvalues are non-negative (with small tolerance for numerical errors)
        A_eigs = torch.linalg.eigvalsh(A)
        G_eigs = torch.linalg.eigvalsh(G)

        assert (
            A_eigs.min() >= -1e-6
        ), f"Activation covariance has negative eigenvalue: {A_eigs.min()}"
        assert (
            G_eigs.min() >= -1e-6
        ), f"Gradient covariance has negative eigenvalue: {G_eigs.min()}"


class TestDirectHookLogic:
    """Test the hook computation logic directly, matching bergson's implementation."""

    def test_forward_hook_slicing_logic(self):
        """Test that the forward hook slicing produces correct results."""
        torch.manual_seed(42)

        # Simulate input activations [N=2, S=5, I=4]
        a = torch.randn(2, 5, 4)

        # Original method: drop last position, then reshape and compute
        a_orig = a[:, :-1].reshape(-1, a.shape[-1]).float()
        A_orig = a_orig.T @ a_orig

        # Bergson method (as in SlicedCovarianceCollector)
        a_sliced = a[:, :-1, :]  # Drop last position
        a_bi = a_sliced.reshape(-1, a_sliced.shape[-1])
        A_bergson = a_bi.mT @ a_bi

        torch.testing.assert_close(
            A_bergson,
            A_orig,
            rtol=1e-5,
            atol=1e-6,
            msg="Forward hook slicing logic mismatch",
        )

    def test_backward_hook_slicing_logic(self):
        """Test that the backward hook slicing produces correct results."""
        torch.manual_seed(42)

        # Simulate gradient output [N=2, S=5, O=8]
        g = torch.randn(2, 5, 8)

        # Original method
        g_orig = g[:, :-1].reshape(-1, g.shape[-1]).float()
        G_orig = g_orig.T @ g_orig

        # Bergson method
        g_sliced = g[:, :-1, :]
        g_bo = g_sliced.reshape(-1, g_sliced.shape[-1])
        G_bergson = g_bo.mT @ g_bo

        torch.testing.assert_close(
            G_bergson,
            G_orig,
            rtol=1e-5,
            atol=1e-6,
            msg="Backward hook slicing logic mismatch",
        )

    def test_covariance_accumulation_logic(self):
        """Test that covariance accumulation works correctly across batches."""
        torch.manual_seed(42)

        d_in = 4
        A_accum = torch.zeros(d_in, d_in)

        batches = [torch.randn(2, 5, d_in) for _ in range(3)]
        total_n = 0

        for a in batches:
            a_sliced = a[:, :-1, :]
            a_bi = a_sliced.reshape(-1, d_in).float()
            A_accum.add_(a_bi.mT @ a_bi)
            total_n += a_bi.shape[0]

        # Verify final result is symmetric
        torch.testing.assert_close(
            A_accum, A_accum.T, rtol=1e-5, atol=1e-6, msg="Accumulated covariance not symmetric"
        )

        # Verify token count
        expected_n = 3 * 2 * 4  # 3 batches * 2 samples * (5-1) positions
        assert total_n == expected_n, f"Token count {total_n} != expected {expected_n}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
