"""Self-consistency check for the eigenvector step.

The previous implementation compared the run's eigenvectors against ground truth
column-by-column via cosine similarity. That is unstable under bfloat16: sign
flips and rotations within nearly-degenerate eigenspaces can both drive
``|cos_sim|`` arbitrarily far from 1 without the decomposition being wrong.

Instead we check that the run's own eigenvectors ``Q`` actually diagonalize the
run's normalized covariance ``C``:

    λ_k = Q[:, k]ᵀ C Q[:, k]    (Rayleigh quotient)
    Q diag(λ) Qᵀ ≈ C            (recomposed covariance matches the input)

Any valid eigendecomposition satisfies this identity exactly, regardless of
sign/rotation choices in ``Q`` — which is what lets the test run in bfloat16.
"""

import os

import pytest
import torch

from tests.ekfac_tests.test_utils import load_sharded_covariances


def _load_run_normalized_covariance(
    ekfac_results_path: str, covariance_type: str
) -> dict[str, torch.Tensor]:
    """Return the run's symmetrized, normalized covariance in float64.

    Matches the promotion bergson applies internally before ``torch.linalg.eigh``.
    """
    cov = load_sharded_covariances(
        os.path.join(ekfac_results_path, f"{covariance_type}_sharded")
    )
    total = torch.load(
        os.path.join(ekfac_results_path, "total_processed.pt"), weights_only=True
    ).item()
    out: dict[str, torch.Tensor] = {}
    for name, C in cov.items():
        C64 = C.to(torch.float64)
        out[name] = ((C64 + C64.T) / 2) / total
    return out


@pytest.mark.parametrize("eigenvector_type", ["activation", "gradient"])
def test_eigenvectors_recompose_covariance(
    ekfac_results_path: str,
    eigenvector_type: str,
) -> None:
    print(f"\nTesting {eigenvector_type} eigenvector recomposition...")

    normalized_cov = _load_run_normalized_covariance(
        ekfac_results_path, eigenvector_type
    )
    eigenvectors = load_sharded_covariances(
        os.path.join(ekfac_results_path, f"eigen_{eigenvector_type}_sharded")
    )

    errors: list[tuple[str, float]] = []
    for name, C in normalized_cov.items():
        Q = eigenvectors[name].to(torch.float64)
        lambdas = (Q.T @ C @ Q).diagonal()  # [d]
        C_recomposed = (Q * lambdas) @ Q.T
        rel = (C_recomposed - C).norm() / C.norm().clamp_min(1e-12)
        errors.append((name, rel.item()))

    max_err = max(e for _, e in errors)
    # Exact in infinite precision; bfloat16-stored Q gives headroom ~1 ULP ≈ 4e-3.
    atol = 5e-3
    assert max_err < atol, (
        f"{eigenvector_type} eigenvector recomposition rel_error={max_err:.2e} "
        f">= atol={atol}\n"
        + "\n".join(f"  {n}: rel_err={e:.2e}" for n, e in errors)
    )
    print(
        f"{eigenvector_type} eigenvectors recompose covariance "
        f"(max rel_err={max_err:.2e})"
    )
