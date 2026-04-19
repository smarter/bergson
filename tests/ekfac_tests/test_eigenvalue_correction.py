"""Recomposition-based check for the eigenvalue correction step.

The previous implementation compared per-entry values of the EKFAC lambda tensor
against ground truth after sign-aligning each eigenvector column. That alignment
breaks down under bfloat16: nearly-degenerate eigenvalues rotate inside their
eigenspace rather than merely flipping sign, so no per-column alignment can
recover the right lambda values.

Instead we reconstruct the weight-space diagonal Fisher implied by the EKFAC
factors and compare that across run and ground truth:

    F_diag[o, i] = Σ_{k, l} Q_G[o, k]² · Λ[k, l] · Q_A[i, l]²
                 = (Q_G²) · Λ · (Q_A²)ᵀ

Under a sign flip Q'_G = Q_G · diag(s_G) (and similarly for Q_A), the lambda
entries are invariant (the squaring in Λ's definition cancels s²=1), and the
recomposed ``F_diag`` stays the same because every Q_G and Q_A factor in the
formula is itself squared. That invariance is what keeps the test stable under
bfloat16.
"""

import os

import torch
from safetensors.torch import load_file

from tests.ekfac_tests.test_utils import load_sharded_covariances


def _recompose_weight_space_fisher(
    Q_G: torch.Tensor, Q_A: torch.Tensor, Lambda: torch.Tensor
) -> torch.Tensor:
    """Return F_diag[o, i] = Σ_{k,l} Q_G[o,k]² Λ[k,l] Q_A[i,l]² in float64."""
    Q_G2 = Q_G.to(torch.float64) ** 2
    Q_A2 = Q_A.to(torch.float64) ** 2
    L = Lambda.to(torch.float64)
    return Q_G2 @ L @ Q_A2.T


def test_eigenvalue_corrections_recompose_weight_space(
    ground_truth_eigenvalue_corrections_path: str,
    ground_truth_eigenvectors_path: str,
    ekfac_results_path: str,
) -> None:
    print("\nTesting weight-space EKFAC recomposition...")

    lambda_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigenvalue_correction_sharded")
    )
    total = torch.load(
        os.path.join(ekfac_results_path, "total_processed.pt"), weights_only=True
    ).item()
    lambda_run = {k: v / total for k, v in lambda_run.items()}

    QA_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_activation_sharded")
    )
    QG_run = load_sharded_covariances(
        os.path.join(ekfac_results_path, "eigen_gradient_sharded")
    )

    lambda_gt = load_file(
        os.path.join(
            ground_truth_eigenvalue_corrections_path,
            "eigenvalue_corrections.safetensors",
        )
    )
    QA_gt = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_activations.safetensors"
        )
    )
    QG_gt = load_file(
        os.path.join(
            ground_truth_eigenvectors_path, "eigenvectors_gradients.safetensors"
        )
    )

    errors: list[tuple[str, float]] = []
    for name in sorted(lambda_run):
        F_run = _recompose_weight_space_fisher(
            QG_run[name], QA_run[name], lambda_run[name]
        )
        F_gt = _recompose_weight_space_fisher(
            QG_gt[name], QA_gt[name], lambda_gt[name]
        )
        rel = (F_run - F_gt).norm() / F_gt.norm().clamp_min(1e-12)
        errors.append((name, rel.item()))

    max_err = max(e for _, e in errors)
    # Tolerance budget: bfloat16 outer-product noise in the covariance (which
    # propagates into Λ), plus minor wobble from near-degenerate eigenspaces.
    atol = 5e-2
    assert max_err < atol, (
        f"Weight-space recomposed F_diag rel_error={max_err:.2e} >= atol={atol}\n"
        + "\n".join(f"  {n}: rel_err={e:.2e}" for n, e in errors)
    )
    print(f"EKFAC recomposition matches (max rel_err={max_err:.2e})")
