#!/usr/bin/env python
"""
Compare K-FAC factors from original and bergson implementations.

Loads both outputs and computes similarity metrics:
- Frobenius norm difference
- Relative error
- Cosine similarity
"""
import argparse
import json
import pathlib
from typing import Dict, Tuple
import torch
import numpy as np


def parse():
    p = argparse.ArgumentParser(
        description="Compare K-FAC factors from original and bergson implementations"
    )
    p.add_argument(
        "--original_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing original .pt files",
    )
    p.add_argument(
        "--bergson_converted_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing converted bergson .pt files",
    )
    p.add_argument(
        "--output_json",
        type=pathlib.Path,
        default="kfac_comparison_results.json",
        help="Output JSON file for comparison metrics",
    )
    return p.parse_args()


def load_original_factors(original_dir: pathlib.Path) -> Dict[str, Dict]:
    """Load factors from original format."""
    factors = {}
    for pt_file in sorted(original_dir.glob("kfac_factors_blk_*.pt")):
        data = torch.load(pt_file, map_location="cpu")
        factors.update(data)
    return factors


def compute_similarity_metrics(
    A1: torch.Tensor, A2: torch.Tensor
) -> Dict[str, float]:
    """Compute similarity metrics between two matrices."""
    # Ensure float32 for numerical stability
    A1 = A1.float()
    A2 = A2.float()

    # Frobenius norm difference
    diff = A1 - A2
    frob_diff = torch.norm(diff, p="fro").item()

    # Relative error
    frob_A1 = torch.norm(A1, p="fro").item()
    rel_error = frob_diff / (frob_A1 + 1e-10)

    # Cosine similarity
    A1_flat = A1.flatten()
    A2_flat = A2.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        A1_flat.unsqueeze(0), A2_flat.unsqueeze(0), dim=1
    ).item()

    # Max absolute difference
    max_abs_diff = torch.max(torch.abs(diff)).item()

    # Mean absolute difference
    mean_abs_diff = torch.mean(torch.abs(diff)).item()

    return {
        "frobenius_norm_diff": frob_diff,
        "relative_error": rel_error,
        "cosine_similarity": cos_sim,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def compare_factors(
    original_dir: pathlib.Path, bergson_converted_dir: pathlib.Path
) -> Dict:
    """Compare factors from both implementations."""
    print(f"Loading original factors from {original_dir}")
    original_factors = load_original_factors(original_dir)

    print(f"Loading bergson (converted) factors from {bergson_converted_dir}")
    bergson_factors = load_original_factors(bergson_converted_dir)

    print(f"\nOriginal factors: {len(original_factors)} modules")
    print(f"Bergson factors: {len(bergson_factors)} modules")

    # Find common keys
    original_keys = set(original_factors.keys())
    bergson_keys = set(bergson_factors.keys())
    common_keys = original_keys & bergson_keys

    print(f"\nCommon modules: {len(common_keys)}")
    if len(original_keys - bergson_keys) > 0:
        print(f"Only in original: {sorted(original_keys - bergson_keys)}")
    if len(bergson_keys - original_keys) > 0:
        print(f"Only in bergson: {sorted(bergson_keys - original_keys)}")

    # Compare each module
    results = {
        "summary": {
            "n_modules": len(common_keys),
            "original_only": sorted(list(original_keys - bergson_keys)),
            "bergson_only": sorted(list(bergson_keys - original_keys)),
        },
        "modules": {},
    }

    for key in sorted(common_keys):
        orig = original_factors[key]
        berg = bergson_factors[key]

        print(f"\n{'='*60}")
        print(f"Module: {key}")
        print(f"{'='*60}")

        # Compare A matrices
        print("\nActivation Covariance (A):")
        print(f"  Original shape: {orig['A'].shape}")
        print(f"  Bergson shape:  {berg['A'].shape}")
        A_metrics = compute_similarity_metrics(orig["A"], berg["A"])
        for metric, value in A_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Compare G matrices
        print("\nGradient Covariance (G):")
        print(f"  Original shape: {orig['G'].shape}")
        print(f"  Bergson shape:  {berg['G'].shape}")
        G_metrics = compute_similarity_metrics(orig["G"], berg["G"])
        for metric, value in G_metrics.items():
            print(f"  {metric}: {value:.6f}")

        # Token counts
        print(f"\nToken counts:")
        print(f"  Original: {orig.get('n_tokens', 'N/A'):,}")
        print(f"  Bergson:  {berg.get('n_tokens', 'N/A'):,}")

        results["modules"][key] = {
            "A": A_metrics,
            "G": G_metrics,
            "n_tokens_original": orig.get("n_tokens", 0),
            "n_tokens_bergson": berg.get("n_tokens", 0),
        }

    # Compute aggregate statistics
    all_A_rel_errors = [
        m["A"]["relative_error"] for m in results["modules"].values()
    ]
    all_G_rel_errors = [
        m["G"]["relative_error"] for m in results["modules"].values()
    ]
    all_A_cos_sims = [
        m["A"]["cosine_similarity"] for m in results["modules"].values()
    ]
    all_G_cos_sims = [
        m["G"]["cosine_similarity"] for m in results["modules"].values()
    ]

    results["summary"]["aggregate"] = {
        "A": {
            "mean_relative_error": float(np.mean(all_A_rel_errors)),
            "max_relative_error": float(np.max(all_A_rel_errors)),
            "mean_cosine_similarity": float(np.mean(all_A_cos_sims)),
            "min_cosine_similarity": float(np.min(all_A_cos_sims)),
        },
        "G": {
            "mean_relative_error": float(np.mean(all_G_rel_errors)),
            "max_relative_error": float(np.max(all_G_rel_errors)),
            "mean_cosine_similarity": float(np.mean(all_G_cos_sims)),
            "min_cosine_similarity": float(np.min(all_G_cos_sims)),
        },
    }

    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}")
    print("\nActivation Covariance (A):")
    for metric, value in results["summary"]["aggregate"]["A"].items():
        print(f"  {metric}: {value:.12f}")
    print("\nGradient Covariance (G):")
    for metric, value in results["summary"]["aggregate"]["G"].items():
        print(f"  {metric}: {value:.12f}")

    return results


def main():
    args = parse()

    results = compare_factors(args.original_dir, args.bergson_converted_dir)

    # Save results
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Comparison results saved to {args.output_json}")


if __name__ == "__main__":
    main()
