#!/usr/bin/env python
"""
Reproduce Figure 2 from the paper: K-FAC eigenvector activation ratios.

Plots memorized/clean activation ratio across MLP layers for different
percentile bands of K-FAC eigenvectors (ordered by FIM).

Paper Section 4: "Figure 2 shows our results. In both LMs and ViTs, K-FAC
shows a more salient divergence between different parts of the eigenspectrum
on memorized and clean data."
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dvclive import Live
from tqdm import tqdm

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from metrics.memorization_evaluator import MemorizationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2: K-FAC eigenvector activation ratios"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["1b", "7b"],
        required=True,
        help="Model size",
    )
    parser.add_argument(
        "--kfac-dir",
        type=Path,
        required=True,
        help="Directory containing K-FAC factors (.pt files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("memorization_kfac/plots/figure2"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to use for each dataset (memorized/clean)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["up", "gate", "down"],
        default="up",
        help="Which MLP projection to analyze",
    )
    return parser.parse_args()


def load_kfac_factors(kfac_dir: Path, layer_idx: int) -> Dict[str, torch.Tensor]:
    """Load K-FAC factors (A and G matrices) for a specific layer."""
    # Find the .pt file containing this layer
    for pt_file in kfac_dir.glob("kfac_factors_blk_*.pt"):
        factors = torch.load(pt_file, map_location="cpu")

        # Check if this layer is in this file
        for key in factors.keys():
            if key.startswith(f"blk{layer_idx}."):
                return factors

    raise FileNotFoundError(f"No K-FAC factors found for layer {layer_idx}")


def get_fim_ordered_eigenvectors(
    A: torch.Tensor, G: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigendecompositions and order by FIM (Fisher Information Matrix).

    Returns:
        eigenvalues_A: Eigenvalues of A
        eigenvectors_A: Eigenvectors of A (columns)
        fim_order: Indices ordering eigenvectors by FIM importance
    """
    # Compute eigendecompositions
    eigenvalues_A, eigenvectors_A = torch.linalg.eigh(A)
    eigenvalues_G, eigenvectors_G = torch.linalg.eigh(G)

    # Convert to numpy for easier manipulation
    eigenvalues_A = eigenvalues_A.numpy()
    eigenvalues_G = eigenvalues_G.numpy()
    eigenvectors_A = eigenvectors_A.numpy()

    # Sort in descending order
    eigenvalues_A = eigenvalues_A[::-1]
    eigenvalues_G = eigenvalues_G[::-1]
    eigenvectors_A = eigenvectors_A[:, ::-1]

    # Compute FIM importance for each pair (i, j) as λ_i^G * λ_j^A
    # For ordering activation eigenvectors, we sum over all G eigenvalues
    fim_importance = np.zeros(len(eigenvalues_A))
    for j in range(len(eigenvalues_A)):
        # Importance of j-th activation eigenvector
        fim_importance[j] = sum(eigenvalues_G[i] * eigenvalues_A[j]
                                for i in range(len(eigenvalues_G)))

    # Order by FIM importance (descending)
    fim_order = np.argsort(fim_importance)[::-1]

    return eigenvalues_A, eigenvectors_A, fim_order


def get_percentile_bands(n_eigenvectors: int) -> Dict[str, np.ndarray]:
    """Get indices for each percentile band."""
    return {
        "top_10": np.arange(0, int(0.10 * n_eigenvectors)),
        "10_25": np.arange(int(0.10 * n_eigenvectors), int(0.25 * n_eigenvectors)),
        "25_50": np.arange(int(0.25 * n_eigenvectors), int(0.50 * n_eigenvectors)),
        "bottom_50": np.arange(int(0.50 * n_eigenvectors), n_eigenvectors),
    }


def compute_activation_strength(
    activations: torch.Tensor,
    eigenvectors_A: np.ndarray,
    fim_order: np.ndarray,
    band_indices: np.ndarray,
) -> float:
    """
    Compute activation strength for a percentile band.

    Projects activations onto eigenvectors and computes norm.
    """
    # activations: [batch_size, seq_len, hidden_dim]
    # Flatten to [n_tokens, hidden_dim]
    activations = activations.reshape(-1, activations.shape[-1]).numpy()

    # Get eigenvectors for this band (in FIM order)
    band_eigenvectors_indices = fim_order[band_indices]
    band_eigenvectors = eigenvectors_A[:, band_eigenvectors_indices]

    # Project activations onto these eigenvectors
    # projected: [n_tokens, n_band_eigenvectors]
    projected = activations @ band_eigenvectors

    # Compute average norm
    norms = np.linalg.norm(projected, axis=1)
    return float(np.mean(norms))


def collect_activations(
    model,
    tokenizer,
    data_loader,
    layer_idx: int,
    projection: str,
) -> torch.Tensor:
    """
    Collect activations from a specific layer and projection.

    Returns:
        activations: [batch_size, seq_len, hidden_dim]
    """
    activations_list = []

    # Hook to capture activations
    def hook_fn(module, input, output):
        # Input is a tuple, get the first element
        act = input[0] if isinstance(input, tuple) else input
        activations_list.append(act.detach().cpu())

    # Register hook
    layer = model.model.layers[layer_idx]
    if projection == "up":
        handle = layer.mlp.up_proj.register_forward_hook(hook_fn)
    elif projection == "gate":
        handle = layer.mlp.gate_proj.register_forward_hook(hook_fn)
    elif projection == "down":
        handle = layer.mlp.down_proj.register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown projection: {projection}")

    # Run inference
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch

            # Ensure input_ids is on the right device
            if hasattr(model, "device"):
                input_ids = input_ids.to(model.device)

            # Forward pass (hook will capture activations)
            model(input_ids)

    # Remove hook
    handle.remove()

    # Concatenate all activations
    all_activations = torch.cat(activations_list, dim=0)
    return all_activations


def analyze_layer(
    model,
    tokenizer,
    evaluator: MemorizationEvaluator,
    kfac_dir: Path,
    layer_idx: int,
    projection: str,
    n_samples: int,
    batch_size: int,
) -> Dict[str, float]:
    """
    Analyze activation ratios for a single layer.

    Returns dict with ratios for each percentile band.
    """
    # Load K-FAC factors for this layer
    try:
        factors = load_kfac_factors(kfac_dir, layer_idx)
        key = f"blk{layer_idx}.{projection}"
        if key not in factors:
            print(f"Warning: {key} not found in K-FAC factors, skipping layer {layer_idx}")
            return None

        A = factors[key]["A"]
        G = factors[key]["G"]
    except FileNotFoundError:
        print(f"Warning: No K-FAC factors for layer {layer_idx}, skipping")
        return None

    # Get FIM-ordered eigenvectors
    eigenvalues_A, eigenvectors_A, fim_order = get_fim_ordered_eigenvectors(A, G)

    # Get percentile bands
    bands = get_percentile_bands(len(eigenvalues_A))

    # Get memorized and clean data loaders
    from torch.utils.data import DataLoader, TensorDataset

    # Get memorized sequences
    mem_dataset = evaluator.memorization_dataset
    mem_samples = mem_dataset.prefix_ids[:n_samples]
    mem_loader = DataLoader(
        TensorDataset(mem_samples),
        batch_size=batch_size,
        shuffle=False,
    )

    # Get clean sequences
    clean_dataset = evaluator.clean_dataset
    clean_samples = clean_dataset.prefix_ids[:n_samples]
    clean_loader = DataLoader(
        TensorDataset(clean_samples),
        batch_size=batch_size,
        shuffle=False,
    )

    # Collect activations
    print(f"  Collecting activations for layer {layer_idx}...")
    mem_activations = collect_activations(
        model, tokenizer, mem_loader, layer_idx, projection
    )
    clean_activations = collect_activations(
        model, tokenizer, clean_loader, layer_idx, projection
    )

    # Compute activation strengths for each band
    results = {}
    for band_name, band_indices in bands.items():
        mem_strength = compute_activation_strength(
            mem_activations, eigenvectors_A, fim_order, band_indices
        )
        clean_strength = compute_activation_strength(
            clean_activations, eigenvectors_A, fim_order, band_indices
        )

        # Compute ratio
        ratio = mem_strength / clean_strength if clean_strength > 0 else 0.0
        results[band_name] = ratio

    return results


def main():
    args = parse_args()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model_size} model...")
    if args.model_size == "7b":
        model_name = "allenai/OLMo-2-1124-7B"
        n_layers = 32
    else:
        model_name = "allenai/OLMo-2-0425-1B"
        n_layers = 16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Initialize evaluator to get datasets
    print("Initializing evaluator and loading datasets...")
    evaluator = MemorizationEvaluator(model, tokenizer, args.model_size)

    # Analyze each layer
    results_by_layer = {}

    print(f"\nAnalyzing {n_layers} layers for {args.projection}_proj...")
    for layer_idx in tqdm(range(n_layers)):
        layer_results = analyze_layer(
            model,
            tokenizer,
            evaluator,
            args.kfac_dir,
            layer_idx,
            args.projection,
            args.n_samples,
            args.batch_size,
        )

        if layer_results is not None:
            results_by_layer[layer_idx] = layer_results

    # Save results
    results_file = args.output_dir / f"figure2_data_{args.model_size}_{args.projection}.json"
    with open(results_file, "w") as f:
        json.dump(results_by_layer, f, indent=2)

    print(f"\n✓ Saved analysis results to {results_file}")

    # Plot using dvclive
    print("\nGenerating plot with dvclive...")
    with Live(str(args.output_dir)) as live:
        # Prepare data for plotting
        layers = sorted(results_by_layer.keys())

        for band_name in ["top_10", "10_25", "25_50", "bottom_50"]:
            ratios = [results_by_layer[layer][band_name] for layer in layers]

            # Create plot data
            for layer, ratio in zip(layers, ratios):
                live.log_metric(f"{band_name}_ratio", ratio, step=layer)

        # Log plot configuration
        live.log_plot(
            "activation_ratios",
            layers,
            {
                "Top 10%": [results_by_layer[l]["top_10"] for l in layers],
                "10-25%": [results_by_layer[l]["10_25"] for l in layers],
                "25-50%": [results_by_layer[l]["25_50"] for l in layers],
                "Bottom 50%": [results_by_layer[l]["bottom_50"] for l in layers],
            },
            template="linear",
            title=f"Figure 2: K-FAC Eigenvector Activation Ratios ({args.projection}_proj)",
            x_label="MLP Layer",
            y_label="Memorized / Clean Activation Ratio",
        )

    print(f"✓ Generated plots in {args.output_dir}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: Figure 2 Reproduction ({args.model_size.upper()}, {args.projection}_proj)")
    print("=" * 70)

    # Find layer with maximum separation
    max_separation_layer = None
    max_separation = 0

    for layer in layers:
        # Separation is difference between bottom 50% and top 10%
        separation = abs(results_by_layer[layer]["bottom_50"] - results_by_layer[layer]["top_10"])
        if separation > max_separation:
            max_separation = separation
            max_separation_layer = layer

    if max_separation_layer is not None:
        r = results_by_layer[max_separation_layer]
        print(f"\nMaximum separation at layer {max_separation_layer}:")
        print(f"  Top 10%:    {r['top_10']:.4f}")
        print(f"  10-25%:     {r['10_25']:.4f}")
        print(f"  25-50%:     {r['25_50']:.4f}")
        print(f"  Bottom 50%: {r['bottom_50']:.4f}")

        # Compare to paper (example from Section 4)
        if args.model_size == "7b" and max_separation_layer == 22:
            print("\nPaper reports for layer 22:")
            print("  Bottom 50%: 23.1% higher on memorized than clean")
            print("  Top 10%: 26% higher on clean than memorized")


if __name__ == "__main__":
    main()
