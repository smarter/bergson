#!/usr/bin/env python
"""Investigate whether bf16 gradient errors are due to vanishing gradients.

This script compares gradient magnitudes between fp32 and bf16 models to test
the hypothesis that early layer errors are caused by gradient underflow in bf16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from bergson.utils.utils import setup_reproducibility


def get_activation_gradient_covariances(
    model_name: str,
    dtype: torch.dtype,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Run forward/backward and collect activation/gradient covariances."""
    setup_reproducibility()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Storage for activations and gradients
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def make_hooks(name: str):
        def fwd_hook(module: nn.Module, inp, out):
            # Get input activation
            if isinstance(inp, tuple):
                a = inp[0]
            else:
                a = inp
            # Store flattened and converted to float32
            activations[name] = a.detach().reshape(-1, a.shape[-1]).float()

        def bwd_hook(module: nn.Module, grad_in, grad_out):
            # Get output gradient
            if isinstance(grad_out, tuple):
                g = grad_out[0]
            else:
                g = grad_out
            if g is not None:
                gradients[name] = g.detach().reshape(-1, g.shape[-1]).float()

        return fwd_hook, bwd_hook

    # Register hooks on linear layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "layers" in name:
            fwd_hook, bwd_hook = make_hooks(name)
            handles.append(module.register_forward_hook(fwd_hook))
            handles.append(module.register_full_backward_hook(bwd_hook))

    # Forward pass
    logits = model(input_ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels[:, 1:].flatten(),
        reduction="sum",
    )

    # Backward pass
    loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute covariances
    act_covs = {name: (a.T @ a) for name, a in activations.items()}
    grad_covs = {name: (g.T @ g) for name, g in gradients.items()}

    del model
    torch.cuda.empty_cache()

    return act_covs, grad_covs


def compare_covariances(model_name: str = "EleutherAI/pythia-14m", seq_len: int = 128):
    """Compare covariance matrices between fp32 and bf16."""
    print(f"\n{'='*80}")
    print("COVARIANCE COMPARISON")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = tokens["input_ids"]
    labels = input_ids.clone()

    print("\nCollecting fp32 covariances...")
    act_fp32, grad_fp32 = get_activation_gradient_covariances(
        model_name, torch.float32, input_ids, labels
    )

    print("Collecting bf16 covariances...")
    act_bf16, grad_bf16 = get_activation_gradient_covariances(
        model_name, torch.bfloat16, input_ids, labels
    )

    print("\n" + "-" * 80)
    print("ACTIVATION COVARIANCES (A^T @ A)")
    print("-" * 80)
    print(f"{'Layer':<45} {'fp32 norm':>12} {'bf16 norm':>12} {'rel_error':>12}")
    print("-" * 80)

    for name in sorted(act_fp32.keys()):
        if name not in act_bf16:
            continue
        fp32_norm = act_fp32[name].norm().item()
        bf16_norm = act_bf16[name].norm().item()
        rel_error = (act_fp32[name] - act_bf16[name]).norm().item() / fp32_norm if fp32_norm > 0 else 0

        short_name = name.replace("gpt_neox.", "").replace("model.", "")
        print(f"{short_name:<45} {fp32_norm:>12.2e} {bf16_norm:>12.2e} {rel_error:>12.2e}")

    print("\n" + "-" * 80)
    print("GRADIENT COVARIANCES (G^T @ G)")
    print("-" * 80)
    print(f"{'Layer':<45} {'fp32 norm':>12} {'bf16 norm':>12} {'rel_error':>12}")
    print("-" * 80)

    for name in sorted(grad_fp32.keys()):
        if name not in grad_bf16:
            continue
        fp32_norm = grad_fp32[name].norm().item()
        bf16_norm = grad_bf16[name].norm().item()
        rel_error = (grad_fp32[name] - grad_bf16[name]).norm().item() / fp32_norm if fp32_norm > 0 else 0

        short_name = name.replace("gpt_neox.", "").replace("model.", "")
        marker = " <<<<" if rel_error > 10 else ""
        print(f"{short_name:<45} {fp32_norm:>12.2e} {bf16_norm:>12.2e} {rel_error:>12.2e}{marker}")


def get_gradient_stats(
    model_name: str,
    dtype: torch.dtype,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[str, float]]:
    """Run forward/backward and collect gradient statistics per layer."""
    setup_reproducibility()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()

    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # Forward pass
    logits = model(input_ids).logits[:, :-1]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels[:, 1:].flatten(),
        reduction="sum",
    )

    # Backward pass
    loss.backward()

    # Collect gradient statistics for linear layers
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and "weight" in name:
            grad = param.grad
            grad_f32 = grad.float()

            # Check for underflow (values that are zero in bf16 but might not be in fp32)
            num_zeros = (grad == 0).sum().item()
            num_elements = grad.numel()

            stats[name] = {
                "mean_abs": grad_f32.abs().mean().item(),
                "max_abs": grad_f32.abs().max().item(),
                "min_abs_nonzero": grad_f32[grad_f32 != 0].abs().min().item()
                if (grad_f32 != 0).any()
                else 0.0,
                "std": grad_f32.std().item(),
                "num_zeros": num_zeros,
                "pct_zeros": 100 * num_zeros / num_elements,
                "dtype": str(grad.dtype),
            }

    del model
    torch.cuda.empty_cache()

    return stats


def compare_gradients(model_name: str = "EleutherAI/pythia-14m", seq_len: int = 128):
    """Compare gradient statistics between fp32 and bf16."""
    print(f"Model: {model_name}")
    print(f"Sequence length: {seq_len}")
    print("=" * 80)

    # Create dummy input
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use real text for more realistic gradients
    text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    input_ids = tokens["input_ids"]
    labels = input_ids.clone()

    print("\nCollecting fp32 gradients...")
    stats_fp32 = get_gradient_stats(model_name, torch.float32, input_ids, labels)

    print("Collecting bf16 gradients...")
    stats_bf16 = get_gradient_stats(model_name, torch.bfloat16, input_ids, labels)

    # Compare and display results
    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON (fp32 vs bf16)")
    print("=" * 80)

    # bf16 smallest normal number is ~1.2e-38, but practical min is ~6e-8 due to mantissa
    bf16_min_normal = torch.finfo(torch.bfloat16).tiny
    bf16_practical_min = 2**-126  # Approximate smallest representable

    print(f"\nbf16 smallest normal: {bf16_min_normal:.2e}")
    print(f"bf16 practical min (approx): {bf16_practical_min:.2e}")

    print("\n" + "-" * 80)
    print(
        f"{'Layer':<50} {'fp32 mean':>12} {'bf16 mean':>12} {'ratio':>10} {'bf16 %zeros':>12}"
    )
    print("-" * 80)

    # Sort by layer depth (assuming layer names contain numbers)
    layer_names = sorted(stats_fp32.keys())

    for name in layer_names:
        if name not in stats_bf16:
            continue

        fp32 = stats_fp32[name]
        bf16 = stats_bf16[name]

        ratio = bf16["mean_abs"] / fp32["mean_abs"] if fp32["mean_abs"] > 0 else float("inf")

        # Highlight problematic layers
        marker = ""
        if ratio < 0.5 or bf16["pct_zeros"] > 10:
            marker = " <<<< ISSUE"

        # Shorten name for display
        short_name = name.replace("gpt_neox.", "").replace("transformer.", "")
        if len(short_name) > 48:
            short_name = "..." + short_name[-45:]

        print(
            f"{short_name:<50} {fp32['mean_abs']:>12.2e} {bf16['mean_abs']:>12.2e} "
            f"{ratio:>10.4f} {bf16['pct_zeros']:>11.1f}%{marker}"
        )

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    fp32_means = [s["mean_abs"] for s in stats_fp32.values()]
    bf16_means = [s["mean_abs"] for s in stats_bf16.values()]
    bf16_zeros = [s["pct_zeros"] for s in stats_bf16.values()]

    print(f"\nfp32 gradient magnitude range: {min(fp32_means):.2e} to {max(fp32_means):.2e}")
    print(f"bf16 gradient magnitude range: {min(bf16_means):.2e} to {max(bf16_means):.2e}")
    print(f"bf16 zero percentage range: {min(bf16_zeros):.1f}% to {max(bf16_zeros):.1f}%")

    # Count layers with significant issues
    issue_layers = sum(
        1
        for name in layer_names
        if name in stats_bf16
        and (
            stats_bf16[name]["mean_abs"] / stats_fp32[name]["mean_abs"] < 0.5
            if stats_fp32[name]["mean_abs"] > 0
            else False
        )
    )
    print(f"\nLayers with >50% gradient magnitude loss: {issue_layers}/{len(layer_names)}")

    # Check if early layers are worse
    print("\n" + "-" * 80)
    print("LAYER DEPTH ANALYSIS (are early layers worse?)")
    print("-" * 80)

    # Group by layer index
    layer_ratios = {}
    for name in layer_names:
        if name not in stats_bf16:
            continue
        # Extract layer number
        for part in name.split("."):
            if part.isdigit():
                layer_idx = int(part)
                if layer_idx not in layer_ratios:
                    layer_ratios[layer_idx] = []
                ratio = (
                    stats_bf16[name]["mean_abs"] / stats_fp32[name]["mean_abs"]
                    if stats_fp32[name]["mean_abs"] > 0
                    else 0
                )
                layer_ratios[layer_idx].append(ratio)
                break

    print(f"{'Layer':>8} {'Avg bf16/fp32 ratio':>20} {'Status':>15}")
    for layer_idx in sorted(layer_ratios.keys()):
        avg_ratio = sum(layer_ratios[layer_idx]) / len(layer_ratios[layer_idx])
        status = "OK" if avg_ratio > 0.9 else "DEGRADED" if avg_ratio > 0.5 else "SEVERE"
        print(f"{layer_idx:>8} {avg_ratio:>20.4f} {status:>15}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Investigate vanishing gradients in bf16")
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-14m",
        help="Model name (default: EleutherAI/pythia-14m)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length (default: 128)",
    )
    args = parser.parse_args()

    compare_gradients(args.model, args.seq_len)
    compare_covariances(args.model, args.seq_len)
