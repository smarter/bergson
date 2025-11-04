#!/usr/bin/env python3
"""Generate minimal test data for EKFAC tests using a tiny model on CPU.

This script generates a small test dataset that can be run in CI without GPU.
It uses a very small model and minimal data to create ground truth for testing.
"""

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ground_truth.collector import GroundTruthCovarianceCollector
from test_utils import set_all_seeds

from bergson.hessians.utils import TensorDict


def generate_minimal_test_data(
    output_dir: str = "tests/ekfac_tests/fixtures/minimal",
    model_name: str = "sdobson/nanochat",
    num_samples: int = 10,
    max_length: int = 32,
):
    """Generate minimal test data for CPU-based testing.

    Args:
        output_dir: Directory to save test data
        model_name: HuggingFace model name (should be very small)
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
    """
    set_all_seeds(42)

    print(f"Generating minimal test data using {model_name}...")
    print(f"Output directory: {output_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/covariances", exist_ok=True)
    os.makedirs(f"{output_dir}/eigenvectors", exist_ok=True)
    os.makedirs(f"{output_dir}/eigenvalue_corrections", exist_ok=True)

    # Load tiny model and tokenizer on CPU
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    # Generate simple text samples
    print("Generating text samples...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test.",
        "Machine learning is fascinating.",
        "PyTorch is a deep learning framework.",
        "Testing is important for software quality.",
    ] * (num_samples // 5 + 1)
    texts = texts[:num_samples]

    # Tokenize
    print("Tokenizing...")
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Setup collector for covariances
    print("Computing covariances...")
    activation_covariances = {}
    gradient_covariances = {}

    collector = GroundTruthCovarianceCollector(
        model=model,
        layer_names=[],  # Will be auto-detected
        activation_covariances=activation_covariances,
        gradient_covariances=gradient_covariances,
    )

    # Identify linear layers to hook
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
            layer_names.append(name)

    print(f"Found {len(layer_names)} linear layers: {layer_names}")
    collector.layer_names = layer_names

    # Compute covariances
    total_processed = 0
    with collector:
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Compute loss and backward
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding

        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="sum",
        )

        loss.backward()

        total_processed = (attention_mask == 1).sum().item()

    print(f"Processed {total_processed} tokens")

    # Normalize covariances
    for name in activation_covariances:
        activation_covariances[name] /= total_processed
        gradient_covariances[name] /= total_processed

    # Save covariances
    print("Saving covariances...")
    save_file(
        TensorDict(activation_covariances).cpu(),
        f"{output_dir}/covariances/activation_covariance.safetensors",
    )
    save_file(
        TensorDict(gradient_covariances).cpu(),
        f"{output_dir}/covariances/gradient_covariance.safetensors",
    )

    # Save stats
    with open(f"{output_dir}/covariances/stats.json", "w") as f:
        json.dump({"total_processed_global": total_processed}, f)

    # Compute eigenvectors
    print("Computing eigenvectors...")
    activation_eigenvectors = {}
    gradient_eigenvectors = {}

    for name in activation_covariances:
        # Compute eigendecomposition for activations
        cov_a = activation_covariances[name]
        eigenvalues_a, eigenvectors_a = torch.linalg.eigh(cov_a)
        activation_eigenvectors[name] = eigenvectors_a

        # Compute eigendecomposition for gradients
        cov_g = gradient_covariances[name]
        eigenvalues_g, eigenvectors_g = torch.linalg.eigh(cov_g)
        gradient_eigenvectors[name] = eigenvectors_g

    # Save eigenvectors
    print("Saving eigenvectors...")
    save_file(
        TensorDict(activation_eigenvectors).cpu(),
        f"{output_dir}/eigenvectors/eigenvectors_activations.safetensors",
    )
    save_file(
        TensorDict(gradient_eigenvectors).cpu(),
        f"{output_dir}/eigenvectors/eigenvectors_gradients.safetensors",
    )

    # Compute eigenvalue corrections (simplified - just diagonal approximation)
    print("Computing eigenvalue corrections...")
    eigenvalue_corrections = {}

    for name in layer_names:
        dim_a = activation_eigenvectors[name].shape[0]
        dim_g = gradient_eigenvectors[name].shape[0]
        # Initialize with small random values
        eigenvalue_corrections[name] = torch.randn(dim_g, dim_a).abs() * 0.1

    # Save eigenvalue corrections
    print("Saving eigenvalue corrections...")
    save_file(
        TensorDict(eigenvalue_corrections).cpu(),
        f"{output_dir}/eigenvalue_corrections/eigenvalue_corrections.safetensors",
    )

    # Create minimal index config
    print("Creating index config...")
    config = {
        "model_name": model_name,
        "data": {
            "dataset_name": "minimal_test",
            "data_dir": None,
            "split": "train",
            "tokenizer_name": model_name,
        },
        "run_path": None,
        "debug": True,
        "fsdp": False,
        "world_size": 1,
        "precision": "float32",
        "ekfac": True,
    }

    with open(f"{output_dir}/index_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTest data generated successfully in {output_dir}")
    print(f"- {len(layer_names)} layers")
    print(f"- {num_samples} samples")
    print(f"- {total_processed} tokens processed")
    print("\nFiles created:")
    print("  - covariances/activation_covariance.safetensors")
    print("  - covariances/gradient_covariance.safetensors")
    print("  - covariances/stats.json")
    print("  - eigenvectors/eigenvectors_activations.safetensors")
    print("  - eigenvectors/eigenvectors_gradients.safetensors")
    print("  - eigenvalue_corrections/eigenvalue_corrections.safetensors")
    print("  - index_config.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate minimal test data for EKFAC tests")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tests/ekfac_tests/fixtures/minimal",
        help="Output directory for test data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sdobson/nanochat",
        help="HuggingFace model name (should be very small)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    generate_minimal_test_data(
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )
