#!/usr/bin/env python3
"""Generate minimal EKFAC test data using a small model on CPU.

This script generates ground truth data for EKFAC tests using a tiny model
that can run on CPU, making tests accessible in CI environments.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from ground_truth.collector import GroundTruthCovarianceCollector
from test_utils import set_all_seeds

from bergson.hessians.utils import TensorDict


def log(msg):
    """Print with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def generate_test_data(
    output_dir: str,
    model_name: str = "calum/tinystories-gpt2-3M",
    num_samples: int = 5,
    max_length: int = 32,
):
    """Generate minimal EKFAC test data.

    Args:
        output_dir: Output directory for test data
        model_name: HuggingFace model name
        num_samples: Number of text samples
        max_length: Maximum sequence length
    """
    log("Starting test data generation")
    set_all_seeds(42)
    log("Seeds set")

    log(f"Configuration:")
    log(f"  Model: {model_name}")
    log(f"  Samples: {num_samples}")
    log(f"  Max length: {max_length}")
    log(f"  Output: {output_dir}")

    # Create directories
    log("Creating output directories...")
    os.makedirs(f"{output_dir}/covariances", exist_ok=True)
    os.makedirs(f"{output_dir}/eigenvectors", exist_ok=True)
    os.makedirs(f"{output_dir}/eigenvalue_corrections", exist_ok=True)
    log("Directories created")

    # Load model
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer loaded")

    log("Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    log("Model loaded and set to eval mode")

    # Generate samples
    log("Generating text samples...")
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world, this is a test.",
        "Machine learning is fascinating.",
        "PyTorch is amazing for deep learning.",
        "Testing is crucial for code quality.",
    ] * ((num_samples // 5) + 1)
    texts = texts[:num_samples]
    log(f"Generated {len(texts)} text samples")

    # Tokenize
    log("Tokenizing text samples...")
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    log(f"Tokenized: input_ids shape = {encodings['input_ids'].shape}")

    # Compute covariances
    log("Setting up covariance collector...")
    activation_covariances = {}
    gradient_covariances = {}

    log("Creating GroundTruthCovarianceCollector...")
    collector = GroundTruthCovarianceCollector(
        model=model,
        activation_covariances=activation_covariances,
        gradient_covariances=gradient_covariances,
        target_modules=None,  # Auto-discover all Linear layers
    )
    log(f"Collector created, found {len(collector.target_info)} linear layers")

    total_processed = 0
    log("Starting forward and backward pass with hooks...")
    with collector:
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        log(f"Prepared inputs: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

        # Forward
        log("Running forward pass...")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log("Forward pass complete")

        # Compute loss and backward
        log("Computing loss...")
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            reduction="sum",
        )
        log(f"Loss computed: {loss.item():.4f}")

        log("Running backward pass...")
        loss.backward()
        log("Backward pass complete")

        total_processed = (attention_mask == 1).sum().item()
        log(f"Total tokens processed: {total_processed}")

    # Normalize
    log("Normalizing covariances...")
    for name in activation_covariances:
        activation_covariances[name] /= total_processed
        gradient_covariances[name] /= total_processed
    log(f"Normalized {len(activation_covariances)} covariance pairs")

    # Save covariances
    log("Saving covariances...")
    save_file(
        TensorDict(activation_covariances).cpu(),
        f"{output_dir}/covariances/activation_covariance.safetensors",
    )
    save_file(
        TensorDict(gradient_covariances).cpu(),
        f"{output_dir}/covariances/gradient_covariance.safetensors",
    )
    log("Covariances saved")

    log("Saving stats...")
    with open(f"{output_dir}/covariances/stats.json", "w") as f:
        json.dump({"total_processed_global": total_processed}, f)
    log("Stats saved")

    # Compute eigenvectors
    log("Computing eigenvectors...")
    activation_eigenvectors = {}
    gradient_eigenvectors = {}

    for i, name in enumerate(activation_covariances, 1):
        log(f"  Computing eigenvectors for layer {i}/{len(activation_covariances)}: {name}")
        _, eigenvectors_a = torch.linalg.eigh(activation_covariances[name])
        activation_eigenvectors[name] = eigenvectors_a

        _, eigenvectors_g = torch.linalg.eigh(gradient_covariances[name])
        gradient_eigenvectors[name] = eigenvectors_g
    log("All eigenvectors computed")

    log("Saving eigenvectors...")
    save_file(
        TensorDict(activation_eigenvectors).cpu(),
        f"{output_dir}/eigenvectors/eigenvectors_activations.safetensors",
    )
    save_file(
        TensorDict(gradient_eigenvectors).cpu(),
        f"{output_dir}/eigenvectors/eigenvectors_gradients.safetensors",
    )
    log("Eigenvectors saved")

    # Create dummy eigenvalue corrections (simplified)
    log("Creating eigenvalue corrections...")
    eigenvalue_corrections = {}
    layer_names = list(activation_covariances.keys())
    for name in layer_names:
        dim_g, dim_a = gradient_eigenvectors[name].shape[0], activation_eigenvectors[name].shape[0]
        eigenvalue_corrections[name] = torch.randn(dim_g, dim_a).abs() * 0.1
    log(f"Created eigenvalue corrections for {len(layer_names)} layers")

    log("Saving eigenvalue corrections...")
    save_file(
        TensorDict(eigenvalue_corrections).cpu(),
        f"{output_dir}/eigenvalue_corrections/eigenvalue_corrections.safetensors",
    )
    log("Eigenvalue corrections saved")

    # Save config
    log("Creating config...")
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

    log("Saving config...")
    with open(f"{output_dir}/index_config.json", "w") as f:
        json.dump(config, f, indent=2)
    log("Config saved")

    log(f"\n{'='*60}")
    log("Test data generated successfully!")
    log(f"  Layers: {len(layer_names)}")
    log(f"  Tokens processed: {total_processed}")
    log(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EKFAC test data")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="calum/tinystories-gpt2-3M")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=32)

    args = parser.parse_args()
    generate_test_data(
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )
