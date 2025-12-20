#!/usr/bin/env python
"""
Convert bergson KFAC format to original memorization_kfac format.

Handles:
- Key name mapping: layers.N.mlp.gate_proj → blkN.gate
- Normalization: raw sums → normalized by n_tokens
- Format conversion: safetensors → .pt
"""
import argparse
import json
import pathlib
from safetensors import safe_open
import torch


def parse():
    p = argparse.ArgumentParser(description="Convert bergson KFAC to original format")
    p.add_argument(
        "--bergson_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing bergson output (activation_covariance_sharded/, gradient_covariance_sharded/, metadata.json)",
    )
    p.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Directory to save converted .pt files",
    )
    return p.parse_args()


def load_bergson_covariances(bergson_dir: pathlib.Path):
    """Load covariances from bergson safetensors format."""
    # Load activation covariances (A matrices)
    A_dict = {}
    act_cov_dir = bergson_dir / "activation_covariance_sharded"
    if act_cov_dir.exists():
        for shard_file in sorted(act_cov_dir.glob("shard_*.safetensors")):
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    A_dict[key] = f.get_tensor(key)

    # Load gradient covariances (G matrices)
    G_dict = {}
    grad_cov_dir = bergson_dir / "gradient_covariance_sharded"
    if grad_cov_dir.exists():
        for shard_file in sorted(grad_cov_dir.glob("shard_*.safetensors")):
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    G_dict[key] = f.get_tensor(key)

    # Load metadata
    metadata_path = bergson_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"metadata.json not found in {bergson_dir}")

    return A_dict, G_dict, metadata


def convert_key_name(bergson_key: str) -> str:
    """
    Convert bergson key to original format.

    Example:
        layers.14.mlp.gate_proj → blk14.gate
        layers.14.mlp.up_proj → blk14.up
        layers.14.mlp.down_proj → blk14.down
    """
    # Parse: layers.N.mlp.X_proj
    parts = bergson_key.split(".")
    if len(parts) != 4 or parts[0] != "layers" or parts[2] != "mlp":
        raise ValueError(f"Unexpected key format: {bergson_key}")

    layer_num = parts[1]
    proj_name = parts[3].replace("_proj", "")  # gate_proj → gate

    return f"blk{layer_num}.{proj_name}"


def convert_bergson_to_original(bergson_dir: pathlib.Path, output_dir: pathlib.Path):
    """Convert bergson output to original format."""
    print(f"Loading bergson output from {bergson_dir}")
    A_dict, G_dict, metadata = load_bergson_covariances(bergson_dir)

    # Read total_processed from the .pt file (saved by EkfacComputer._collector)
    total_processed_path = bergson_dir / "total_processed_covariances.pt"
    total_processed = torch.load(total_processed_path, map_location="cpu").item()

    blocks = metadata["blocks"]

    print(f"Found {len(A_dict)} activation covariances")
    print(f"Found {len(G_dict)} gradient covariances")
    print(f"Total valid positions: {total_processed:,}")
    print(f"Blocks: {blocks}")

    # Group by block
    block_data = {}
    for bergson_key in A_dict.keys():
        original_key = convert_key_name(bergson_key)

        # Extract block number
        block_num = int(original_key.split(".")[0].replace("blk", ""))

        if block_num not in block_data:
            block_data[block_num] = {}

        # Normalize matrices by total_processed (valid positions only)
        A_normalized = A_dict[bergson_key].float() / total_processed
        G_normalized = G_dict[bergson_key].float() / total_processed

        block_data[block_num][original_key] = {
            "A": A_normalized,
            "G": G_normalized,
            "n_tokens": total_processed,  # Store as n_tokens for compatibility
        }

    # Save one file per block (matching original format)
    output_dir.mkdir(parents=True, exist_ok=True)

    for block_num, data in sorted(block_data.items()):
        output_file = output_dir / f"kfac_factors_blk_{block_num}.pt"
        torch.save(data, output_file)
        print(f"✓ Saved {output_file} ({len(data)} projections)")

    print(f"\n✓ Conversion complete! Output saved to {output_dir}")


def main():
    args = parse()
    convert_bergson_to_original(args.bergson_dir, args.output_dir)


if __name__ == "__main__":
    main()
