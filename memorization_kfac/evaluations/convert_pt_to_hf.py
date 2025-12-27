#!/usr/bin/env python
"""Convert legacy .pt model files (state_dict) to HuggingFace format.

Previously, eval_mem_kfac.py saved edited models as:
    torch.save({"model_state_dict": model.state_dict()}, model_path)

This script converts those to HuggingFace format so they can be used with olmes benchmarks.

Usage:
    python convert_pt_to_hf.py --pt-file /path/to/model.pt --base-model allenai/OLMo-2-1124-7B --output-dir /path/to/output

    # Or convert all .pt files in a directory:
    python convert_pt_to_hf.py --pt-dir /path/to/models --base-model allenai/OLMo-2-1124-7B
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_pt_to_hf(pt_path: Path, base_model: str, output_dir: Path, dtype: str = "bfloat16") -> None:
    """Convert a single .pt file to HuggingFace format.

    Args:
        pt_path: Path to the .pt file containing {"model_state_dict": ...}
        base_model: HuggingFace model ID for the base model architecture
        output_dir: Directory to save the converted model
        dtype: Model dtype (bfloat16 or float32)
    """
    print(f"Loading base model: {base_model}")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",  # Load on CPU for conversion
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading state dict from: {pt_path}")
    checkpoint = torch.load(pt_path, map_location="cpu")

    # Handle both formats: {"model_state_dict": ...} or direct state_dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict into model
    model.load_state_dict(state_dict)

    # Save in HuggingFace format
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to HuggingFace format: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Done! Model saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert legacy .pt model files to HuggingFace format")

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--pt-file", type=str, help="Path to a single .pt file to convert")
    input_group.add_argument("--pt-dir", type=str, help="Directory containing .pt files to convert")

    # Required args
    parser.add_argument("--base-model", type=str, required=True,
                        help="HuggingFace model ID for the base model (e.g., allenai/OLMo-2-1124-7B)")

    # Optional args
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same location as .pt file, without .pt extension)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"],
                        help="Model dtype (default: bfloat16)")

    args = parser.parse_args()

    if args.pt_file:
        # Convert single file
        pt_path = Path(args.pt_file)
        if not pt_path.exists():
            raise FileNotFoundError(f"File not found: {pt_path}")

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default: same location, remove .pt extension
            output_dir = pt_path.parent / pt_path.stem

        convert_pt_to_hf(pt_path, args.base_model, output_dir, args.dtype)

    else:
        # Convert all .pt files in directory
        pt_dir = Path(args.pt_dir)
        if not pt_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pt_dir}")

        pt_files = list(pt_dir.rglob("*.pt"))
        if not pt_files:
            print(f"No .pt files found in {pt_dir}")
            return

        print(f"Found {len(pt_files)} .pt files to convert")

        for pt_path in pt_files:
            # Output dir: same location, remove .pt extension
            output_dir = pt_path.parent / pt_path.stem

            if output_dir.exists() and (output_dir / "config.json").exists():
                print(f"Skipping {pt_path} - already converted to {output_dir}")
                continue

            try:
                convert_pt_to_hf(pt_path, args.base_model, output_dir, args.dtype)
            except Exception as e:
                print(f"Error converting {pt_path}: {e}")
                continue


if __name__ == "__main__":
    main()
