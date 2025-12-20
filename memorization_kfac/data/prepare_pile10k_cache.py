#!/usr/bin/env python
"""
Generate pre-tokenized cache for pile10k perplexity evaluation.

Creates a .pt file with tokenized pile10k data in blocks of fixed size.
This is used for BSN-style perplexity evaluation in eval_mem_kfac.py.
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate pile10k pt_cache")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent / "pile10k_None.txt",
        help="Input pile10k text file (default: pile10k_None.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "evaluations" / "bsn_dependencies" / "data" / "olmo2_clean_pt_cache_112.pt",
        help="Output pt cache file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="allenai/OLMo-2-0425-1B",
        help="Model name for tokenizer (default: OLMo-2 1B)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=112,
        help="Block size for tokenized chunks (default: 112)",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Reading text from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Tokenizing...")
    token_ids = tokenizer(text, add_special_tokens=False).input_ids

    print(f"Total tokens: {len(token_ids):,}")

    # Reshape into blocks of block_size
    n_blocks = len(token_ids) // args.block_size
    truncated_length = n_blocks * args.block_size
    token_ids = token_ids[:truncated_length]

    tokenized_tensor = torch.tensor(token_ids, dtype=torch.long).view(-1, args.block_size)

    print(f"Created tensor of shape {tokenized_tensor.shape}")

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {args.output}...")
    torch.save(tokenized_tensor, args.output)

    print(f"✓ Successfully created {args.output} ({args.output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
