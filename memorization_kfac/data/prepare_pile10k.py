#!/usr/bin/env python
"""
Download and prepare pile10k dataset from HuggingFace.

Downloads NeelNanda/pile-10k and converts it to a plain text file
for use in NDCG evaluation.
"""
import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and prepare pile10k dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "pile10k_None.txt",
        help="Output text file path (default: pile10k_None.txt in data directory)",
    )
    args = parser.parse_args()

    print(f"Downloading NeelNanda/pile-10k from HuggingFace...")
    dataset = load_dataset("NeelNanda/pile-10k", split="train")

    print(f"Writing {len(dataset)} documents to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        for example in dataset:
            # Write each document on a single line (newlines replaced with spaces)
            text = example["text"].replace("\n", " ").replace("\r", " ")
            f.write(text + "\n")

    print(f"✓ Successfully created {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
