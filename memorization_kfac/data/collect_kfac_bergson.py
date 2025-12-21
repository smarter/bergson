#!/usr/bin/env python
"""
K-FAC factor collection using bergson's EKFAC implementation.

This script provides the same CLI interface as collect_kfac_multilayer.py
but uses bergson's covariance collection infrastructure with proper valid_mask handling.
"""
import argparse
import json
import os
import pathlib
import random
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from bergson.data import IndexConfig, DataConfig
from bergson.hessians.ekfac_compute import EkfacComputer
from bergson.hessians.collector import HookCollectorBase
from bergson.hessians.sharded_computation import ShardedMul
import torch.distributed as dist
from dataclasses import dataclass
from torch import Tensor


def parse():
    """Parse command-line arguments (compatible with collect_kfac_multilayer.py)."""
    p = argparse.ArgumentParser(description="Collect K-FAC factors using bergson")
    p.add_argument("--model", default="allenai/OLMo-2-1124-7B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])

    p.add_argument("--corpus", choices=["olmo", "dolmo"], default="dolmo")
    p.add_argument("--nbytes", type=int, default=100_000_000)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument(
        "--layers_per_pass",
        type=int,
        default=1,
        help="Process this many target layers before flushing.",
    )
    p.add_argument(
        "--target_blocks",
        type=int,
        nargs="+",
        default=list(range(16)),
        help="0-based encoder block ids",
    )
    p.add_argument("--save_dir", type=pathlib.Path, default="kfac_out")
    p.add_argument(
        "--sample_labels",
        action="store_true",
        help="If set, use multinomial-sampled labels",
    )
    p.add_argument(
        "--kfac-only",
        action="store_true",
        dest="kfac_only",
        help="Use KFAC (covariances only) instead of EKFAC (with eigenvalue correction). "
        "When enabled, only CovarianceCollector is used without LambdaCollector.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


class StreamingDataset:
    """Simple streaming dataset wrapper for bergson."""

    def __init__(self, corpus: str, tokenizer, seq_len: int, nbytes: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.budget = nbytes

        HF_DATASETS = {
            "olmo": "allenai/olmo-mix-1124",
            "dolmo": "allenai/dolmino-mix-1124",
        }

        # Load streaming dataset
        if corpus == "olmo":
            from datasets import Features, Value

            self.ds = load_dataset(
                "json",
                data_files={
                    "train": f"hf://datasets/{HF_DATASETS[corpus]}/data/**/*.json*"
                },
                split="train",
                streaming=True,
                features=Features({"text": Value("string")}),
            )
        else:  # dolmo
            self.ds = load_dataset(
                "json",
                data_files={
                    "train": f"hf://datasets/{HF_DATASETS[corpus]}/data/**/*.json*"
                },
                split="train",
                streaming=True,
            )

    def __iter__(self):
        """Yield sequences of token IDs."""
        buf, seen = [], 0
        for sample in self.ds:
            txt = sample["text"].strip()
            if not txt:
                continue
            seen += len(txt.encode())
            buf.extend(self.tok(txt, add_special_tokens=False).input_ids)

            while len(buf) >= self.seq_len:
                yield buf[: self.seq_len]
                buf = buf[self.seq_len :]

            if seen >= self.budget:
                return


class SimpleDataset:
    """Simple dataset wrapper for EkfacComputer that holds pre-tokenized sequences."""

    def __init__(self, sequences: list[list[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if isinstance(idx, (list, slice)):
            # Handle batch indexing - return a single dict with batched data
            if isinstance(idx, slice):
                indices = range(*idx.indices(len(self)))
            else:
                indices = idx
            # Return single dict with list of sequences
            return {"input_ids": [self.sequences[i] for i in indices]}
        else:
            # Single item - return dict with single sequence
            return {"input_ids": self.sequences[idx]}


def main():
    args = parse()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Fix device string
    if args.device.startswith("cuda") and ":" not in args.device:
        args.device = "cuda:0"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map={"": args.device},
        trust_remote_code=True,
    )

    # Enable gradient checkpointing to reduce memory usage during forward pass
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()

    # Disable dropout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    # Create save directory
    args.save_dir.mkdir(parents=True, exist_ok=True)

    method = "KFAC" if args.kfac_only else "EKFAC"
    print(f"Using {method} method (kfac_only={args.kfac_only})")
    print(f"Processing blocks: {args.target_blocks}")
    print(f"Streaming ~{args.nbytes / 1e6:.0f}MB from {args.corpus}")

    # Collect sequences from streaming dataset
    print("Collecting sequences from streaming dataset...")
    streaming_ds = StreamingDataset(args.corpus, tokenizer, args.seq_len, args.nbytes)
    sequences = list(tqdm(streaming_ds, desc="Loading sequences"))
    print(f"Collected {len(sequences)} sequences")

    # Create dataset wrapper
    dataset = SimpleDataset(sequences)

    # Create batches as lists of indices
    batches = []
    for i in range(0, len(sequences), args.batch_size):
        batches.append(list(range(i, min(i + args.batch_size, len(sequences)))))

    # Prepare target modules (gate, up, down projections for specified blocks)
    # Note: EkfacComputer uses model.base_model internally, so names should not include "model." prefix
    target_modules = set()
    for blk in args.target_blocks:
        target_modules.add(f"layers.{blk}.mlp.gate_proj")
        target_modules.add(f"layers.{blk}.mlp.up_proj")
        target_modules.add(f"layers.{blk}.mlp.down_proj")

    # Create IndexConfig
    idx_config = IndexConfig(
        run_path=args.save_dir,
        data=None,
        sample=args.sample_labels,
    )

    # Use EkfacComputer directly - it handles all the logic!
    print(f"Computing {method} using EkfacComputer...")
    ekfac = EkfacComputer(
        model=model,
        data=dataset,
        batches=batches,
        target_modules=target_modules,
        cfg=idx_config,
    )

    # Compute covariances (handles valid_mask, total_processed, etc.)
    # total_processed is saved to disk at influence_results/total_processed_covariances.pt
    ekfac.compute_covariance()

    # Save metadata with batch information for conversion to original format
    metadata = {
        "blocks": args.target_blocks,
        "format": "bergson",
        "kfac_only": args.kfac_only,
        "method": method,
        "num_batches": len(batches),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
    }

    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"✓ Saved {method} factors for blocks {args.target_blocks}")
    print("✓ Collection complete using bergson")


if __name__ == "__main__":
    main()
