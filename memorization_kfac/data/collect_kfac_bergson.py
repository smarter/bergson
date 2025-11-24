#!/usr/bin/env python
"""
K-FAC factor collection using bergson's EKFAC implementation.

This script provides the same CLI interface as collect_kfac_multilayer.py
but uses bergson's covariance collection infrastructure.
"""
import argparse
import json
import os
import pathlib
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from bergson.data import IndexConfig, DataConfig
from bergson.hessians.collector import CovarianceCollector, HookCollectorBase
from bergson.hessians.sharded_computation import ShardedMul


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


class BergsonCovarianceCollector:
    """Collect K-FAC factors using bergson's infrastructure."""

    def __init__(
        self,
        model,
        tokenizer,
        target_blocks: list[int],
        save_dir: pathlib.Path,
        dtype: torch.dtype,
        kfac_only: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_blocks = target_blocks
        self.save_dir = save_dir
        self.dtype = dtype
        self.kfac_only = kfac_only
        self.rank = rank
        self.world_size = world_size

        # Discover target modules (gate, up, down projections for specified blocks)
        # Note: model.base_model uses names without 'model.' prefix
        self.target_modules = set()
        for blk in target_blocks:
            self.target_modules.add(f"layers.{blk}.mlp.gate_proj")
            self.target_modules.add(f"layers.{blk}.mlp.up_proj")
            self.target_modules.add(f"layers.{blk}.mlp.down_proj")

        # Get target info using bergson's discovery
        self.target_info = HookCollectorBase.discover_targets(
            model.base_model, self.target_modules
        )

        # Initialize sharded computation helper
        self.shard_computer = ShardedMul(
            target_info=self.target_info, lambda_damp_factor=1e-5
        )

        # Create collector
        self.collector = CovarianceCollector(
            model.base_model,
            target_modules=self.target_modules,
            dtype=dtype,
            shard_computer=self.shard_computer,
            rank=rank,
            path=str(save_dir),
            slice_for_lm=True,  # Slice [:, :-1] to match original KFAC implementation
        )

        self.total_tokens = 0

    def collect(self, data_loader, sample_labels: bool = False):
        """Collect covariances from data."""
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        with self.collector:
            for batch_idx, input_ids in enumerate(
                tqdm(data_loader, desc=f"Collecting covariances")
            ):
                # Prepare batch
                x = torch.tensor(input_ids, device=self.model.device).unsqueeze(0)
                mask = x != self.tokenizer.pad_token_id

                # Prepare labels (shift by 1)
                labels = x.clone()
                labels[:, :-1] = x[:, 1:]
                labels[:, -1] = -100
                labels[mask == 0] = -100

                # Forward pass
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x, attention_mask=mask).logits[:, :-1].float()

                # Compute loss
                if sample_labels:
                    # Multinomial sampling
                    with torch.no_grad():
                        y = torch.multinomial(
                            torch.softmax(logits, dim=-1).reshape(-1, logits.size(-1)), 1
                        ).squeeze(1)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), y
                    )
                else:
                    # Gold labels
                    loss = ce_loss(
                        logits.reshape(-1, logits.size(-1)), labels[:, :-1].reshape(-1)
                    )

                # Backward pass - this triggers the hooks
                loss.backward()

                self.total_tokens += mask[:, 1:].sum().item()

    def save_factors(self):
        """Save collected factors in bergson format."""
        # The factors are already saved by the collector in bergson format
        # Save metadata about the collection
        metadata = {
            "blocks": self.target_blocks,
            "n_tokens": self.total_tokens,
            "format": "bergson",
            "kfac_only": self.kfac_only,
            "method": "KFAC" if self.kfac_only else "EKFAC",
        }

        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, indent=2, fp=f)

        method_str = "KFAC" if self.kfac_only else "EKFAC"
        print(
            f"✓ Saved {method_str} factors for blocks {self.target_blocks} ({self.total_tokens:,} tokens)"
        )


def main():
    args = parse()

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

    # Enable gradient checkpointing
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

    # Process target blocks
    method = "KFAC" if args.kfac_only else "EKFAC"
    print(f"Using {method} method (kfac_only={args.kfac_only})")
    print(f"Processing blocks: {args.target_blocks}")
    print(f"Streaming ~{args.nbytes / 1e6:.0f}MB from {args.corpus}")

    # Create streaming dataset
    streaming_ds = StreamingDataset(args.corpus, tokenizer, args.seq_len, args.nbytes)

    # Create collector
    collector = BergsonCovarianceCollector(
        model=model,
        tokenizer=tokenizer,
        target_blocks=args.target_blocks,
        save_dir=args.save_dir,
        dtype=model.dtype,
        kfac_only=args.kfac_only,
    )

    # Collect covariances
    collector.collect(streaming_ds, sample_labels=args.sample_labels)

    # Save factors
    collector.save_factors()

    print("✓ Collection complete using bergson")


if __name__ == "__main__":
    main()
