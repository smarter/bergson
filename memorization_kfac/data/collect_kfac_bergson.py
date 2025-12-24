#!/usr/bin/env python
"""
K-FAC factor collection using bergson's EKFAC implementation.

This script provides the same CLI interface as collect_kfac_multilayer.py
but uses bergson's covariance collection infrastructure with proper valid_mask handling.

It additionally support multiple GPUs via the --world_size and --fsdp flags.
"""
import argparse
import json
import pathlib
import random

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from bergson.data import IndexConfig
from bergson.distributed import distributed_computing
from bergson.hessians.ekfac_compute import EkfacComputer
from bergson.hessians.compute_all import compute_all_factors


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
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument(
        "--world_size",
        type=int,
        default=None,
        help="Number of GPUs to use. If None, uses all available GPUs when --fsdp is set, otherwise uses 1.",
    )
    p.add_argument(
        "--fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallel (FSDP) for multi-GPU training.",
    )
    return p.parse_args()


def collect_sequences(corpus: str, tokenizer, seq_len: int, nbytes: int) -> list[list[int]]:
    """Collect fixed-length sequences from a streaming dataset."""
    HF_DATASETS = {
        "olmo": "allenai/olmo-mix-1124",
        "dolmo": "allenai/dolmino-mix-1124",
    }

    if corpus == "olmo":
        from datasets import Features, Value

        ds = load_dataset(
            "json",
            data_files={
                "train": f"hf://datasets/{HF_DATASETS[corpus]}/data/**/*.json*"
            },
            split="train",
            streaming=True,
            features=Features({"text": Value("string")}),
        )
    else:
        ds = load_dataset(
            "json",
            data_files={
                "train": f"hf://datasets/{HF_DATASETS[corpus]}/data/**/*.json*"
            },
            split="train",
            streaming=True,
        )

    sequences = []
    buf, seen = [], 0
    for sample in tqdm(ds, desc="Loading sequences"):
        txt = sample["text"].strip()
        if not txt:
            continue
        seen += len(txt.encode())
        buf.extend(tokenizer(txt, add_special_tokens=False).input_ids)

        while len(buf) >= seq_len:
            sequences.append(buf[:seq_len])
            buf = buf[seq_len:]

        if seen >= nbytes:
            break

    return sequences


def kfac_worker(
    model, ds, processor, *, batches, target_modules, cfg
):
    """Worker function for KFAC collection."""
    model.gradient_checkpointing_enable()
    #model.enable_input_require_grads()
    #model.config.use_cache = False

    compute_all_factors(model, data, processor, batches=batches, target_modules=target_modules, cfg=cfg)

def main():
    args = parse()
    args.save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.world_size is not None:
        world_size = args.world_size
    elif args.fsdp:
        world_size = torch.cuda.device_count()
    else:
        world_size = 1

    print(f"Streaming ~{args.nbytes / 1e6:.0f}MB from {args.corpus}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    sequences = collect_sequences(args.corpus, tokenizer, args.seq_len, args.nbytes)
    print(f"Collected {len(sequences)} sequences")

    # length column required by allocate_batches
    hf_dataset = Dataset.from_dict({
        "input_ids": sequences,
        "length": [args.seq_len] * len(sequences),
    })

    precision_map = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}

    idx_config = IndexConfig(
        run_path=str(args.save_dir),
        model=args.model,
        fsdp=args.fsdp,
        precision=precision_map.get(args.dtype, "bf16"),
        token_batch_size=args.batch_size * args.seq_len,
        data=None,
        sample=args.sample_labels,
        world_size=world_size if world_size > 1 else None,
    )

    target_modules = set()
    for blk in args.target_blocks:
        target_modules.add(f"layers.{blk}.mlp.gate_proj")
        target_modules.add(f"layers.{blk}.mlp.up_proj")
        target_modules.add(f"layers.{blk}.mlp.down_proj")

    distributed_computing(
        cfg=idx_config,
        worker_fn=kfac_worker,
        setup_data=False,
        setup_model=True,
        setup_processor=True, # We don't use the processor, but this is needed for `distributed_computing` to call `worker_fn` with the right arguments
        dataset=hf_dataset,
        target_modules=target_modules,
    )

    metadata = {
        "target_modules": sorted(target_modules),
        "blocks": sorted(args.target_blocks),
        "format": "bergson",
        "has_eigendecomposition": True,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "world_size": world_size,
        "fsdp": args.fsdp,
    }
    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump(metadata, indent=2, fp=f)

    print(f"Saved KFAC factors for blocks {args.target_blocks}")


if __name__ == "__main__":
    main()
