#!/usr/bin/env python
"""
K-FAC factor collection using bergson's EKFAC implementation.

This script provides the same CLI interface as collect_kfac_multilayer.py
but uses bergson's covariance / eigendecomposition / eigenvalue-correction
pipeline with proper valid_mask handling.

It additionally supports multiple GPUs via the --world_size and --fsdp flags.
"""
import argparse
import json
import os
import pathlib
import random
import shutil
from datetime import timedelta

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from bergson.config import DistributedConfig, HessianConfig, IndexConfig
from bergson.data import allocate_batches
from bergson.distributed import launch_distributed_run
from bergson.hessians.eigenvectors import compute_eigendecomposition
from bergson.hessians.hessian_approximations import collect_hessians
from bergson.utils.worker_utils import setup_model_and_peft


LEGACY_RENAMES = {
    "activation_sharded": "activation_covariance_sharded",
    "gradient_sharded": "gradient_covariance_sharded",
    "eigen_activation_sharded": "activation_eigen_sharded",
    "eigen_gradient_sharded": "gradient_eigen_sharded",
}


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
    rank: int,
    local_rank: int,
    world_size: int,
    index_cfg: IndexConfig,
    hessian_cfg: HessianConfig,
    ds: Dataset,
    target_modules: set,
):
    """Mirror of ``bergson.hessians.hessian_approximations.hessian_worker`` that
    uses a pre-loaded dataset and caller-supplied ``target_modules``.
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if world_size > 1:
        addr = os.environ.get("MASTER_ADDR", "localhost")
        port = os.environ.get("MASTER_PORT", "29500")

        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{addr}:{port}",
            device_id=torch.device(f"cuda:{local_rank}"),
            rank=rank,
            timeout=timedelta(hours=1),
            world_size=world_size,
        )

    model, _ = setup_model_and_peft(index_cfg)

    partial_dir = pathlib.Path(str(index_cfg.partial_run_path))
    total_processed_path = partial_dir / "total_processed.pt"
    cov_total_path = partial_dir / "total_processed_covariances.pt"
    lambda_total_path = partial_dir / "total_processed_lambda_correction.pt"

    kwargs = {
        "model": model,
        "data": ds,
        "index_cfg": index_cfg,
        "hessian_cfg": hessian_cfg,
        "target_modules": target_modules,
        "attention_cfgs": {},
    }

    batches = allocate_batches(ds["length"][:], index_cfg.token_batch_size)
    kwargs["batches"] = batches

    # Covariance pass.
    collect_hessians(**kwargs)

    dist.barrier() if dist.is_initialized() else None

    if rank == 0 and total_processed_path.exists():
        shutil.copyfile(total_processed_path, cov_total_path)

    total_processed = torch.load(
        total_processed_path, map_location="cpu", weights_only=False
    )

    compute_eigendecomposition(
        str(partial_dir / "activation_sharded"), total_processed=total_processed
    )
    compute_eigendecomposition(
        str(partial_dir / "gradient_sharded"), total_processed=total_processed
    )

    dist.barrier() if dist.is_initialized() else None

    if hessian_cfg.ev_correction:
        collect_hessians(**kwargs, ev_correction=True)
        dist.barrier() if dist.is_initialized() else None
        if rank == 0 and total_processed_path.exists():
            shutil.move(str(total_processed_path), str(lambda_total_path))


def _rename_legacy_paths(root: pathlib.Path) -> None:
    """Rename bergson's upstream directory names to the legacy ones expected
    by the rest of memorization_kfac/ and the companion notebooks."""
    for src_name, dst_name in LEGACY_RENAMES.items():
        src = root / src_name
        dst = root / dst_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            src.rename(dst)


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

    hf_dataset = Dataset.from_dict({
        "input_ids": sequences,
        "length": [args.seq_len] * len(sequences),
    })

    precision_map = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}

    run_path = args.save_dir / "influence_results"

    dist_cfg = DistributedConfig(
        nproc_per_node=world_size, nnode=1, node_rank=0
    )

    index_cfg = IndexConfig(
        run_path=str(run_path),
        model=args.model,
        fsdp=args.fsdp,
        precision=precision_map.get(args.dtype, "bf16"),
        token_batch_size=args.batch_size * args.seq_len,
        distributed=dist_cfg,
        overwrite=True,
    )

    hessian_cfg = HessianConfig(
        method="kfac",
        ev_correction=True,
        # fp32 accumulators keep the bf16 outer-product path numerically stable.
        hessian_dtype="fp32",
        use_dataset_labels=not args.sample_labels,
    )

    target_modules = set()
    # Activation checkpointing wrappers (FSDP + gradient checkpointing) rename modules.
    blk_suffix = "._checkpoint_wrapped_module" if args.fsdp else ""
    for blk in args.target_blocks:
        target_modules.add(f"layers.{blk}{blk_suffix}.mlp.gate_proj")
        target_modules.add(f"layers.{blk}{blk_suffix}.mlp.up_proj")
        target_modules.add(f"layers.{blk}{blk_suffix}.mlp.down_proj")

    partial = pathlib.Path(str(run_path) + ".part")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)

    launch_distributed_run(
        "kfac",
        kfac_worker,
        [index_cfg, hessian_cfg, hf_dataset, target_modules],
        dist_cfg,
    )

    if partial.exists():
        if run_path.exists():
            shutil.rmtree(run_path)
        shutil.move(str(partial), str(run_path))

    _rename_legacy_paths(run_path)

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
