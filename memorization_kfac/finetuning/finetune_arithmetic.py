#!/usr/bin/env python
"""
Fine-tune a model on arithmetic data using unsloth and LoRA.

This script fine-tunes an edited model (or base model) on arithmetic
expressions to recover math abilities lost during memorization removal.
"""
import unsloth  # noqa: F401 - must be imported first for patching
import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


def load_jsonl_dataset(path: Path) -> tuple[Dataset, str]:
    """Load a JSONL dataset, auto-detecting format (text or messages).

    Returns:
        Tuple of (dataset, format) where format is "text" or "messages".
    """
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    # Detect format from first record
    if not records:
        raise ValueError(f"Empty dataset: {path}")

    if "messages" in records[0]:
        return Dataset.from_dict({"messages": [r["messages"] for r in records]}), "messages"
    else:
        return Dataset.from_dict({"text": [r["text"] for r in records]}), "text"


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune model on arithmetic data using unsloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument("--model-path", type=str, required=True, help="Path to model (edited or base)")
    parser.add_argument("--data-path", type=Path, required=True, help="Path to arithmetic JSONL")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")

    # LoRA configuration
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")

    # Training configuration
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Warmup steps")

    # Precision and quantization
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"], help="Data type"
    )
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization")

    # Output options
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA weights into base model")

    args = parser.parse_args()

    # Map dtype string to actual dtype
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Fix tokenizer to prevent EOS token issues during training.
    # If pad_token == eos_token, the model learns to ignore EOS, causing repetition.
    # See: https://github.com/huggingface/trl/issues/1283
    tokenizer.padding_side = "right"
    if tokenizer.pad_token == tokenizer.eos_token or tokenizer.pad_token is None:
        # Add a separate pad token to avoid masking EOS during training
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        print(f"  Added separate pad token: {tokenizer.pad_token}")

    # Set chat template if not present (needed for base models like OLMo-2)
    # Template from: https://huggingface.co/allenai/OLMo-2-1124-7B-SFT
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ '<|system|>\\n' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'user' %}{{ '<|user|>\\n' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{% if not loop.last %}{{ '<|assistant|>\\n' + message['content'] + eos_token + '\\n' }}"
            "{% else %}{{ '<|assistant|>\\n' + message['content'] + eos_token }}{% endif %}"
            "{% endif %}"
            "{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
            "{% endfor %}"
        )
        print("  Set OLMo-2 chat template")

    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Loading dataset from {args.data_path}...")
    dataset, data_format = load_jsonl_dataset(args.data_path)
    print(f"  Loaded {len(dataset)} examples (format: {data_format})")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training config using SFTConfig (newer trl API)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=args.dtype == "float16",
        bf16=args.dtype == "bfloat16",
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        seed=42,
        # SFT-specific settings
        max_seq_length=args.max_seq_length,
        dataset_text_field="text" if data_format == "text" else None,
        packing=False,
    )

    # For chat format, create a formatting function that applies the chat template
    formatting_func = None
    if data_format == "messages":

        def formatting_func(examples):
            # formatting_func receives batched examples and must return a list of strings
            return [
                tokenizer.apply_chat_template(msgs, tokenize=False)
                for msgs in examples["messages"]
            ]

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    # Save model
    if args.merge_lora:
        print("Merging LoRA weights and saving...")
        model.save_pretrained_merged(
            str(args.output_dir / "merged"),
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        print("Saving LoRA adapter...")
        model.save_pretrained(str(args.output_dir / "adapter"))
        tokenizer.save_pretrained(str(args.output_dir / "adapter"))

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
