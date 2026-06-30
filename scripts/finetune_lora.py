#!/usr/bin/env python3
"""
scripts/finetune_lora.py
─────────────────────────
Standalone CLI for LoRA / QLoRA fine-tuning.
Designed to be run directly (not via the API) for large training jobs.

Usage:
    python scripts/finetune_lora.py \
        --base_model  mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset     tatsu-lab/alpaca \
        --output_dir  ./models/my_adapter \
        --epochs      3 \
        --lora_r      16 \
        --use_4bit

Dataset format expected (JSONL or HuggingFace):
    {"instruction": "...", "input": "...", "output": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tuning script")
    p.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--dataset", required=True, help="HF dataset id or local JSONL path")
    p.add_argument("--output_dir", default="./models/lora_adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--use_4bit", action="store_true", help="QLoRA 4-bit quantisation")
    p.add_argument("--hf_token", default=os.getenv("HF_TOKEN", ""))
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Which linear layers to apply LoRA to",
    )
    return p.parse_args()


def format_alpaca(example: dict) -> str:
    """Convert Alpaca-style example to instruction-tuning prompt."""
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


def load_dataset(dataset_arg: str, hf_token: str):
    from datasets import Dataset, load_dataset as hf_load

    if dataset_arg.endswith(".jsonl") or dataset_arg.endswith(".json"):
        with open(dataset_arg) as f:
            data = [json.loads(line) for line in f if line.strip()]
        return Dataset.from_list(data)

    return hf_load(dataset_arg, split="train", token=hf_token or None)


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"\n{'='*60}")
    print(f"  LoRA Fine-Tuning")
    print(f"  Base model : {args.base_model}")
    print(f"  Dataset    : {args.dataset}")
    print(f"  Output     : {args.output_dir}")
    print(f"  4-bit QLoRA: {args.use_4bit}")
    print(f"{'='*60}\n")

    # ── Quantisation ────────────────────────────────────────────────────────
    bnb_config = None
    if args.use_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print("✓ 4-bit QLoRA quantisation enabled")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        token=args.hf_token or None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model ────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        token=args.hf_token or None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"✓ LoRA applied — trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.hf_token)

    # Apply Alpaca formatting if columns present
    if "instruction" in dataset.column_names:
        dataset = dataset.map(
            lambda ex: {"text": format_alpaca(ex)},
            remove_columns=dataset.column_names,
        )
        print(f"✓ Applied Alpaca formatting to {len(dataset):,} examples")
    else:
        # Assume 'text' column already present
        assert "text" in dataset.column_names, (
            "Dataset must have 'text' or 'instruction'/'output' columns"
        )

    # ── Training ─────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit" if args.use_4bit else "adamw_torch",
        learning_rate=args.lr,
        weight_decay=0.001,
        fp16=torch.cuda.is_available() and not args.use_4bit,
        bf16=args.use_4bit and torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=args.max_seq_len,
    )

    print("\n🚀 Starting training...\n")
    result = trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metadata
    meta = {
        "base_model": args.base_model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_loss": round(result.training_loss, 4),
        "train_runtime_s": round(result.metrics.get("train_runtime", 0)),
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(Path(args.output_dir) / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  Loss      : {meta['train_loss']}")
    print(f"  Runtime   : {meta['train_runtime_s']}s")
    print(f"  Adapter   : {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
