#!/usr/bin/env python3
"""Minimum viable SFT fine-tuning on tool-use trajectories.

Fine-tunes Qwen3-8B with LoRA on verified tool-use trajectories
to teach the model multi-step evidence gathering.

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/run_sft.py \
        --data training_data/output/sft_sharegpt.json \
        --output training_data/output/qwen3-8b-tooluse-lora \
        --epochs 3
"""
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


def format_sharegpt_to_chat(example):
    """Convert ShareGPT format to chat template format."""
    messages = []
    for turn in example["conversations"]:
        if turn["from"] == "human":
            messages.append({"role": "user", "content": turn["value"]})
        elif turn["from"] == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})
        elif turn["from"] == "tool":
            messages.append({"role": "user", "content": f"Tool observation: {turn['value']}"})
    return {"messages": messages}


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning on tool-use trajectories")
    parser.add_argument("--data", type=Path, default=Path("training_data/output/sft_sharegpt.json"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output", type=Path, default=Path("training_data/output/qwen3-8b-tooluse-lora"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit training samples for quick testing")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    raw_data = json.load(open(args.data))
    print(f"  Total examples: {len(raw_data)}")

    if args.max_samples:
        raw_data = raw_data[:args.max_samples]
        print(f"  Using first {len(raw_data)} examples")

    # Convert to chat format
    converted = [format_sharegpt_to_chat(ex) for ex in raw_data]
    dataset = Dataset.from_list(converted)
    print(f"  Dataset size: {len(dataset)}")

    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training config
    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        max_length=args.max_seq_length,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} grad accum = {args.batch_size * args.grad_accum} effective")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max seq length: {args.max_seq_length}")

    trainer.train()

    # Save
    print(f"\nSaving to {args.output}...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done!")


if __name__ == "__main__":
    main()
