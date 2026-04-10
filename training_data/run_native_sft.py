#!/usr/bin/env python3
"""SFT training using Qwen3.5 native tool calling format.

Pre-tokenizes training data using apply_chat_template with tools parameter
to avoid the {% generation %} bug in trl SFTTrainer (GitHub #1522).

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/run_native_sft.py \
        --data training_data/output/sft_native_tools.jsonl \
        --output training_data/output/qwen35-9b-native-tools-lora \
        --epochs 2
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
    Trainer,
    DataCollatorForLanguageModeling,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and pre-tokenize data
    print(f"Loading data from {args.data}...")
    raw_examples = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.max_samples:
        raw_examples = raw_examples[:args.max_samples]
    print(f"  Examples: {len(raw_examples)}")

    # Pre-tokenize using apply_chat_template with tools
    # This avoids the {% generation %} bug in SFTTrainer
    texts = []
    skipped = 0
    for ex in raw_examples:
        try:
            # Support both formats:
            # 1. Already has "messages" and "tools" (from construct_tool_trajectories.py)
            # 2. Has "conversations" in ShareGPT format (legacy)
            if "messages" in ex:
                messages = ex["messages"]
                tools = ex.get("tools")
            elif "conversations" in ex:
                # Convert ShareGPT to messages
                messages = []
                for turn in ex["conversations"]:
                    role = "assistant" if turn["from"] == "gpt" else ("tool" if turn["from"] == "tool" else "user")
                    messages.append({"role": role, "content": turn["value"]})
                tools = None
            else:
                skipped += 1
                continue

            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  Skip: {e}")

    print(f"  Tokenized: {len(texts)}, Skipped: {skipped}")

    # Tokenize to input_ids
    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Split
    split = tokenized.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Check token lengths
    lengths = [len(x) for x in train_dataset["input_ids"]]
    print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)//len(lengths)}")

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA
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

    # Data collator with padding for variable-length sequences
    def collate_fn(features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
            batch["attention_mask"].append([1] * len(f["input_ids"]) + [0] * pad_len)
            batch["labels"].append(f["labels"] + [-100] * pad_len)
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

    # Training
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        report_to="tensorboard",
        logging_dir=str(args.output / "tensorboard"),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        # Skip eval -- padding issues with variable-length sequences
    )

    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.lr}")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    trainer.train()

    # Save immediately after training, before eval can crash
    print(f"\nSaving to {args.output}...")
    save_dir = args.output / "adapter"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the actual PEFT model, unwrapping any Trainer/DDP wrappers
    save_model = trainer.model
    while hasattr(save_model, 'module'):
        save_model = save_model.module

    # Disable gradient checkpointing before saving (can interfere)
    if hasattr(save_model, 'disable_input_require_grads'):
        try:
            save_model.disable_input_require_grads()
        except:
            pass

    try:
        save_model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        print(f"Saved via model.save_pretrained")
    except Exception as e:
        print(f"save_pretrained failed: {e}")
        # Fallback: manually save adapter weights
        state_dict = {k: v for k, v in save_model.state_dict().items() if "lora" in k.lower()}
        torch.save(state_dict, str(save_dir / "adapter_weights.pt"))
        # Save config manually
        if hasattr(save_model, 'peft_config'):
            for adapter_name, config in save_model.peft_config.items():
                config.save_pretrained(str(save_dir))
        print(f"Saved via manual extraction ({len(state_dict)} tensors)")

    # Verify
    import os
    saved_files = os.listdir(save_dir)
    print(f"Files in {save_dir}: {saved_files}")
    if "adapter_config.json" in saved_files:
        print("adapter_config.json found -- save successful")
    else:
        print("WARNING: adapter_config.json NOT found")

    print("Done!")


if __name__ == "__main__":
    main()
