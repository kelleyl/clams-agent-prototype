#!/usr/bin/env python3
"""GRPO (Group Relative Policy Optimization) training for tool-use agent.

After SFT teaches basic tool-use format, GRPO teaches:
- When to search vs when to answer (efficiency)
- Which tool to use for which question type (selection)
- When enough evidence has been gathered (stopping)

Following NVIDIA ToolOrchestra's multi-objective reward:
  R(trajectory) = correctness * efficiency_bonus

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/run_grpo.py \
        --model training_data/output/qwen3-8b-diverse-lora \
        --base-model Qwen/Qwen3-8B \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/qwen3-8b-grpo \
        --epochs 1
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent))
from generate_trajectories_v2 import execute_tool, TOOL_DESCRIPTIONS


# ============================================================
# Reward function
# ============================================================

def compute_reward(question, predicted_answer, expected_answer, tool_calls, index_data):
    """Multi-objective reward for a trajectory.

    Components:
    1. Correctness (0 or 1): does the answer match?
    2. Efficiency bonus (0.5-1.0): fewer unnecessary tools = higher bonus
    3. Coverage bonus (0-0.3): using diverse tool types is rewarded

    Returns: float reward in [0, 1.3]
    """
    # Correctness: fuzzy match
    pred_lower = (predicted_answer or "").lower().strip()
    exp_lower = (expected_answer or "").lower().strip()

    if not pred_lower or not exp_lower:
        return 0.0

    # Check exact, substring, or keyword overlap
    correct = False
    if pred_lower == exp_lower:
        correct = True
    elif exp_lower in pred_lower or pred_lower in exp_lower:
        correct = True
    else:
        pred_words = set(pred_lower.split())
        exp_words = set(exp_lower.split())
        stops = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or"}
        pred_content = pred_words - stops
        exp_content = exp_words - stops
        if exp_content and len(pred_content & exp_content) >= len(exp_content) * 0.5:
            correct = True

    if not correct:
        return 0.0

    # Efficiency: penalize excessive tool calls (optimal is 1-3)
    n_tools = len(tool_calls)
    if n_tools == 0:
        efficiency = 0.3  # answered without tools -- not great
    elif n_tools <= 2:
        efficiency = 1.0  # optimal
    elif n_tools <= 4:
        efficiency = 0.8
    else:
        efficiency = 0.5  # too many calls

    # Coverage: reward using different tool types
    tool_types = set(tc.split("(")[0] for tc in tool_calls)
    if len(tool_types) >= 3:
        coverage = 0.3
    elif len(tool_types) >= 2:
        coverage = 0.2
    elif len(tool_types) >= 1:
        coverage = 0.1
    else:
        coverage = 0.0

    return efficiency + coverage


def reward_fn(completions, prompts=None, **kwargs):
    """Batch reward function for GRPO.

    Each completion is a generated trajectory. We parse it to extract
    tool calls and answer, then score against ground truth.
    """
    rewards = []
    questions = kwargs.get("questions", [])
    expected_answers = kwargs.get("expected_answers", [])
    index_data_list = kwargs.get("index_data_list", [])

    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else completion[0].get("content", "")

        # Extract answer
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        predicted = answer_match.group(1).strip() if answer_match else text[-200:]

        # Extract tool calls
        tool_calls = re.findall(r'<tool>(.*?)</tool>', text, re.DOTALL)

        expected = expected_answers[i] if i < len(expected_answers) else ""
        idx = index_data_list[i] if i < len(index_data_list) else {}

        reward = compute_reward(
            questions[i] if i < len(questions) else "",
            predicted, expected, tool_calls, idx
        )
        rewards.append(reward)

    return rewards


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GRPO training for tool-use agent")
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to SFT LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of rollouts per question for GRPO")
    args = parser.parse_args()

    # Load questions
    with open(args.questions) as f:
        questions = [json.loads(l) for l in f][:args.max_questions]

    # Load indexes
    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    # Build dataset
    prompts = []
    expected_answers = []
    index_data_list = []

    system_msg = f"You are a video understanding agent. Use tools to find evidence.\n{TOOL_DESCRIPTIONS}\nUse <tool>tool_call</tool> to search, <answer>your answer</answer> when done."

    for q in questions:
        vid = q.get("video_id", "")
        idx = indexes.get(vid)
        if not idx:
            continue

        prompt = f"{system_msg}\n\nVideo: {vid}\nQuestion: {q['question']}"
        prompts.append(prompt)
        expected_answers.append(q.get("answer", ""))
        index_data_list.append(idx)

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset: {len(dataset)} questions")

    # Load model
    print(f"Loading {args.base_model} + LoRA from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.model))
    model = model.merge_and_unload()  # merge LoRA for GRPO

    # Apply fresh LoRA for GRPO training
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=512,
        learning_rate=5e-6,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    # Custom reward that passes metadata
    def batch_reward_fn(completions, **kwargs):
        return reward_fn(
            completions,
            questions=[q["question"] for q in questions],
            expected_answers=expected_answers,
            index_data_list=index_data_list,
        )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=batch_reward_fn,
    )

    print(f"\nStarting GRPO training...")
    print(f"  Questions: {len(dataset)}")
    print(f"  Generations per question: {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Reward: correctness * efficiency + coverage")

    trainer.train()

    print(f"\nSaving to {args.output}...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done!")


if __name__ == "__main__":
    main()
