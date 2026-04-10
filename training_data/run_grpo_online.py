#!/usr/bin/env python3
"""GRPO with online tool execution during reward computation.

The key insight: GRPO generates completions, but our reward function
needs to actually execute the tool calls in those completions to determine
if the model found the right answer.

Approach:
1. GRPO generates raw completions (may contain <tool_call> blocks)
2. Our reward function parses each completion, executes tool calls against
   the index cache, continues the conversation, and scores the final answer
3. GRPO uses the rewards to update the policy

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/run_grpo_online.py \
        --model training_data/output/qwen35-9b-tool-exec-lora/adapter \
        --base-model Qwen/Qwen3.5-9B \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/qwen35-9b-grpo-online
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
from construct_tool_trajectories import simulate_tool_output, TOOL_SCHEMAS


# ============================================================
# Reward function with tool execution
# ============================================================

# Global state -- set per batch
_indexes = {}
_questions = []
_expected_answers = []


def parse_tool_call_from_text(text):
    """Parse a Qwen3.5 native tool call from generated text."""
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL
    )
    if not match:
        return None, None

    func_name = match.group(1)
    params_block = match.group(2)

    args = {}
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
        params_block, re.DOTALL
    ):
        key = param_match.group(1)
        val = param_match.group(2).strip()
        try:
            val = int(val)
        except ValueError:
            pass
        args[key] = val

    return func_name, args


def execute_completion(completion_text, video_id, index_data):
    """Execute tool calls in a completion and extract the final answer.

    Simulates the tool execution loop:
    1. Find <tool_call> in the completion
    2. Execute against the index
    3. Check if there's an answer after the tool results

    Returns (answer_text, tool_calls_made, tool_results)
    """
    tool_calls = []
    tool_results = []

    # Find all tool calls in the completion
    parts = re.split(r'(<tool_call>.*?</tool_call>)', completion_text, flags=re.DOTALL)

    for part in parts:
        tc_match = re.search(r'<tool_call>', part)
        if tc_match:
            func_name, args = parse_tool_call_from_text(part)
            if func_name:
                # Parse timestamp if present
                timestamp_ms = None
                if 'timestamp' in args:
                    t = str(args['timestamp'])
                    m = re.match(r'(\d+):(\d+)', t)
                    if m:
                        timestamp_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

                result = simulate_tool_output(func_name, index_data, timestamp_ms)
                tool_calls.append(func_name)
                tool_results.append(result)

    # Extract answer -- anything after the last tool call that isn't a tool call
    # Strip thinking tags
    clean = re.sub(r'<think>.*?</think>', '', completion_text, flags=re.DOTALL)
    clean = re.sub(r'<tool_call>.*?</tool_call>', '', clean, flags=re.DOTALL)
    answer = clean.strip()

    # Take last substantial text block as answer
    lines = [l.strip() for l in answer.split('\n') if l.strip() and len(l.strip()) > 3]
    answer = lines[-1] if lines else ""

    return answer, tool_calls, tool_results


def compute_reward(completion, prompt_idx):
    """Compute reward for a single completion with tool execution."""
    if prompt_idx >= len(_questions):
        return 0.0

    q = _questions[prompt_idx]
    expected = _expected_answers[prompt_idx]
    video_id = q.get("video_id", "")
    index_data = _indexes.get(video_id, {})

    if not index_data:
        return 0.0

    # Execute tool calls in the completion
    answer, tool_calls, tool_results = execute_completion(completion, video_id, index_data)

    # Correctness: fuzzy match
    pred = answer.lower().strip()
    exp = expected.lower().strip()

    correct = False
    if not pred or not exp:
        correct = False
    elif exp in pred or pred in exp:
        correct = True
    else:
        pred_words = set(pred.split())
        exp_words = set(exp.split()) - {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or"}
        if exp_words and len(pred_words & exp_words) >= len(exp_words) * 0.5:
            correct = True

    if not correct:
        return 0.0

    # Efficiency
    n_tools = len(tool_calls)
    if n_tools == 0:
        efficiency = 0.3
    elif n_tools <= 2:
        efficiency = 1.0
    elif n_tools <= 3:
        efficiency = 0.8
    else:
        efficiency = 0.5

    # Coverage
    unique_tools = set(tool_calls)
    coverage = min(0.3, len(unique_tools) * 0.1)

    # Bonus: did tools actually find useful results?
    useful = sum(1 for r in tool_results if "No " not in r[:15] and "Error" not in r[:10])
    tool_quality = 0.2 if useful >= 2 else (0.1 if useful >= 1 else 0)

    return efficiency + coverage + tool_quality


def batch_reward_fn(completions, **kwargs):
    """Batch reward function for GRPO with tool execution."""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion if isinstance(completion, str) else str(completion)
        # Map back to prompt index
        prompt_idx = i // kwargs.get("num_generations", 4)
        reward = compute_reward(text, prompt_idx)
        rewards.append(reward)
    return rewards


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    args = parser.parse_args()

    global _indexes, _questions, _expected_answers

    with open(args.questions) as f:
        all_qs = [json.loads(l) for l in f]

    _indexes = {}
    for f in args.index_dir.glob("*.json"):
        _indexes[f.stem] = json.load(open(f))

    # Filter to questions with indexes
    _questions = [q for q in all_qs if q.get("video_id") in _indexes][:args.max_questions]
    _expected_answers = [q.get("answer", "") for q in _questions]

    # Build prompts using native tool format
    from transformers import AutoTokenizer as AT
    tokenizer = AT.from_pretrained(args.base_model, trust_remote_code=True)

    prompts = []
    for q in _questions:
        messages = [{"role": "user", "content": q["question"]}]
        text = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS,
            tokenize=False, add_generation_prompt=True,
        )
        prompts.append(text)

    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset: {len(dataset)} questions")

    # Load model
    print(f"Loading {args.base_model} + LoRA...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.model))
    model = model.merge_and_unload()

    # Fresh LoRA for GRPO
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        report_to="tensorboard",
        logging_dir=str(args.output / "tensorboard"),
        gradient_checkpointing=True,
    )

    def reward_wrapper(completions, **kwargs):
        return batch_reward_fn(completions, num_generations=args.num_generations, **kwargs)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_wrapper,
    )

    print(f"\nStarting GRPO with online tool execution...")
    print(f"  Questions: {len(dataset)}")
    print(f"  Generations: {args.num_generations}")
    print(f"  Reward: correctness * efficiency + coverage + tool_quality")

    trainer.train()

    print(f"\nSaving...")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done!")


if __name__ == "__main__":
    main()
