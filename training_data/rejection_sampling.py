#!/usr/bin/env python3
"""Rejection sampling fine-tuning (RFT/RAFT).

Generate N rollouts per question using the SFT model with real tool execution.
Keep only the rollouts that find the correct answer. SFT on those.

This is simpler than GRPO and often matches its performance
(per recent findings from Bespoke Labs and others).

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/rejection_sampling.py \
        --adapter training_data/output/qwen35-9b-tool-exec-lora/adapter \
        --base-model Qwen/Qwen3.5-9B \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/rejection_sampled_trajectories.jsonl \
        --rollouts-per-question 4
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
from construct_tool_trajectories import simulate_tool_output, TOOL_SCHEMAS
from prepare_native_tool_sft import TOOL_SCHEMAS as NATIVE_SCHEMAS


def parse_tool_call(text):
    """Parse Qwen3.5 native tool call."""
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL
    )
    if not match:
        return None, None
    func_name = match.group(1)
    params = {}
    for m in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', match.group(2), re.DOTALL):
        val = m.group(2).strip()
        try:
            val = int(val)
        except ValueError:
            pass
        params[m.group(1)] = val
    return func_name, params


def execute_tool_call(func_name, params, index_data):
    """Execute a tool call against the index cache."""
    timestamp_ms = None
    if 'timestamp' in params:
        m = re.match(r'(\d+):(\d+)', str(params['timestamp']))
        if m:
            timestamp_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    if 'start_time' in params:
        m = re.match(r'(\d+):(\d+)', str(params['start_time']))
        if m:
            timestamp_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    return simulate_tool_output(func_name, index_data, timestamp_ms)


def generate_rollout(model, tokenizer, question, index_data, tools, max_turns=3):
    """Generate one rollout with real tool execution."""
    messages = [{"role": "user", "content": question}]
    tool_calls = []
    observations = []

    for turn in range(max_turns):
        text = tokenizer.apply_chat_template(
            messages, tools=tools,
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300,
                temperature=0.7, do_sample=True, top_p=0.9,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=False)
        response = response.replace("<|im_end|>", "").strip()

        # Check for tool call
        func_name, params = parse_tool_call(response)
        if func_name:
            observation = execute_tool_call(func_name, params, index_data)
            tool_calls.append({"name": func_name, "params": params})
            observations.append(observation)

            # Build assistant message with tool call
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""

            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>" if think else "",
                "tool_calls": [{"type": "function", "function": {"name": func_name, "arguments": params}}]
            })
            messages.append({"role": "tool", "content": observation[:800]})
        else:
            # No tool call -- this is the final answer
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            messages.append({"role": "assistant", "content": response})
            return {
                "messages": messages,
                "answer": clean,
                "tool_calls": [tc["name"] for tc in tool_calls],
                "observations": observations,
                "turns": turn + 1,
            }

    # Max turns
    return {
        "messages": messages,
        "answer": "",
        "tool_calls": [tc["name"] for tc in tool_calls],
        "observations": observations,
        "turns": max_turns,
    }


def check_correct(predicted, expected):
    """Check if predicted answer matches expected."""
    pred = predicted.lower().strip()
    exp = expected.lower().strip()
    if not pred or not exp:
        return False
    if exp in pred or pred in exp:
        return True
    exp_words = set(exp.split()) - {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or"}
    pred_words = set(pred.split())
    if exp_words and len(exp_words & pred_words) >= len(exp_words) * 0.5:
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollouts-per-question", type=int, default=4)
    parser.add_argument("--max-questions", type=int, default=100)
    args = parser.parse_args()

    # Load
    with open(args.questions) as f:
        questions = [json.loads(l) for l in f][:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    print(f"Generating rollouts for {len(questions)} questions...")
    print(f"Rollouts per question: {args.rollouts_per_question}")

    stats = defaultdict(int)
    accepted = []

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            idx = indexes.get(q.get("video_id"))
            if not idx:
                continue

            expected = q.get("answer", "")
            best_rollout = None
            best_tools = 999

            for r in range(args.rollouts_per_question):
                rollout = generate_rollout(model, tokenizer, q["question"], idx, NATIVE_SCHEMAS)

                if check_correct(rollout["answer"], expected):
                    stats["correct"] += 1
                    # Keep the most efficient correct rollout
                    if len(rollout["tool_calls"]) < best_tools:
                        best_rollout = rollout
                        best_tools = len(rollout["tool_calls"])
                else:
                    stats["incorrect"] += 1

            if best_rollout:
                # Save as training example
                example = {
                    "messages": best_rollout["messages"],
                    "tools": NATIVE_SCHEMAS,
                    "metadata": {
                        "question_id": q.get("id", ""),
                        "video_id": q.get("video_id", ""),
                        "expected_answer": expected,
                        "predicted_answer": best_rollout["answer"],
                        "tool_calls": best_rollout["tool_calls"],
                        "turns": best_rollout["turns"],
                    }
                }
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                fout.flush()
                stats["accepted"] += 1
            else:
                stats["rejected"] += 1

            if (i + 1) % 10 == 0:
                total = stats["correct"] + stats["incorrect"]
                print(f"  [{i+1}/{len(questions)}] "
                      f"correct: {stats['correct']}/{total} ({stats['correct']/max(1,total)*100:.0f}%), "
                      f"accepted: {stats['accepted']}", flush=True)

    total = stats["correct"] + stats["incorrect"]
    print(f"\nDone.")
    print(f"Total rollouts: {total}")
    print(f"Correct: {stats['correct']} ({stats['correct']/max(1,total)*100:.0f}%)")
    print(f"Accepted questions: {stats['accepted']} / {len(questions)}")
    print(f"Rejected (all wrong): {stats['rejected']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
