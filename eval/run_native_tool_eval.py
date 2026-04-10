#!/usr/bin/env python3
"""Evaluate the SFT model using Qwen3.5 native tool calling format.

The model generates <tool_call> blocks which we parse and execute
against the real index, then feed back as <tool_response>.

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval/run_native_tool_eval.py \
        --benchmark qa-data/benchmark/v3/benchmark_combined.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v3_native_tool_eval.jsonl \
        --adapter training_data/output/qwen35-9b-native-tools-lora
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
from construct_tool_trajectories import TOOL_SCHEMAS, simulate_tool_output


def load_model(base_model, adapter_path):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, tools, max_new_tokens=300):
    text = tokenizer.apply_chat_template(
        messages, tools=tools,
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def parse_tool_call(response):
    """Parse native Qwen3.5 tool call from response text."""
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        response, re.DOTALL
    )
    if not match:
        return None

    func_name = match.group(1)
    params_block = match.group(2)

    args = {}
    for param_match in re.finditer(
        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>',
        params_block, re.DOTALL
    ):
        key = param_match.group(1)
        val = param_match.group(2).strip()
        # Try to parse as int
        try:
            val = int(val)
        except ValueError:
            pass
        args[key] = val

    return {"name": func_name, "arguments": args}


def execute_parsed_tool(parsed_call, index_data, video_id):
    """Execute a parsed tool call against the index cache."""
    name = parsed_call["name"]
    args = parsed_call["arguments"]

    # Parse timestamp if present
    timestamp_ms = None
    for key in ["timestamp", "start_time"]:
        if key in args:
            m = re.match(r'(\d+):(\d+)', str(args[key]))
            if m:
                timestamp_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

    return simulate_tool_output(name, index_data, timestamp_ms)


def run_agent(model, tokenizer, question, video_id, index_data, mc_options=None, max_turns=3):
    messages = [{"role": "user", "content": question}]

    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(mc_options.items()))
        messages[0]["content"] += f"\n\nOptions:\n{opts}\nSelect one option (A/B/C/D) after searching for evidence."

    tool_calls = []
    observations = []

    for turn in range(max_turns):
        response = generate(model, tokenizer, messages, TOOL_SCHEMAS)

        # Strip end token
        response = response.replace("<|im_end|>", "").strip()

        # Check for tool call
        parsed = parse_tool_call(response)

        if parsed:
            # Execute tool
            observation = execute_parsed_tool(parsed, index_data, video_id)
            tool_calls.append(f"{parsed['name']}({parsed['arguments']})")
            observations.append(observation[:500])

            # Add to conversation in native format
            messages.append({
                "role": "assistant",
                "content": response.split("<tool_call>")[0].strip() if "<tool_call>" in response else "",
                "tool_calls": [{
                    "type": "function",
                    "function": parsed
                }]
            })
            messages.append({"role": "tool", "content": observation[:800]})
        else:
            # No tool call -- this is the final answer
            # Strip thinking tags
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            return {
                "answer": clean,
                "tool_calls": tool_calls,
                "observations": observations,
                "turns": turn + 1,
            }

    # Max turns -- extract whatever answer we have
    return {
        "answer": response if response else "",
        "tool_calls": tool_calls,
        "observations": observations,
        "turns": max_turns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.base_model, str(args.adapter))

    print(f"\nNative Tool Eval")
    print(f"Questions: {len(questions)}")
    print(f"Max turns: {args.max_turns}")

    stats = {"total": 0, "no_index": 0, "total_tools": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            fmt = q.get("format", "free_text")

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                continue

            result = run_agent(
                model, tokenizer, q["question"], video_id, idx,
                mc_options=mc_options, max_turns=args.max_turns
            )

            predicted = result["answer"]
            if fmt == "multiple_choice":
                # Extract MC letter
                p = predicted.strip().upper()
                if p and p[0] in "ABCD" and (len(p) == 1 or p[1] in ".):, "):
                    predicted = p[0]
                else:
                    m = re.search(r'(?:answer|option)\s*(?:is\s*)?([A-D])\b', p, re.IGNORECASE)
                    if m:
                        predicted = m.group(1).upper()
                    else:
                        m = re.search(r'\b([A-D])\b', p)
                        if m:
                            predicted = m.group(1).upper()

            stats["total"] += 1
            stats["total_tools"] += len(result["tool_calls"])

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": q["question"],
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "tool_calls": result["tool_calls"],
                "num_tool_calls": len(result["tool_calls"]),
                "turns": result["turns"],
            }
            fout.write(json.dumps(output) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                n = max(1, stats["total"])
                print(f"  [{i+1}/{len(questions)}] avg tools: {stats['total_tools']/n:.1f}", flush=True)

    n = max(1, stats["total"])
    print(f"\nDone. {stats['total']} answered, avg tools: {stats['total_tools']/n:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
