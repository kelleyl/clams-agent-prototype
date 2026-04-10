#!/usr/bin/env python3
"""Legacy eval for the older SFT-only tool API.

Loads the LoRA adapter on top of Qwen3-8B and runs it on the benchmark
using the same tool-use format it was trained on.

This script is retained for older V2 / legacy experiments. For the current
V3 policy + answerer evaluation flow, use:
  1. eval/run_policy_answerer_eval.py to generate prediction artifacts
  2. eval/score_predictions.py to compute final MC / free-text metrics

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval/run_sft_eval.py \
        --benchmark qa-data/benchmark/v2/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v2_sft_qwen3_8b.jsonl \
        --adapter training_data/output/qwen3-8b-tooluse-lora
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
from generate_trajectories_v2 import execute_tool, TOOL_DESCRIPTIONS


def load_model(base_model, adapter_path):
    """Load base model with LoRA adapter."""
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
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


def generate_response(model, tokenizer, messages, max_new_tokens=800):
    """Generate a response from the model."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response


SFT_SYSTEM = """You are a video understanding agent. Use tools to find evidence, then answer.

Available tools:
- search_asr(video_id, query): Search speech transcripts for keywords
- search_ocr(video_id, query): Search on-screen text (chyrons, titles)
- search_entities(video_id, query): Search named entities
- get_visual(video_id, start_ms, end_ms): Get visual descriptions at a time
- get_speakers(video_id, start_ms, end_ms): Get speaker info at a time
- get_chapter(video_id, time_ms): Get the chapter/topic at a time
- browse_timeline(video_id, start_ms, end_ms): Get everything in a time range

Use <think>reasoning</think> before each action.
Use <tool>tool_call(args)</tool> to search.
Use <answer>your answer</answer> when you have enough evidence."""


def run_sft_agent(model, tokenizer, question, video_id, index_data, mc_options=None, max_turns=3):
    """Run the SFT model in a tool-use loop."""
    user_msg = question
    if mc_options:
        opts = "\n".join("  %s. %s" % (k, v) for k, v in sorted(mc_options.items()))
        user_msg += "\n\nOptions:\n" + opts
        user_msg += "\n\nSearch the index for evidence, then select the correct option."

    messages = [
        {"role": "system", "content": SFT_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    tool_calls = []
    observations = []

    for turn in range(max_turns):
        response = generate_response(model, tokenizer, messages)

        # Strip thinking tags
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Check for answer
        answer_match = re.search(r'<answer>(.*?)</answer>', clean, re.DOTALL)
        if answer_match and len(tool_calls) > 0:
            answer = answer_match.group(1).strip()
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "observations": observations,
                "turns": turn + 1,
            }

        # Check for tool call
        tool_match = re.search(r'<tool>(.*?)</tool>', clean, re.DOTALL)
        if tool_match:
            tool_call = tool_match.group(1).strip()
            observation = execute_tool(tool_call, index_data)
            tool_calls.append(tool_call)
            observations.append(observation[:500])

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation[:800]}"})
        else:
            # No tool and no answer -- try to extract answer from raw text
            if mc_options:
                for letter in "ABCD":
                    if letter in clean[:5]:
                        return {"answer": letter, "tool_calls": tool_calls,
                                "observations": observations, "turns": turn + 1}
            return {
                "answer": clean[:200],
                "tool_calls": tool_calls,
                "observations": observations,
                "turns": turn + 1,
            }

    # Max turns -- extract whatever answer we can
    return {
        "answer": clean[:200] if clean else "",
        "tool_calls": tool_calls,
        "observations": observations,
        "turns": max_turns,
    }


def answer_from_evidence(evidence_text, question, mc_options, model_name="qwen3:8b"):
    """Use base Ollama model to answer MC from gathered evidence."""
    import requests
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    opts = "\n".join("  %s. %s" % (k, v) for k, v in sorted(mc_options.items()))
    prompt = f"""Based on this evidence from a video index, answer the question.

Evidence:
{evidence_text}

Question: {question}

Options:
{opts}

Select one option (A/B/C/D). Return ONLY the letter."""

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 50},
    }, timeout=60)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    # Strip thinking
    import re as _re
    content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B")
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

    print("WARNING: eval/run_sft_eval.py is a legacy path for the older tool API.")
    print("For current V3 evaluation, prefer run_policy_answerer_eval.py + score_predictions.py\n")

    print(f"\nSFT Model Evaluation")
    print(f"Questions: {len(questions)}")
    print(f"Max turns: {args.max_turns}")

    stats = {"total": 0, "no_index": 0, "total_tools": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            question_text = q["question"]
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

            result = run_sft_agent(
                model, tokenizer, question_text, video_id, idx,
                mc_options=mc_options, max_turns=args.max_turns
            )

            predicted_answer = result["answer"]
            if fmt == "multiple_choice" and mc_options:
                # Two-stage: SFT model gathered evidence, base model answers MC
                evidence = "\n".join(result["observations"]) if result["observations"] else predicted_answer
                if evidence:
                    mc_answer = answer_from_evidence(evidence, question_text, mc_options)
                    p = mc_answer.strip().upper()
                    if p and p[0] in "ABCD":
                        predicted_answer = p[0]
                    else:
                        m = re.search(r'\b([A-D])\b', p)
                        if m:
                            predicted_answer = m.group(1).upper()
                        else:
                            predicted_answer = mc_answer
                else:
                    predicted_answer = "A"  # fallback

            stats["total"] += 1
            stats["total_tools"] += len(result["tool_calls"])

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted_answer,
                "tool_calls": result["tool_calls"],
                "num_tool_calls": len(result["tool_calls"]),
                "turns": result["turns"],
                "reasoning_type": q.get("reasoning_type", ""),
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
