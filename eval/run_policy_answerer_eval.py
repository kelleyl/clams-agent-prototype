#!/usr/bin/env python3
"""Two-stage eval: SFT policy model gathers evidence, separate answerer model answers.

Stage 1 (Policy): The SFT-trained model decides which tools to call.
  It generates tool calls, we execute them, feed back results.
  The policy only needs to be good at tool selection, not comprehension.

Stage 2 (Answerer): A fresh LLM call with the question + all gathered
  evidence produces the answer. This model sees the full context and
  just needs reading comprehension.

This separates tool-use skill from answer synthesis skill.

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval/run_policy_answerer_eval.py \
        --benchmark qa-data/raw/qa_v3_test.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v3_policy_answerer_eval.jsonl \
        --policy-adapter training_data/output/qwen35-9b-tool-exec-lora/adapter \
        --answerer-model qwen3:8b
"""
import argparse
import json
import os
import re
import sys
import time
import requests
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
from construct_tool_trajectories import simulate_tool_output


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Tool schemas with model/task_mode parameters matching run_grpo_env.py.
# These give the policy model the ability to select model quality and task focus.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Run speech recognition on the video. Returns timestamped transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "ASR model. 'parakeet' is fast but may miss unclear speech. 'whisper' is slower but more accurate on noisy audio.",
                        "enum": ["parakeet", "whisper"],
                        "default": "parakeet",
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_text_scenes",
            "description": "Detect frames containing text (chyrons, slates, credits). Fast but may miss some. Returns timestamps and scene types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text",
            "description": "Run OCR on a specific frame to read on-screen text. Use after finding a text scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"},
                    "model": {
                        "type": "string",
                        "description": "VLM for OCR. 'qwen-small' is fastest. 'qwen-8b' is good for clear text. 'qwen-30b' reads degraded or small text best.",
                        "enum": ["qwen-small", "qwen-8b", "qwen-30b"],
                        "default": "qwen-small",
                    },
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "caption_frame",
            "description": "Analyze a video frame with a vision-language model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"},
                    "model": {
                        "type": "string",
                        "description": "VLM to use. 'smolvlm' is fastest. 'qwen-small' is fast. 'qwen-8b' balances speed and detail. 'qwen-30b' gives richest output.",
                        "enum": ["smolvlm", "qwen-small", "qwen-8b", "qwen-30b"],
                        "default": "smolvlm",
                    },
                    "task_mode": {
                        "type": "string",
                        "description": "What to look for. 'general_scene' for broad description. 'text_focus' to read on-screen text. 'person_identity' to identify people. 'action_event' to describe what is happening.",
                        "enum": ["general_scene", "text_focus", "person_identity", "action_event"],
                        "default": "general_scene",
                    },
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_speakers",
            "description": "Run speaker diarization to identify who speaks when.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "Run NER on text to find people, organizations, places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Which text to analyze: 'asr' or 'ocr'"}
                },
                "required": ["source"]
            }
        }
    },
]

# Default model per tool (cheapest option, matches run_grpo_env.py)
TOOL_MODEL_DEFAULTS = {
    "run_asr": "parakeet",
    "extract_text": "qwen-small",
    "caption_frame": "smolvlm",
}


# ============================================================
# Stage 1: Policy model (tool selection)
# ============================================================

def load_policy_model(base_model, adapter_path):
    print(f"Loading policy model: {base_model} + {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def policy_generate(model, tokenizer, messages, tools):
    text = tokenizer.apply_chat_template(
        messages, tools=tools,
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=300,
            temperature=0.3, do_sample=True, top_p=0.9,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def parse_tool_call(text):
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


def _parse_timestamp_ms(params):
    """Extract timestamp in ms from tool call params."""
    for key in ["timestamp", "start_time"]:
        if key in params:
            m = re.match(r'(\d+):(\d+)', str(params[key]))
            if m:
                return (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    return None


def execute_tool(func_name, params, index_data):
    """Execute a tool call via the canonical simulate_tool_output interface.

    All model/task_mode logic lives in simulate_tool_output() in
    construct_tool_trajectories.py. This function just parses params
    and delegates, ensuring train/eval parity.
    """
    timestamp_ms = _parse_timestamp_ms(params)
    model = params.get("model", TOOL_MODEL_DEFAULTS.get(func_name))
    task_mode = params.get("task_mode")

    return simulate_tool_output(func_name, index_data, timestamp_ms,
                                model=model, task_mode=task_mode)


def gather_evidence(model, tokenizer, question, index_data, max_turns=3):
    """Run policy model to gather evidence through tool calls."""
    messages = [{"role": "user", "content": question}]
    tool_calls = []
    evidence_pieces = []

    for turn in range(max_turns):
        response = policy_generate(model, tokenizer, messages, TOOL_SCHEMAS)
        response = response.replace("<|im_end|>", "").strip()

        func_name, params = parse_tool_call(response)
        if func_name:
            observation = execute_tool(func_name, params, index_data)
            tool_calls.append({
                "tool": func_name,
                "params": params,
            })
            evidence_pieces.append({
                "tool": func_name,
                "params": params,
                "result": observation[:500],
            })

            # Add to conversation for next turn
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""

            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>" if think else "",
                "tool_calls": [{"type": "function", "function": {"name": func_name, "arguments": params}}]
            })
            messages.append({"role": "tool", "content": observation[:800]})
        else:
            # Policy decided to stop searching
            break

    return tool_calls, evidence_pieces


# ============================================================
# Stage 2: Answerer model (evidence comprehension)
# ============================================================

ANSWERER_PROMPT = """You are answering a question about a broadcast video using evidence gathered by a search agent.

Evidence gathered:
{evidence}

Question: {question}
{mc_section}

Answer based ONLY on the evidence above. For multiple-choice, you MUST select one option (A/B/C/D).
Return a JSON object with:
- "answer": your answer{mc_note}
- "reasoning": brief explanation of which evidence supports your answer"""


def call_answerer(question, evidence_pieces, mc_options=None, model="qwen3:8b"):
    """Call the answerer model via Ollama with gathered evidence."""
    # Format evidence
    evidence_text = ""
    for i, ev in enumerate(evidence_pieces):
        evidence_text += f"\n[{ev['tool']}] {ev['result'][:300]}\n"

    mc_section = ""
    mc_note = ""
    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(mc_options.items()))
        mc_section = f"\nOptions:\n{opts}"
        mc_note = " (for MC, provide the letter AND text)"

    prompt = ANSWERER_PROMPT.format(
        evidence=evidence_text,
        question=question,
        mc_section=mc_section,
        mc_note=mc_note,
    )

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 200},
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        # Strip thinking
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

        try:
            result = json.loads(content)
            return result.get("answer", content)
        except json.JSONDecodeError:
            return content
    except Exception as e:
        return f"Error: {e}"


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path, required=True)
    parser.add_argument("--policy-base", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--answerer-model", type=str, default="qwen3:8b",
                        help="Ollama model for answer synthesis")
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

    # Load policy model
    policy_model, policy_tokenizer = load_policy_model(args.policy_base, str(args.policy_adapter))

    print(f"\nPolicy + Answerer Evaluation")
    print(f"Policy: {args.policy_base} + {args.policy_adapter}")
    print(f"Answerer: {args.answerer_model} via Ollama")
    print(f"Questions: {len(questions)}")
    print(f"Max turns: {args.max_turns}")

    stats = {"total": 0, "no_index": 0, "total_tools": 0, "mc_total": 0, "mc_correct": 0}

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

            # Stage 1: Policy gathers evidence
            tool_calls, evidence = gather_evidence(
                policy_model, policy_tokenizer,
                q["question"], idx, max_turns=args.max_turns
            )

            # Stage 2: Answerer synthesizes answer from evidence
            if evidence:
                raw_answer = call_answerer(
                    q["question"], evidence,
                    mc_options=mc_options, model=args.answerer_model
                )
            else:
                raw_answer = "insufficient evidence"

            # Extract MC letter
            predicted = raw_answer
            if fmt == "multiple_choice" and isinstance(predicted, str):
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

            # Inline scoring preview: MC only. Free-text must be scored post-hoc.
            is_correct = None
            if fmt == "multiple_choice":
                stats["mc_total"] += 1
                is_correct = str(predicted).strip().upper()[:1] == str(expected).strip().upper()[:1]
                if is_correct:
                    stats["mc_correct"] += 1

            stats["total"] += 1
            stats["total_tools"] += len(tool_calls)

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": q["question"],
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "raw_answer": raw_answer[:200],
                "tool_calls": [tc["tool"] for tc in tool_calls],
                "tool_trace": tool_calls,
                "evidence": evidence,
                "num_tool_calls": len(tool_calls),
                "evidence_count": len(evidence),
                "reasoning_type": q.get("reasoning_type", ""),
                "modalities_required": q.get("modalities_required", []),
                "source_segment_times": q.get("source_segment_times", []),
                "inline_correct": is_correct,
                "needs_posthoc_scoring": fmt != "multiple_choice",
            }
            fout.write(json.dumps(output) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                n = max(1, stats["total"])
                mc_n = max(1, stats["mc_total"])
                print(f"  [{i+1}/{len(questions)}] "
                      f"MC preview: {stats['mc_correct']}/{stats['mc_total']} ({stats['mc_correct']/mc_n*100:.0f}%), "
                      f"avg tools: {stats['total_tools']/n:.1f}", flush=True)

            time.sleep(0.3)

    n = max(1, stats["total"])
    print(f"\nDone.")
    mc_n = max(1, stats["mc_total"])
    print(f"MC preview only: {stats['mc_correct']}/{stats['mc_total']} ({stats['mc_correct']/mc_n*100:.1f}%)")
    print(f"Free-text requires post-hoc scoring with eval/score_predictions.py")
    print(f"Avg tools: {stats['total_tools']/n:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
