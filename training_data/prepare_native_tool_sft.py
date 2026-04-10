#!/usr/bin/env python3
"""Convert diverse trajectories to Qwen3.5 native tool calling format.

Uses tokenizer.apply_chat_template() with tools parameter to produce
correctly formatted training data that aligns with Qwen3.5's native
function calling convention.

Usage:
    python training_data/prepare_native_tool_sft.py \
        --trajectories training_data/output/trajectories_v3_diverse.jsonl \
        --output training_data/output/sft_native_tools.jsonl \
        --model Qwen/Qwen3.5-9B
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


# Tool schemas matching our 7 index query tools
# Tool schemas -- video_id is NOT a parameter (bound at system level)
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_asr",
            "description": "Search speech transcripts for keywords. Returns matching segments with timestamps and position in the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for in speech"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Search on-screen text like chyrons, titles, and graphics. Returns text with timestamps and scene labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search in on-screen text"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_entities",
            "description": "Search named entities (people, organizations, places) in the video index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Entity name or keywords to search"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_visual",
            "description": "Get visual descriptions of what is shown on screen in a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_ms": {"type": "integer", "description": "Start time in milliseconds"},
                    "end_ms": {"type": "integer", "description": "End time in milliseconds"}
                },
                "required": ["start_ms", "end_ms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_speakers",
            "description": "Get who is speaking in a time range, with speaker names if identified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_ms": {"type": "integer", "description": "Start time in milliseconds"},
                    "end_ms": {"type": "integer", "description": "End time in milliseconds"}
                },
                "required": ["start_ms", "end_ms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_chapter",
            "description": "Get the chapter/topic at a given time in the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_ms": {"type": "integer", "description": "Time in milliseconds"}
                },
                "required": ["time_ms"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_timeline",
            "description": "Get everything happening in a time range across all layers (speech, on-screen text, visual, speakers).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_ms": {"type": "integer", "description": "Start time in milliseconds"},
                    "end_ms": {"type": "integer", "description": "End time in milliseconds"}
                },
                "required": ["start_ms", "end_ms"]
            }
        }
    },
]


def parse_tool_call_string(tool_call_str):
    """Parse a tool call string like 'search_asr(video_id="abc", query="test")' into structured format."""
    match = re.match(r'(\w+)\((.*)\)', tool_call_str.strip())
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2)

    args = {}
    for part in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([\d]+))', args_str):
        key = part[0]
        if key == "video_id":
            continue  # video_id is bound at system level, not a model parameter
        if part[1]:
            args[key] = part[1]
        elif part[2]:
            args[key] = int(part[2])

    return {"name": func_name, "arguments": args}


def convert_trajectory(traj):
    """Convert a trajectory from custom format to native Qwen3.5 tool calling messages."""
    messages = []
    conversation = traj.get("conversation", [])

    for turn in conversation:
        role = turn["role"]
        content = turn["content"]

        if role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            # Parse out <think> and <tool> / <answer> blocks
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
            tool_match = re.search(r'<tool>(.*?)</tool>', content, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)

            think_text = think_match.group(1).strip() if think_match else ""

            if tool_match:
                # Assistant makes a tool call
                tool_str = tool_match.group(1).strip()
                parsed = parse_tool_call_string(tool_str)

                if parsed:
                    msg = {
                        "role": "assistant",
                        "content": f"<think>\n{think_text}\n</think>" if think_text else "<think>\nLet me search for this information.\n</think>",
                        "tool_calls": [{
                            "type": "function",
                            "function": parsed
                        }]
                    }
                    messages.append(msg)
                else:
                    # Couldn't parse -- skip this turn
                    messages.append({"role": "assistant", "content": content})

            elif answer_match:
                # Final answer
                answer_text = answer_match.group(1).strip()
                msg = {
                    "role": "assistant",
                    "content": f"<think>\n{think_text}\n</think>\n\n{answer_text}" if think_text else answer_text
                }
                messages.append(msg)

            else:
                # Plain assistant message
                messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            # Tool response
            messages.append({"role": "tool", "content": content})

    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--verify", action="store_true", help="Verify format with tokenizer")
    args = parser.parse_args()

    trajs = [json.loads(l) for l in open(args.trajectories)]
    print(f"Loaded {len(trajs)} trajectories")

    # Load tokenizer for verification
    if args.verify:
        print(f"Loading tokenizer: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    converted = []
    stats = Counter()
    skipped = 0

    for i, traj in enumerate(trajs):
        messages = convert_trajectory(traj)

        # Validate: must have at least user + assistant with tool_call + tool + assistant with answer
        has_tool_call = any(m.get("tool_calls") for m in messages if m["role"] == "assistant")
        has_tool_response = any(m["role"] == "tool" for m in messages)
        has_answer = any(m["role"] == "assistant" and not m.get("tool_calls") and len(m.get("content", "")) > 5
                        for m in messages)

        if not (has_tool_call and has_tool_response):
            skipped += 1
            continue

        # Count tool types used
        for m in messages:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    stats[tc["function"]["name"]] += 1

        example = {
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "metadata": {
                "question_id": traj.get("question_id", ""),
                "video_id": traj.get("video_id", ""),
                "strategy": traj.get("strategy", ""),
            }
        }

        # Verify with tokenizer
        if args.verify and i < 3:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tools=TOOL_SCHEMAS,
                    tokenize=False, add_generation_prompt=False
                )
                token_count = len(tokenizer.encode(text))
                print(f"\nExample {i}: {token_count} tokens")
                # Show the tool call part
                tc_start = text.find("<tool_call>")
                if tc_start >= 0:
                    print(f"  Tool call format: {text[tc_start:tc_start+200]}...")
                tr_start = text.find("<tool_response>")
                if tr_start >= 0:
                    print(f"  Tool response: {text[tr_start:tr_start+100]}...")

                if token_count > args.max_seq_length:
                    print(f"  WARNING: exceeds max_seq_length ({args.max_seq_length})")
            except Exception as e:
                print(f"  Tokenization error: {e}")

        converted.append(example)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for ex in converted:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nConverted: {len(converted)} / {len(trajs)} ({skipped} skipped)")
    print(f"Tool usage: {dict(stats.most_common())}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
