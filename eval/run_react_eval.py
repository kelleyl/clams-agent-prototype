#!/usr/bin/env python3
"""Evaluate a prompted ReAct agent on the QA benchmark.

The model iteratively selects tools, executes them against the real index,
observes results, and reasons toward an answer. This is the intermediate
comparison point between RAG (context-stuffing) and the fine-tuned agent.

Usage:
    python eval/run_react_eval.py \
        --benchmark qa-data/benchmark/v2/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v2_react_qwen3_8b.jsonl \
        --model qwen3:8b
"""
import argparse
import json
import os
import re
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
from generate_trajectories_v2 import execute_tool, TOOL_DESCRIPTIONS

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


REACT_SYSTEM = """You are a video understanding agent answering questions about broadcast videos.
You have access to a structured video index with these tools:

{tools}

To answer a question, you should:
1. Think about what information you need
2. Call a tool to search the index
3. Observe the result
4. Decide if you need more information (call another tool) or can answer

Format your response as:
Thought: [your reasoning about what to search for]
Action: [tool_call, e.g. search_asr(video_id="abc", query="unemployment")]
(wait for observation)
Thought: [reasoning about what you learned]
Action: [another tool call if needed]
(wait for observation)
Thought: [final reasoning combining evidence]
Answer: [your final answer]

{mc_section}

IMPORTANT: One search is usually NOT enough. Different tools reveal different information:
- search_asr finds WHAT was said (topics, claims, quotes)
- search_ocr finds WHO said it (chyrons with names and titles)
- get_speakers finds speaker attribution at a time range
- get_visual finds what was SHOWN on screen

A good strategy:
1. Search ASR for the topic to find the relevant time range
2. Then search OCR or get_speakers at that time range to find identities/titles
3. Then answer combining both

Example:
  Question: "What organization does the guest discussing farm policy represent?"
  Thought: I need to find who's discussing farm policy. Let me search ASR first.
  Action: search_asr(video_id="abc", query="farm policy")
  Observation: [450s-480s] "The farm credit system is facing serious problems..."
  Thought: Found the discussion at 450s. Now I need the speaker's identity from the chyron.
  Action: search_ocr(video_id="abc", query="farm")
  Observation: [455s-460s] "JOHN SMITH / NATIONAL FARMERS UNION" (Chyron)
  Thought: The chyron identifies the speaker as John Smith from National Farmers Union.
  Answer: National Farmers Union

Use 2-3 tool calls to gather evidence from multiple layers before answering."""


def call_llm(messages, model, temperature=0.3, max_tokens=800):
    """Call Ollama chat API."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    # Strip thinking tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def extract_action(text):
    """Extract tool call from model output."""
    # Match Action: tool_name(args)
    m = re.search(r'Action:\s*(\w+\(.*?\))', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Also try without Action: prefix
    m = re.search(r'(\w+)\(video_id=.*?\)', text)
    if m:
        return m.group(0).strip()
    return None


def extract_answer(text):
    """Extract final answer from model output."""
    m = re.search(r'Answer:\s*(.+?)(?:\n|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def run_react_agent(question, video_id, index_data, model, mc_options=None, max_turns=3):
    """Run a ReAct loop: think -> act -> observe -> repeat -> answer."""
    mc_section = ""
    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in mc_options.items())
        mc_section = f"Options:\n{opts}\nYou MUST select one option (A/B/C/D)."

    system = REACT_SYSTEM.format(tools=TOOL_DESCRIPTIONS, mc_section=mc_section)
    user_msg = f"Video: {video_id}\nQuestion: {question}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    tool_calls = []
    observations = []

    for turn in range(max_turns):
        # Get model response
        response = call_llm(messages, model)

        # Check for final answer -- but require at least 1 tool call first
        answer = extract_answer(response)
        if answer and len(tool_calls) > 0:
            messages.append({"role": "assistant", "content": response})
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "observations": observations,
                "turns": turn + 1,
                "full_trace": [m["content"] for m in messages if m["role"] != "system"],
            }

        # Extract and execute tool call
        action = extract_action(response)
        if not action:
            if len(tool_calls) == 0:
                # Force a tool call -- model tried to answer without searching
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"You must search the index first. Try: search_asr(video_id=\"{video_id}\", query=\"relevant keywords from the question\")"})
                continue
            # Has used tools but no more actions -- force an answer
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": "Please provide your final Answer: based on what you found."})
            final = call_llm(messages, model)
            answer = extract_answer(final) or final.strip()
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "observations": observations,
                "turns": turn + 1,
                "full_trace": [m["content"] for m in messages if m["role"] != "system"],
            }

        # Execute tool
        observation = execute_tool(action, index_data)
        tool_calls.append(action)
        observations.append(observation[:500])

        # Add to conversation
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Observation: {observation[:800]}"})

    # Max turns reached -- force answer
    messages.append({"role": "user", "content": "You've used all tool calls. Provide your final Answer: now."})
    final = call_llm(messages, model)
    answer = extract_answer(final) or final.strip()

    return {
        "answer": answer,
        "tool_calls": tool_calls,
        "observations": observations,
        "turns": max_turns,
        "full_trace": [m["content"] for m in messages if m["role"] != "system"],
    }


def main():
    parser = argparse.ArgumentParser(description="ReAct agent evaluation")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=3)
    args = parser.parse_args()

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    # Pre-load indexes
    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"ReAct Agent Evaluation")
    print(f"Model: {args.model}")
    print(f"Questions: {len(questions)}")
    print(f"Max turns: {args.max_turns}")

    stats = {"total": 0, "answered": 0, "no_index": 0, "avg_turns": 0, "avg_tools": 0}

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

            # Run ReAct agent
            result = run_react_agent(
                question_text, video_id, idx, args.model,
                mc_options=mc_options, max_turns=args.max_turns
            )

            # Extract MC letter if applicable
            predicted_answer = result["answer"]
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                p = predicted_answer.strip().upper()
                if p and p[0] in "ABCD" and (len(p) == 1 or p[1] in ".):, "):
                    predicted_answer = p[0]
                else:
                    m = re.search(r'(?:answer|option|choice)\s*(?:is\s*)?([A-D])\b', p, re.IGNORECASE)
                    if m:
                        predicted_answer = m.group(1).upper()
                    else:
                        m = re.search(r'\b([A-D])\b', p)
                        if m:
                            predicted_answer = m.group(1).upper()

            stats["total"] += 1
            stats["answered"] += 1
            stats["avg_turns"] += result["turns"]
            stats["avg_tools"] += len(result["tool_calls"])

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
                avg_t = stats["avg_turns"] / max(1, stats["answered"])
                avg_tc = stats["avg_tools"] / max(1, stats["answered"])
                print(f"  [{i+1}/{len(questions)}] avg turns: {avg_t:.1f}, avg tools: {avg_tc:.1f}", flush=True)

            time.sleep(0.5)

    n = max(1, stats["answered"])
    print(f"\nDone.")
    print(f"Total: {stats['total']}, No index: {stats['no_index']}")
    print(f"Avg turns: {stats['avg_turns']/n:.1f}")
    print(f"Avg tool calls: {stats['avg_tools']/n:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
