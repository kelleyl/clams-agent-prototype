#!/usr/bin/env python3
"""Ablation studies for the policy + answerer architecture.

Conditions:
- random_tools: Pick random tools, feed results to answerer
- oracle: Give answerer the ground-truth evidence from source_segment_times
- no_tools: Answerer gets question only (parametric knowledge)

This script generates prediction artifacts for ablation conditions. Final
metrics should be computed with eval/score_predictions.py rather than relying
only on the inline MC preview printed here.

Usage:
    python eval/run_ablation_answerer.py \
        --benchmark qa-data/benchmark/v3/test_benchmark.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v3_ablation_answerer.jsonl \
        --condition random_tools \
        --answerer-model qwen3:8b
"""
import argparse
import json
import os
import random
import re
import sys
import time
import requests
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
from construct_tool_trajectories import simulate_tool_output

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ANSWERER_PROMPT = """You are answering a question about a broadcast video using evidence gathered by a search agent.

Evidence gathered:
{evidence}

Question: {question}
{mc_section}

Answer based ONLY on the evidence above. For multiple-choice, you MUST select one option (A/B/C/D).
Return a JSON object with:
- "answer": your answer{mc_note}
- "reasoning": brief explanation"""


def call_answerer(question, evidence_text, mc_options=None, model="qwen3:8b"):
    mc_section = ""
    mc_note = ""
    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in sorted(mc_options.items()))
        mc_section = f"\nOptions:\n{opts}"
        mc_note = " (letter AND text)"

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
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        try:
            return json.loads(content).get("answer", content)
        except json.JSONDecodeError:
            return content
    except Exception as e:
        return f"Error: {e}"


def random_tools_evidence(index_data, n_tools=2):
    """Pick random tools and return their output."""
    tool_options = ["run_asr", "detect_text_scenes", "caption_frame", "identify_speakers"]
    duration_ms = index_data.get("duration_ms", 0)

    evidence = []
    chosen = random.sample(tool_options, min(n_tools, len(tool_options)))
    for tool in chosen:
        if tool in ("caption_frame",) and duration_ms > 0:
            ts = random.randint(0, duration_ms)
            result = simulate_tool_output(tool, index_data, ts)
        else:
            result = simulate_tool_output(tool, index_data)
        evidence.append(f"[{tool}] {result[:300]}")

    return "\n\n".join(evidence), chosen


def oracle_evidence(question_data, index_data):
    """Get ground-truth evidence from source_segment_times and grounding."""
    grounding = question_data.get("grounding", {}) or {}
    evidence_parts = []

    # Use grounding excerpts directly
    if grounding.get("speech_excerpt"):
        evidence_parts.append(f"[Speech] {grounding['speech_excerpt']}")
    if grounding.get("visual_reference"):
        evidence_parts.append(f"[Visual] {grounding['visual_reference']}")
    if grounding.get("ocr_reference"):
        evidence_parts.append(f"[On-screen text] {grounding['ocr_reference']}")

    # Also get index data at source time
    source_time = question_data.get("source_time", "")
    if source_time:
        m = re.match(r'(\d+):(\d+)', source_time)
        if m:
            ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
            # Get ASR and OCR near source time
            layers = index_data.get("layers", {})
            for layer_name in ["asr", "ocr", "visual_captions"]:
                for item in layers.get(layer_name, {}).get("items", []):
                    if item.get("start_ms", 0) < ms + 15000 and item.get("end_ms", 0) > ms - 15000:
                        text = item.get("text", "")
                        if text and len(text) > 20:
                            evidence_parts.append(f"[{layer_name} @{item['start_ms']//1000}s] {text[:200]}")
                            break

    return "\n\n".join(evidence_parts) if evidence_parts else "No evidence available."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=["random_tools", "oracle"], required=True)
    parser.add_argument("--answerer-model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    random.seed(42)

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Ablation: {args.condition}")
    print(f"Answerer: {args.answerer_model}")
    print(f"Questions: {len(questions)}")
    print("Note: inline accuracy here is MC-only preview. Score saved predictions with eval/score_predictions.py for final metrics.\n")

    stats = {"total": 0, "correct": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            fmt = q.get("format", "free_text")
            idx = indexes.get(video_id)
            if not idx:
                continue

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            # Get evidence based on condition
            tool_calls = []
            if args.condition == "random_tools":
                evidence, tool_calls = random_tools_evidence(idx)
            elif args.condition == "oracle":
                evidence = oracle_evidence(q, idx)
                tool_calls = ["oracle"]

            # Answer
            raw_answer = call_answerer(q["question"], evidence, mc_options, args.answerer_model)

            predicted = raw_answer
            if fmt == "multiple_choice" and isinstance(predicted, str):
                p = predicted.strip().upper()
                if p and p[0] in "ABCD" and (len(p) == 1 or p[1] in ".):, "):
                    predicted = p[0]
                else:
                    m = re.search(r'\b([A-D])\b', p)
                    if m:
                        predicted = m.group(1).upper()

            is_correct = str(predicted).strip().upper()[:1] == str(expected).strip().upper()[:1] if fmt == "multiple_choice" else False

            stats["total"] += 1
            if is_correct:
                stats["correct"] += 1

            output = {
                "question_id": q.get("id", ""),
                "video_id": video_id,
                "question": q["question"],
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "condition": args.condition,
                "tool_calls": tool_calls,
                "correct": is_correct,
            }
            fout.write(json.dumps(output) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                n = max(1, stats["total"])
                print(f"  [{i+1}/{len(questions)}] {stats['correct']}/{stats['total']} ({stats['correct']/n*100:.0f}%)")

            time.sleep(0.3)

    n = max(1, stats["total"])
    print(f"\nDone. {args.condition}: {stats['correct']}/{stats['total']} ({stats['correct']/n*100:.1f}%)")


if __name__ == "__main__":
    main()
