#!/usr/bin/env python3
"""Generate Chain-of-Thought reasoning traces from benchmark QA pairs.

For each benchmark question, constructs a reasoning trace showing how
to find and combine evidence from the video index. Uses the grounding
fields (speech_excerpt, visual_reference, ocr_reference) to build
deterministic traces.

Output format follows LongVILA-R1 / Video-R1 SFT training pattern:
question → <think> reasoning </think> → <answer> answer </answer>

Usage:
    python training_data/generate_cot_traces.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/cot_traces.jsonl
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def build_cot_trace(question_data, index_data=None):
    """Build a chain-of-thought trace from a benchmark question."""
    question = question_data["question"]
    video_id = question_data.get("video_id", "")
    fmt = question_data.get("format", "free_text")
    grounding = question_data.get("grounding", {})

    speech_ex = grounding.get("speech_excerpt") or ""
    visual_ref = grounding.get("visual_reference") or ""
    ocr_ref = grounding.get("ocr_reference") or ""
    reasoning = grounding.get("reasoning") or ""
    modalities = question_data.get("modalities_required", [])

    # Get the answer
    if fmt == "multiple_choice":
        answer = question_data.get("mc_correct_text", "")
    else:
        answer = question_data.get("reference_answer",
                  question_data.get("answer", ""))

    # Build the thinking trace
    think_parts = []

    # Step 1: Identify what we need
    think_parts.append(
        f"This question asks about video {video_id}. "
        f"I need to find evidence combining {' and '.join(modalities) if modalities else 'multiple modalities'}."
    )

    # Step 2: Identify relevant evidence
    evidence_parts = []
    if speech_ex:
        evidence_parts.append(f"Speech transcript says: \"{speech_ex[:200]}\"")
    if ocr_ref:
        evidence_parts.append(f"On-screen text shows: \"{ocr_ref[:200]}\"")
    if visual_ref:
        evidence_parts.append(f"Visual frame shows: \"{visual_ref[:200]}\"")

    if evidence_parts:
        think_parts.append("Relevant evidence found: " + " | ".join(evidence_parts))

    # Step 3: Cross-modal reasoning
    if reasoning:
        think_parts.append(f"Cross-modal reasoning: {reasoning}")
    elif speech_ex and (visual_ref or ocr_ref):
        think_parts.append(
            "Combining the speech content with the visual/on-screen information "
            "to synthesize the answer."
        )

    # Step 4: For MC, include distractor analysis
    if fmt == "multiple_choice":
        correct_letter = question_data.get("mc_correct", "")
        options = question_data.get("mc_options", {})
        if correct_letter and options:
            think_parts.append(
                f"Looking at the options, {correct_letter} matches the evidence. "
                f"The other options are incorrect based on the combined evidence."
            )

    thinking = " ".join(think_parts)
    trace = f"<think>{thinking}</think>\n<answer>{answer}</answer>"

    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning traces from benchmark")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/output/cot_traces.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    # Pre-load indexes
    indexes = {}
    for idx_file in args.index_dir.glob("*.json"):
        indexes[idx_file.stem] = json.load(open(idx_file))

    print(f"Generating CoT traces for {len(questions)} questions...")

    results = []
    skipped = 0

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            idx_data = indexes.get(video_id)

            trace = build_cot_trace(q, idx_data)
            fmt = q.get("format", "free_text")
            answer = q.get("mc_correct_text", q.get("reference_answer", q.get("answer", "")))

            result = {
                "id": f"cot-{q.get('id', f'q-{i:04d}')}",
                "question_id": q.get("id", f"q-{i:04d}"),
                "video_id": video_id,
                "question": q["question"],
                "format": fmt,
                "answer": answer,
                "trace": trace,
                "modalities_required": q.get("modalities_required", []),
                "reasoning_type": q.get("reasoning_type", ""),
            }

            # Add SFT conversation format
            result["conversation"] = [
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": trace},
            ]

            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Generated {len(results)} CoT traces ({skipped} skipped)")
    print(f"Output: {args.output}")

    # Stats
    by_format = {}
    for r in results:
        fmt = r["format"]
        by_format[fmt] = by_format.get(fmt, 0) + 1
    print(f"By format: {by_format}")


if __name__ == "__main__":
    main()
