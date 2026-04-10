#!/usr/bin/env python3
"""Convert verified trajectories to SFT training formats.

Outputs:
- ShareGPT format (for axolotl, LLaMA-Factory)
- Alpaca format (for simple fine-tuning)

Usage:
    python training_data/prepare_sft_from_trajectories.py \
        --trajectories training_data/output/trajectories_v2.jsonl \
        --output-dir training_data/output
"""
import argparse
import json
from pathlib import Path
from collections import Counter


def load_benchmark_mc(benchmark_dir):
    """Load MC options from benchmark files."""
    mc_options = {}
    for name in ["benchmark.jsonl", "mc_questions.jsonl"]:
        path = benchmark_dir / name
        if path.exists():
            for line in open(path):
                q = json.loads(line)
                if q.get("format") == "multiple_choice" and q.get("mc_options"):
                    mc_options[q.get("id", "")] = {
                        "mc_options": q["mc_options"],
                        "mc_correct": q.get("mc_correct", ""),
                        "mc_correct_text": q.get("mc_correct_text", ""),
                    }
    return mc_options


def format_trajectory_for_sft(traj, mc_data=None):
    """Convert a trajectory to chat format with MC support.

    For MC questions: adds options to user message, formats answer as letter.
    For all: ensures tool observations flow into subsequent reasoning.
    """
    qid = traj.get("question_id", "")
    mc_info = (mc_data or {}).get(qid)

    # Override format if MC options exist in benchmark
    if mc_info:
        traj["format"] = "multiple_choice"

    conv = {"conversations": []}

    # Build user message
    user_msg = traj["question"]
    if mc_info:
        opts = "\n".join("  %s. %s" % (k, v) for k, v in sorted(mc_info["mc_options"].items()))
        user_msg += "\n\nOptions:\n" + opts
        user_msg += "\n\nUse the available tools to search for evidence, then select the correct option (A/B/C/D)."
    else:
        user_msg += "\n\nUse the available tools to search for evidence, then provide your answer."

    conv["conversations"].append({"from": "human", "value": user_msg})

    # Add tool-use turns
    for turn in traj["conversation"][1:]:  # skip original user message
        if turn["role"] == "assistant":
            content = turn["content"]
            # For MC: replace <answer>free text</answer> with <answer>letter</answer>
            if mc_info and "<answer>" in content:
                import re
                correct_letter = mc_info["mc_correct"]
                correct_text = mc_info.get("mc_correct_text", "")
                content = re.sub(
                    r'<answer>.*?</answer>',
                    '<answer>%s. %s</answer>' % (correct_letter, correct_text),
                    content, flags=re.DOTALL
                )
            conv["conversations"].append({"from": "gpt", "value": content})
        elif turn["role"] == "tool":
            conv["conversations"].append({"from": "tool", "value": turn["content"]})

    conv["metadata"] = {
        "video_id": traj["video_id"],
        "question_id": qid,
        "tools_used": traj["tools_used"],
        "num_turns": traj["num_turns"],
        "format": traj.get("format", "free_text"),
    }
    return conv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training_data/output"))
    parser.add_argument("--benchmark-dir", type=Path, default=None,
                        help="Directory with MC benchmark files (for MC option injection)")
    parser.add_argument("--verified-only", action="store_true", default=True)
    args = parser.parse_args()

    trajs = [json.loads(l) for l in open(args.trajectories)]
    if args.verified_only:
        trajs = [t for t in trajs if t["verification"]["all_verified"]]

    print("Trajectories: %d total, %d verified" % (
        sum(1 for l in open(args.trajectories)), len(trajs)))

    methods = Counter(t["verification"].get("method", "llm") for t in trajs)
    print("Methods: %s" % dict(methods))

    tools = Counter()
    for t in trajs:
        for tool in t.get("tools_used", []):
            tools[tool] += 1
    print("Tool usage: %s" % dict(tools.most_common()))

    # Load MC options if available
    mc_data = {}
    if args.benchmark_dir:
        mc_data = load_benchmark_mc(args.benchmark_dir)
        print("MC options loaded: %d questions" % len(mc_data))

    # Format for SFT
    sharegpt = []
    mc_count = 0
    for t in trajs:
        conv = format_trajectory_for_sft(t, mc_data)
        sharegpt.append(conv)
        if conv["metadata"]["format"] == "multiple_choice":
            mc_count += 1

    sharegpt_path = args.output_dir / "sft_sharegpt.json"
    with open(sharegpt_path, "w") as f:
        json.dump(sharegpt, f, indent=2, ensure_ascii=False)
    print("ShareGPT: %d examples (%d MC, %d FT) -> %s" % (
        len(sharegpt), mc_count, len(sharegpt) - mc_count, sharegpt_path))


if __name__ == "__main__":
    main()
