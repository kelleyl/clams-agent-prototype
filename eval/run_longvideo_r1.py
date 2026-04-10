#!/usr/bin/env python3
"""Evaluate LongVideo-R1 on our QA benchmark.

Runs each question through LongVideo-R1's multi-step reasoning pipeline
(reasoning model + caption model + video QA model) and collects answers.

Outputs a JSONL file with predictions for scoring.

Usage:
    python eval/run_longvideo_r1.py \
        --questions qa-data/raw/llm_comprehension_qa.jsonl \
        --output eval/results/longvideo_r1_predictions.jsonl \
        --video-dir /mnt/llc/llc_data/clams/wgbh/NewsHour \
        --reasoning-url http://127.0.0.1:25600/v1 \
        --caption-url http://127.0.0.1:9081/v1

    # Or run via SSH on aristotle:
    ssh aristotle 'cd ~/LongVideo-R1 && source .venv/bin/activate && python eval_benchmark.py ...'
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def find_video_path(video_id, video_dirs):
    """Find the video file for a given video ID."""
    for vdir in video_dirs:
        # Try direct match
        for ext in [".mp4", ".mkv", ".ts"]:
            path = Path(vdir) / f"{video_id}{ext}"
            if path.exists():
                return str(path)
    return None


def run_longvideo_r1(question, video_path, config):
    """Run LongVideo-R1 CLI on a single question.

    Returns the answer text and timing info, or None on error.
    """
    cmd = [
        "python3", config["cli_path"],
        "--video_path", video_path,
        "--question", question,
        "--reasoning_base_url", config["reasoning_url"],
        "--caption_base_url", config["caption_url"],
        "--videoqa_base_url", config["videoqa_url"],
        "--caption_model", config.get("caption_model", "Qwen2.5-VL-32B"),
        "--videoqa_model", config.get("videoqa_model", "Qwen2.5-VL-32B"),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.get("timeout", 900),  # 15 min max
            cwd=config.get("cwd"),
        )

        # Parse answer from output
        answer = None
        timing = {}
        for line in result.stdout.split("\n"):
            if line.startswith("[Answer]"):
                answer = line[len("[Answer]"):].strip()
            elif line.startswith("[Model Timing Summary]"):
                timing_str = line[len("[Model Timing Summary]"):].strip()
                # Parse: "reasoning: 32.3s / 9 calls, get_caption: 241.2s / 13 calls, ..."
                for part in timing_str.split(","):
                    part = part.strip()
                    if ":" in part:
                        key, val = part.split(":", 1)
                        timing[key.strip()] = val.strip()

        return {
            "answer": answer,
            "timing": timing,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"answer": None, "timing": {}, "error": "timeout"}
    except Exception as e:
        return {"answer": None, "timing": {}, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate LongVideo-R1 on QA benchmark")
    parser.add_argument("--questions", type=Path, required=True,
                        help="JSONL file with questions")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL with predictions")
    parser.add_argument("--video-dir", type=str, nargs="+",
                        default=["/mnt/llc/llc_data/clams/wgbh/NewsHour",
                                 "/home/kmlynch/fuzzymemories_rerun/masked_videos"],
                        help="Directories to search for video files")
    parser.add_argument("--reasoning-url", type=str,
                        default="http://127.0.0.1:25600/v1")
    parser.add_argument("--caption-url", type=str,
                        default="http://127.0.0.1:9081/v1")
    parser.add_argument("--cli-path", type=str,
                        default="/home/kmlynch/LongVideo-R1/cli.py")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit number of questions (for testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip questions already in output file")
    args = parser.parse_args()

    # Load questions
    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Load existing predictions to skip
    existing = set()
    if args.skip_existing and args.output.exists():
        with open(args.output) as f:
            for l in f:
                pred = json.loads(l)
                existing.add(pred.get("question_id", ""))

    config = {
        "reasoning_url": args.reasoning_url,
        "caption_url": args.caption_url,
        "videoqa_url": args.caption_url,  # Same server for caption + VQA
        "cli_path": args.cli_path,
        "cwd": str(Path(args.cli_path).parent),
        "timeout": 900,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(questions)} questions with LongVideo-R1...")
    print(f"Video dirs: {args.video_dir}")

    total = len(questions)
    correct = 0
    errors = 0

    with open(args.output, "a") as fout:
        for i, q in enumerate(questions):
            qid = q.get("id", f"q-{i}")
            if qid in existing:
                print(f"  [{i+1}/{total}] SKIP (exists): {qid}")
                continue

            video_id = q.get("video_id", q.get("source_guid", ""))
            video_path = find_video_path(video_id, args.video_dir)

            if not video_path:
                print(f"  [{i+1}/{total}] NO VIDEO: {video_id}")
                errors += 1
                continue

            question_text = q["question"]
            expected = q.get("answer", "")

            print(f"  [{i+1}/{total}] {video_id[-25:]}: {question_text[:60]}...", end=" ", flush=True)
            t_start = time.time()

            result = run_longvideo_r1(question_text, video_path, config)
            elapsed = time.time() - t_start

            prediction = {
                "question_id": qid,
                "video_id": video_id,
                "question": question_text,
                "expected_answer": expected if isinstance(expected, str) else json.dumps(expected),
                "predicted_answer": result.get("answer", ""),
                "timing": result.get("timing", {}),
                "elapsed_seconds": round(elapsed, 1),
                "error": result.get("error"),
            }

            fout.write(json.dumps(prediction) + "\n")
            fout.flush()

            status = "OK" if result.get("answer") else "FAIL"
            print(f"{status} ({elapsed:.0f}s)")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
