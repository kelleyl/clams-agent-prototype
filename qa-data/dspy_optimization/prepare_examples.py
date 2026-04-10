#!/usr/bin/env python3
"""Prepare DSPy Examples from existing QA data for GEPA optimization.

Loads llm_comprehension_qa.jsonl, groups by source_evidence (segment),
scores existing questions with the LLM judge to establish baseline quality,
and splits into train/dev sets.

Usage:
    python prepare_examples.py                     # Prepare + score baseline
    python prepare_examples.py --skip-scoring      # Prepare without judge scoring
    python prepare_examples.py --max-examples 50   # Limit example count
"""
import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import dspy

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))
from qa_metric import judge_question, compute_question_score

logger = logging.getLogger(__name__)

QA_DATA_DIR = Path(__file__).resolve().parent.parent
RAW_QA_FILE = QA_DATA_DIR / "raw" / "llm_comprehension_qa.jsonl"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def load_qa_data(qa_file: Path = RAW_QA_FILE) -> list[dict]:
    """Load all QA items from the JSONL file."""
    items = []
    with open(qa_file) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} QA items from {qa_file}")
    return items


def group_by_segment(items: list[dict]) -> list[dict]:
    """Group QA items by source_evidence to create segment-level examples.

    Each segment becomes one DSPy example: the input is the segment evidence,
    and the expected output is the list of questions generated for that segment.
    """
    groups = defaultdict(list)
    for item in items:
        evidence = item.get("source_evidence", "")
        if not evidence:
            continue
        # Use first 200 chars as grouping key (evidence can be long)
        key = evidence[:200]
        groups[key].append(item)

    segments = []
    for key, questions in groups.items():
        first = questions[0]
        # Extract broadcast context
        vid = first.get("video_id", "")
        date = first.get("broadcast_date", "")
        broadcast_context = f"VIDEO: {vid}"
        if date:
            broadcast_context = f"BROADCAST DATE: {date}\n{broadcast_context}"

        segments.append({
            "segment_evidence": first["source_evidence"],
            "broadcast_context": broadcast_context,
            "questions": questions,
            "video_id": vid,
        })

    logger.info(f"Grouped into {len(segments)} unique segments "
                f"({sum(len(s['questions']) for s in segments)} total questions)")
    return segments


def score_existing_questions(segments: list[dict],
                             judge_model: str = "Qwen/Qwen3.5-9B",
                             api_base: str = "http://localhost:8100/v1",
                             backend: str = "vllm") -> list[dict]:
    """Score existing questions with the LLM judge to establish baseline."""
    logger.info(f"Scoring existing questions with judge model: {judge_model}")
    total_questions = sum(len(s["questions"]) for s in segments)
    scored = 0

    for seg in segments:
        seg_scores = []
        for q in seg["questions"]:
            scores = judge_question(
                q, seg["segment_evidence"],
                model=judge_model, api_base=api_base,
                backend=backend,
            )
            if scores:
                composite = compute_question_score(scores)
                q["judge_scores"] = scores
                q["composite_score"] = composite
                seg_scores.append(composite)
            scored += 1
            if scored % 20 == 0:
                logger.info(f"  Scored {scored}/{total_questions} questions")

        if seg_scores:
            seg["avg_quality"] = sum(seg_scores) / len(seg_scores)
        else:
            seg["avg_quality"] = 0.0

    # Log baseline statistics
    all_composites = [
        q["composite_score"]
        for seg in segments
        for q in seg["questions"]
        if "composite_score" in q
    ]
    if all_composites:
        avg = sum(all_composites) / len(all_composites)
        logger.info(f"Baseline quality: {avg:.3f} average composite score "
                    f"({len(all_composites)} questions scored)")

        # Per-dimension averages
        dims = ["cognitive_level", "naturalness", "groundedness", "information_need"]
        for dim in dims:
            vals = [
                q["judge_scores"][dim]
                for seg in segments
                for q in seg["questions"]
                if "judge_scores" in q and dim in q.get("judge_scores", {})
            ]
            if vals:
                logger.info(f"  {dim}: {sum(vals)/len(vals):.2f}/5.0")

    return segments


def segments_to_dspy_examples(segments: list[dict]) -> list[dspy.Example]:
    """Convert segment dicts to DSPy Examples for GEPA."""
    examples = []
    for seg in segments:
        ex = dspy.Example(
            segment_evidence=seg["segment_evidence"],
            broadcast_context=seg["broadcast_context"],
        ).with_inputs("segment_evidence", "broadcast_context")
        examples.append(ex)
    return examples


def split_train_dev(examples: list[dspy.Example],
                    train_ratio: float = 0.3,
                    seed: int = 42) -> tuple[list, list]:
    """Split examples into train and dev sets."""
    import random
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    split_point = int(len(indices) * train_ratio)
    train_indices = sorted(indices[:split_point])
    dev_indices = sorted(indices[split_point:])

    trainset = [examples[i] for i in train_indices]
    devset = [examples[i] for i in dev_indices]

    logger.info(f"Split: {len(trainset)} train, {len(devset)} dev "
                f"(ratio={train_ratio})")
    return trainset, devset


def save_prepared_data(segments: list[dict], trainset: list, devset: list,
                       output_dir: Path = OUTPUT_DIR):
    """Save prepared data for later use."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scored segments
    segments_file = output_dir / "scored_segments.jsonl"
    with open(segments_file, "w") as f:
        for seg in segments:
            # Strip source_evidence from saved file (too large, can reload)
            save_seg = {
                "broadcast_context": seg["broadcast_context"],
                "video_id": seg["video_id"],
                "avg_quality": seg.get("avg_quality"),
                "num_questions": len(seg["questions"]),
                "questions": [
                    {
                        "question": q["question"],
                        "answer": q["answer"],
                        "reasoning_type": q.get("reasoning_type", ""),
                        "modalities_required": q.get("modalities_required", []),
                        "judge_scores": q.get("judge_scores"),
                        "composite_score": q.get("composite_score"),
                    }
                    for q in seg["questions"]
                ],
            }
            f.write(json.dumps(save_seg, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(segments)} scored segments to {segments_file}")

    # Save split indices for reproducibility
    split_file = output_dir / "split_info.json"
    with open(split_file, "w") as f:
        json.dump({
            "total_segments": len(segments),
            "train_size": len(trainset),
            "dev_size": len(devset),
            "total_questions": sum(len(s["questions"]) for s in segments),
        }, f, indent=2)

    logger.info(f"Saved split info to {split_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare DSPy examples for GEPA QA optimization")
    parser.add_argument("--qa-file", type=str, default=str(RAW_QA_FILE),
                        help="Path to llm_comprehension_qa.jsonl")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Skip LLM judge scoring (faster)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Limit number of segment examples")
    parser.add_argument("--judge-model", type=str,
                        default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3.5-9B"),
                        help="Judge model for scoring")
    parser.add_argument("--api-base", type=str,
                        default=os.environ.get("API_BASE", "http://localhost:8100/v1"),
                        help="API base URL")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["ollama", "vllm"],
                        help="LLM backend")
    parser.add_argument("--train-ratio", type=float, default=0.3,
                        help="Fraction of data for training")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load and group
    items = load_qa_data(Path(args.qa_file))
    segments = group_by_segment(items)

    if args.max_examples:
        segments = segments[:args.max_examples]
        logger.info(f"Limited to {len(segments)} segments")

    # Score baseline
    if not args.skip_scoring:
        segments = score_existing_questions(
            segments,
            judge_model=args.judge_model,
            api_base=args.api_base,
            backend=args.backend,
        )

    # Convert and split
    examples = segments_to_dspy_examples(segments)
    trainset, devset = split_train_dev(examples, train_ratio=args.train_ratio)

    # Save
    save_prepared_data(segments, trainset, devset)

    # Summary
    print(f"\nPrepared {len(segments)} segment examples:")
    print(f"  Train: {len(trainset)}")
    print(f"  Dev:   {len(devset)}")
    print(f"  Total questions: {sum(len(s['questions']) for s in segments)}")

    if not args.skip_scoring:
        all_scores = [
            q.get("composite_score", 0)
            for seg in segments
            for q in seg["questions"]
            if "composite_score" in q
        ]
        if all_scores:
            print(f"  Baseline quality: {sum(all_scores)/len(all_scores):.3f} "
                  f"(min={min(all_scores):.3f}, max={max(all_scores):.3f})")


if __name__ == "__main__":
    main()
