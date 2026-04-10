#!/usr/bin/env python3
"""Score segment retrieval quality against ground truth time ranges.

Evaluates whether the retrieval system found the correct time range
for each question, independent of answer quality.

Metrics:
- Hit rate: did any retrieved segment overlap the ground truth?
- Temporal IoU: intersection over union of retrieved vs ground truth ranges
- Precision@K: what fraction of retrieved segments are relevant?

Supports scoring predictions from run_index_qa.py (which records cited_segments)
or direct retrieval output.

Usage:
    # Score predictions file (uses cited_segments from grounding)
    python eval/score_retrieval.py \
        --predictions eval/results/index_qa_35.jsonl \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --output eval/results/retrieval_scores.jsonl

    # Score with oracle comparison
    python eval/score_retrieval.py \
        --predictions eval/results/index_qa_35.jsonl \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --oracle
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_time_range(s):
    """Parse time range strings like '3.0s-15.0s' or '3s-15s' into (start_ms, end_ms)."""
    m = re.match(r'([\d.]+)s?\s*-\s*([\d.]+)s?', s.strip())
    if m:
        return int(float(m.group(1)) * 1000), int(float(m.group(2)) * 1000)
    return None


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def overlap_ms(a_start, a_end, b_start, b_end):
    if not overlaps(a_start, a_end, b_start, b_end):
        return 0
    return min(a_end, b_end) - max(a_start, b_start)


def temporal_iou(a_start, a_end, b_start, b_end):
    """Intersection over union of two time ranges."""
    intersection = overlap_ms(a_start, a_end, b_start, b_end)
    if intersection == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def score_retrieval(prediction, benchmark_q):
    """Score retrieval for a single question.

    Returns dict with hit, best_iou, precision metrics.
    """
    # Get ground truth time ranges
    gt_times = benchmark_q.get("source_segment_times", [])
    if not gt_times:
        return None

    gt_ranges = []
    for t in gt_times:
        if isinstance(t, dict):
            gt_ranges.append((t.get("start_ms", 0), t.get("end_ms", 0)))
        elif isinstance(t, (list, tuple)) and len(t) >= 2:
            gt_ranges.append((t[0], t[1]))

    if not gt_ranges:
        return None

    # Get retrieved time ranges from prediction
    retrieved_ranges = []

    # From cited_segments in grounding
    cited = prediction.get("grounding", {}).get("cited_segments", [])
    if isinstance(cited, list):
        for seg in cited:
            if isinstance(seg, str):
                parsed = parse_time_range(seg)
                if parsed:
                    retrieved_ranges.append(parsed)
            elif isinstance(seg, dict):
                s = seg.get("start_ms", seg.get("start", 0))
                e = seg.get("end_ms", seg.get("end", 0))
                if isinstance(s, str):
                    s = float(s.replace("s", "")) * 1000
                if isinstance(e, str):
                    e = float(e.replace("s", "")) * 1000
                retrieved_ranges.append((int(s), int(e)))

    if not retrieved_ranges:
        return {
            "hit": False,
            "best_iou": 0.0,
            "mean_iou": 0.0,
            "precision": 0.0,
            "n_retrieved": 0,
            "n_ground_truth": len(gt_ranges),
            "no_retrieval": True,
        }

    # Score
    # Hit: does any retrieved segment overlap any ground truth?
    hit = False
    best_iou = 0.0
    ious = []
    relevant_retrieved = 0

    for r_start, r_end in retrieved_ranges:
        is_relevant = False
        for gt_start, gt_end in gt_ranges:
            if overlaps(r_start, r_end, gt_start, gt_end):
                hit = True
                is_relevant = True
                iou = temporal_iou(r_start, r_end, gt_start, gt_end)
                ious.append(iou)
                best_iou = max(best_iou, iou)
        if is_relevant:
            relevant_retrieved += 1

    precision = relevant_retrieved / len(retrieved_ranges) if retrieved_ranges else 0.0
    mean_iou = sum(ious) / len(ious) if ious else 0.0

    return {
        "hit": hit,
        "best_iou": round(best_iou, 4),
        "mean_iou": round(mean_iou, 4),
        "precision": round(precision, 4),
        "n_retrieved": len(retrieved_ranges),
        "n_relevant": relevant_retrieved,
        "n_ground_truth": len(gt_ranges),
        "no_retrieval": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Score retrieval quality")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--oracle", action="store_true",
                        help="Also compute oracle retrieval scores (upper bound)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.predictions.with_suffix(".retrieval.jsonl")

    # Load benchmark as lookup
    benchmark = {}
    with open(args.benchmark) as f:
        for line in f:
            q = json.loads(line)
            benchmark[q.get("id", "")] = q

    # Load predictions
    with open(args.predictions) as f:
        predictions = [json.loads(line) for line in f]

    print(f"Scoring retrieval: {len(predictions)} predictions, {len(benchmark)} benchmark questions")

    # Score each prediction
    stats = defaultdict(list)
    no_gt = 0
    no_retrieval = 0

    with open(args.output, "w") as fout:
        for pred in predictions:
            qid = pred.get("question_id", "")
            bq = benchmark.get(qid)
            if not bq:
                continue

            result = score_retrieval(pred, bq)
            if result is None:
                no_gt += 1
                continue

            if result["no_retrieval"]:
                no_retrieval += 1

            pred["retrieval_scores"] = result
            fout.write(json.dumps(pred) + "\n")

            stats["hit"].append(1 if result["hit"] else 0)
            stats["best_iou"].append(result["best_iou"])
            stats["precision"].append(result["precision"])
            if not result["no_retrieval"]:
                stats["mean_iou_with_retrieval"].append(result["mean_iou"])

    # Report
    n = len(stats["hit"])
    print(f"\nRetrieval Scores (n={n})")
    print(f"  No ground truth times: {no_gt}")
    print(f"  No retrieved segments: {no_retrieval}")
    print(f"  Hit rate: {sum(stats['hit'])}/{n} = {sum(stats['hit'])/n*100:.1f}%")
    print(f"  Mean best IoU: {sum(stats['best_iou'])/n:.3f}")
    print(f"  Mean precision: {sum(stats['precision'])/n:.3f}")
    if stats["mean_iou_with_retrieval"]:
        m = stats["mean_iou_with_retrieval"]
        print(f"  Mean IoU (when retrieval exists): {sum(m)/len(m):.3f}")

    # By format
    for fmt in ["multiple_choice", "free_text"]:
        fmt_preds = [p for p in predictions if p.get("format") == fmt and "retrieval_scores" in p]
        if fmt_preds:
            hits = sum(1 for p in fmt_preds if p["retrieval_scores"]["hit"])
            print(f"\n  {fmt}: hit={hits}/{len(fmt_preds)} ({hits/len(fmt_preds)*100:.1f}%)")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
