#!/usr/bin/env python3
"""Split videos into train/test sets for orchestration agent training.

Stratified by collection, ensuring held-out videos are representative.
All questions for a video go to the same split (no question-level leakage).

Usage:
    python training_data/split_videos.py
    python training_data/split_videos.py --test-fraction 0.25
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)


def classify_collection(video_id: str) -> str:
    if video_id.startswith("cpb-aacip-507") or video_id.startswith("cpb-aacip-225"):
        return "NewsHour"
    elif video_id.startswith("cpb-aacip-526"):
        return "Peabody"
    elif video_id.startswith("cpb-aacip-516") or video_id.startswith("cpb-aacip-75"):
        return "NJN"
    elif any(video_id.startswith(p) for p in ["cpb-aacip-15", "cpb-aacip-114",
                                                "cpb-aacip-151", "cpb-aacip-189"]):
        return "IMLS"
    elif video_id.startswith("cpb-aacip"):
        return "Other"
    else:
        return "FuzzyMemories"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--questions", type=Path,
                        default=Path("qa-data/raw/llm_comprehension_qa.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_data/splits"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all videos and their metadata
    videos = {}
    for f in sorted(args.index_dir.glob("*.json")):
        doc = json.load(open(f))
        vid = f.stem
        videos[vid] = {
            "collection": classify_collection(vid),
            "duration_ms": doc.get("duration_ms", 0),
            "segments": len(doc.get("segments", [])),
            "has_video_file": bool(doc.get("video_path")) and os.path.exists(doc.get("video_path", "")),
        }

    # Count questions per video
    q_counts = defaultdict(int)
    if args.questions.exists():
        with open(args.questions) as f:
            for line in f:
                q = json.loads(line)
                q_counts[q.get("video_id", "")] += 1

    for vid in videos:
        videos[vid]["questions"] = q_counts.get(vid, 0)

    # Group by collection
    by_collection = defaultdict(list)
    for vid, info in videos.items():
        by_collection[info["collection"]].append(vid)

    # Stratified split: take test_fraction from each collection
    train_vids = []
    test_vids = []

    for collection, vids in sorted(by_collection.items()):
        random.shuffle(vids)
        n_test = max(1, int(len(vids) * args.test_fraction))

        # Prefer videos with video files for test set (enables live eval)
        with_file = [v for v in vids if videos[v]["has_video_file"]]
        without_file = [v for v in vids if not videos[v]["has_video_file"]]

        test_pool = with_file[:n_test]
        if len(test_pool) < n_test:
            test_pool.extend(without_file[:n_test - len(test_pool)])

        test_set = set(test_pool[:n_test])
        for v in vids:
            if v in test_set:
                test_vids.append(v)
            else:
                train_vids.append(v)

    # Write splits
    with open(args.output_dir / "train_videos.txt", "w") as f:
        for v in sorted(train_vids):
            f.write(v + "\n")

    with open(args.output_dir / "test_videos.txt", "w") as f:
        for v in sorted(test_vids):
            f.write(v + "\n")

    # Stats
    train_qs = sum(videos[v]["questions"] for v in train_vids)
    test_qs = sum(videos[v]["questions"] for v in test_vids)
    train_segs = sum(videos[v]["segments"] for v in train_vids)
    test_segs = sum(videos[v]["segments"] for v in test_vids)

    stats = {
        "train": {
            "videos": len(train_vids),
            "questions": train_qs,
            "segments": train_segs,
            "by_collection": {},
        },
        "test": {
            "videos": len(test_vids),
            "questions": test_qs,
            "segments": test_segs,
            "by_collection": {},
            "with_video_file": sum(1 for v in test_vids if videos[v]["has_video_file"]),
        },
    }

    for split_name, split_vids in [("train", train_vids), ("test", test_vids)]:
        col_counts = defaultdict(int)
        for v in split_vids:
            col_counts[videos[v]["collection"]] += 1
        stats[split_name]["by_collection"] = dict(col_counts)

    with open(args.output_dir / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("Train/Test Video Split")
    print("=" * 50)
    print(f"Train: {len(train_vids)} videos, {train_qs} questions, {train_segs} segments")
    print(f"Test:  {len(test_vids)} videos, {test_qs} questions, {test_segs} segments")
    print(f"Test videos with .mp4: {stats['test']['with_video_file']}")
    print()
    print("By collection:")
    print(f"  {'Collection':15s} {'Train':>6s} {'Test':>6s}")
    all_cols = sorted(set(list(stats["train"]["by_collection"].keys()) +
                          list(stats["test"]["by_collection"].keys())))
    for col in all_cols:
        tr = stats["train"]["by_collection"].get(col, 0)
        te = stats["test"]["by_collection"].get(col, 0)
        print(f"  {col:15s} {tr:6d} {te:6d}")
    print()
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
