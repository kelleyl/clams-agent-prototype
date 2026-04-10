#!/usr/bin/env python3
"""
Compare face-tracker v1 vs v2 outputs side by side.

Reads MMIF files from both output directories and produces a summary table
showing how quality filtering and tighter clustering affected results.
"""

import json
import sys
from pathlib import Path


def parse_face_mmif(mmif_path: Path) -> dict:
    """Extract key stats from a face-tracker MMIF output."""
    with open(mmif_path) as f:
        data = json.load(f)

    # Find the face-tracker view with the most annotations
    face_view = None
    best_ann_count = 0
    for view in data.get("views", []):
        app_id = view.get("metadata", {}).get("app", "")
        if "face" in app_id.lower():
            ann_count = len(view.get("annotations", []))
            if ann_count > best_ann_count:
                face_view = view
                best_ann_count = ann_count

    if face_view is None:
        return {"persons": 0, "timeframes": 0, "bboxes": 0, "person_labels": []}

    annotations = face_view.get("annotations", [])
    timeframes = [a for a in annotations
                  if "TimeFrame" in a.get("@type", "")]
    bboxes = [a for a in annotations
              if "BoundingBox" in a.get("@type", "")]

    # Extract unique person labels
    person_labels = set()
    for tf in timeframes:
        label = tf.get("properties", {}).get("label", "")
        if label:
            person_labels.add(label)

    # Compute per-person segment counts
    person_segments = {}
    for tf in timeframes:
        label = tf.get("properties", {}).get("label", "")
        person_segments[label] = person_segments.get(label, 0) + 1

    # Compute per-person total detections (bboxes)
    person_detections = {}
    for bb in bboxes:
        label = bb.get("properties", {}).get("label", "")
        person_detections[label] = person_detections.get(label, 0) + 1

    # Persons with >1 segment (likely recurring characters)
    recurring = sum(1 for count in person_segments.values() if count > 1)

    # Average detections per person
    avg_dets = (len(bboxes) / len(person_labels)) if person_labels else 0

    return {
        "persons": len(person_labels),
        "timeframes": len(timeframes),
        "bboxes": len(bboxes),
        "recurring": recurring,
        "avg_dets_per_person": avg_dets,
        "person_segments": person_segments,
    }


def short_name(filename: str) -> str:
    """Shorten video filename for display."""
    name = filename.replace("_faces.mmif", "").replace("_faces_v2.mmif", "")
    # Truncate long names
    if len(name) > 45:
        name = name[:42] + "..."
    return name


def main():
    v1_dir = Path(__file__).parent.parent / "data" / "face_tracker_outputs"
    v2_dir = Path(__file__).parent.parent / "data" / "face_tracker_outputs_v2"

    if not v1_dir.exists():
        print(f"v1 directory not found: {v1_dir}")
        sys.exit(1)
    if not v2_dir.exists():
        print(f"v2 directory not found: {v2_dir}")
        sys.exit(1)

    # Match v1 and v2 files by video name
    v1_files = {f.stem.replace("_faces", ""): f for f in v1_dir.glob("*_faces.mmif")}
    v2_files = {f.stem.replace("_faces_v2", ""): f for f in v2_dir.glob("*_faces_v2.mmif")}

    all_videos = sorted(set(v1_files.keys()) | set(v2_files.keys()))

    if not all_videos:
        print("No matching files found.")
        sys.exit(1)

    # Header
    print()
    print("=" * 120)
    print("Face Tracker v1 vs v2 Comparison")
    print("v2 changes: quality filtering (minFaceQuality=0.35) + tighter clustering (dist 0.65 vs 0.8)")
    print("=" * 120)
    print()
    print(f"{'Video':<47} {'v1 Persons':>10} {'v2 Persons':>10} {'Δ':>5}  "
          f"{'v1 TFs':>7} {'v2 TFs':>7} {'Δ':>5}  "
          f"{'v1 BBs':>7} {'v2 BBs':>7} {'Δ%':>6}")
    print("-" * 120)

    totals = {"v1_p": 0, "v2_p": 0, "v1_tf": 0, "v2_tf": 0, "v1_bb": 0, "v2_bb": 0}

    for video in all_videos:
        name = short_name(video)

        v1 = parse_face_mmif(v1_files[video]) if video in v1_files else None
        v2 = parse_face_mmif(v2_files[video]) if video in v2_files else None

        if v1 and v2:
            dp = v2["persons"] - v1["persons"]
            dtf = v2["timeframes"] - v1["timeframes"]
            dbb_pct = ((v2["bboxes"] - v1["bboxes"]) / v1["bboxes"] * 100) if v1["bboxes"] else 0

            print(f"{name:<47} {v1['persons']:>10} {v2['persons']:>10} {dp:>+5}  "
                  f"{v1['timeframes']:>7} {v2['timeframes']:>7} {dtf:>+5}  "
                  f"{v1['bboxes']:>7} {v2['bboxes']:>7} {dbb_pct:>+5.0f}%")

            totals["v1_p"] += v1["persons"]
            totals["v2_p"] += v2["persons"]
            totals["v1_tf"] += v1["timeframes"]
            totals["v2_tf"] += v2["timeframes"]
            totals["v1_bb"] += v1["bboxes"]
            totals["v2_bb"] += v2["bboxes"]
        elif v1:
            print(f"{name:<47} {v1['persons']:>10} {'—':>10} {'':>5}  "
                  f"{v1['timeframes']:>7} {'—':>7} {'':>5}  "
                  f"{v1['bboxes']:>7} {'—':>7} {'':>6}")
        elif v2:
            print(f"{name:<47} {'—':>10} {v2['persons']:>10} {'':>5}  "
                  f"{'—':>7} {v2['timeframes']:>7} {'':>5}  "
                  f"{'—':>7} {v2['bboxes']:>7} {'':>6}")

    print("-" * 120)

    # Totals
    dp = totals["v2_p"] - totals["v1_p"]
    dtf = totals["v2_tf"] - totals["v1_tf"]
    dbb_pct = ((totals["v2_bb"] - totals["v1_bb"]) / totals["v1_bb"] * 100) if totals["v1_bb"] else 0

    print(f"{'TOTAL':<47} {totals['v1_p']:>10} {totals['v2_p']:>10} {dp:>+5}  "
          f"{totals['v1_tf']:>7} {totals['v2_tf']:>7} {dtf:>+5}  "
          f"{totals['v1_bb']:>7} {totals['v2_bb']:>7} {dbb_pct:>+5.0f}%")
    print()

    # Per-video detail: top persons by segment count
    print("=" * 120)
    print("Per-video detail: top 5 persons by segment count")
    print("=" * 120)
    for video in all_videos:
        v1 = parse_face_mmif(v1_files[video]) if video in v1_files else None
        v2 = parse_face_mmif(v2_files[video]) if video in v2_files else None
        if not v1 or not v2:
            continue
        if v1["persons"] == 0 and v2["persons"] == 0:
            continue

        name = short_name(video)
        print(f"\n  {name}")
        print(f"  {'v1':>4}: ", end="")
        top_v1 = sorted(v1["person_segments"].items(), key=lambda x: -x[1])[:5]
        print("  ".join(f"{label}({count})" for label, count in top_v1) if top_v1 else "(none)")
        print(f"  {'v2':>4}: ", end="")
        top_v2 = sorted(v2["person_segments"].items(), key=lambda x: -x[1])[:5]
        print("  ".join(f"{label}({count})" for label, count in top_v2) if top_v2 else "(none)")

    print()


if __name__ == "__main__":
    main()
