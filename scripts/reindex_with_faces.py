#!/usr/bin/env python3
"""Merge face tracker MMIF output into existing video indexes.

Adds two things to each index:
1. Per-segment `persons` field: list of person IDs visible in that segment
2. Top-level `persons` field: per-person summary with appearance times and
   representative bounding box info (supports both lookup directions)

Usage:
    python scripts/reindex_with_faces.py
    python scripts/reindex_with_faces.py --faces-dir data/face_tracker_outputs
    python scripts/reindex_with_faces.py --videos ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INDEX_DIR = Path(__file__).parent.parent / "data" / "video_indexes"
DEFAULT_FACES_DIR = Path(__file__).parent.parent / "data" / "face_tracker_outputs"


def parse_face_tracker_mmif(mmif_path: Path) -> dict:
    """Parse face tracker MMIF output into structured face data.

    Returns:
        {
            "persons": {
                "person_0": {
                    "segments": [{"start_ms": ..., "end_ms": ..., "bbox_count": ...}],
                    "total_screen_time_ms": ...,
                    "total_detections": ...,
                    "representative": {"timePoint": ..., "coordinates": ...},
                },
                ...
            },
            "timeframes": [...],  # raw TimeFrame data for segment matching
            "bboxes": {...},      # raw BoundingBox data keyed by ID
        }
    """
    with open(mmif_path) as f:
        mmif = json.load(f)

    # Find the face-tracker view
    app_view = None
    for v in mmif.get("views", []):
        app = v.get("metadata", {}).get("app", "")
        if "face-tracker" in app:
            app_view = v
            break

    if app_view is None:
        logger.warning(f"No face-tracker view in {mmif_path}")
        return {"persons": {}, "timeframes": [], "bboxes": {}}

    anns = app_view.get("annotations", [])

    bboxes = {}
    timeframes = []
    for a in anns:
        props = a["properties"]
        if "BoundingBox" in a["@type"]:
            bboxes[props["id"]] = props
        elif "TimeFrame" in a["@type"]:
            timeframes.append(props)

    # Build per-person summary
    persons = {}
    for tf in timeframes:
        label = tf["label"]
        if label not in persons:
            persons[label] = {
                "segments": [],
                "total_screen_time_ms": 0,
                "total_detections": 0,
                "representative": None,
            }
        seg_info = {
            "start_ms": tf["start"],
            "end_ms": tf["end"],
            "bbox_count": len(tf.get("targets", [])),
        }
        persons[label]["segments"].append(seg_info)
        persons[label]["total_screen_time_ms"] += tf["end"] - tf["start"]
        persons[label]["total_detections"] += len(tf.get("targets", []))

        # Use representative from this timeframe if we don't have one yet
        reps = tf.get("representatives", [])
        if reps and persons[label]["representative"] is None:
            rep_id = reps[0]
            if rep_id in bboxes:
                bb = bboxes[rep_id]
                persons[label]["representative"] = {
                    "timePoint": bb["timePoint"],
                    "coordinates": bb["coordinates"],
                }

    return {"persons": persons, "timeframes": timeframes, "bboxes": bboxes}


def find_persons_in_segment(start_ms: int, end_ms: int, timeframes: list) -> list:
    """Find all person labels that overlap with a given segment time range."""
    persons = set()
    for tf in timeframes:
        tf_start = tf["start"]
        tf_end = tf["end"]
        # Check temporal overlap
        if tf_start < end_ms and tf_end > start_ms:
            persons.add(tf["label"])
    return sorted(persons)


def merge_faces_into_index(index_path: Path, face_data: dict) -> dict:
    """Merge face tracker data into an existing video index."""
    with open(index_path) as f:
        index = json.load(f)

    timeframes = face_data["timeframes"]
    persons_summary = face_data["persons"]

    # Add per-segment person info
    for seg in index["segments"]:
        seg["persons"] = find_persons_in_segment(
            seg["start_ms"], seg["end_ms"], timeframes
        )

    # Add top-level persons summary (sorted by screen time)
    sorted_persons = sorted(
        persons_summary.items(),
        key=lambda x: x[1]["total_screen_time_ms"],
        reverse=True,
    )
    index["persons"] = {label: data for label, data in sorted_persons}

    return index


def main():
    parser = argparse.ArgumentParser(description="Merge face tracker output into video indexes")
    parser.add_argument(
        "--faces-dir",
        default=str(DEFAULT_FACES_DIR),
        help=f"Directory with face tracker MMIF outputs (default: {DEFAULT_FACES_DIR})",
    )
    parser.add_argument(
        "--index-dir",
        default=str(INDEX_DIR),
        help=f"Directory with video index JSON files (default: {INDEX_DIR})",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Only process these video IDs (default: all)",
    )
    args = parser.parse_args()

    faces_dir = Path(args.faces_dir)
    index_dir = Path(args.index_dir)

    if not faces_dir.exists():
        logger.error(f"Face tracker output directory not found: {faces_dir}")
        logger.info("Copy outputs from aristotle first:")
        logger.info(f"  scp aristotle:/home/kmlynch/chicago_tv/face_tracker_batch/output/*.mmif {faces_dir}/")
        sys.exit(1)

    face_files = sorted(faces_dir.glob("*_faces.mmif"))
    if not face_files:
        logger.error(f"No *_faces.mmif files found in {faces_dir}")
        sys.exit(1)

    logger.info(f"Found {len(face_files)} face tracker outputs")

    for face_file in face_files:
        video_name = face_file.stem.replace("_faces", "")

        if args.videos and video_name not in args.videos:
            continue

        index_path = index_dir / f"{video_name}.json"
        if not index_path.exists():
            logger.warning(f"No index found for {video_name}, skipping")
            continue

        logger.info(f"Processing {video_name}...")

        face_data = parse_face_tracker_mmif(face_file)
        if not face_data["persons"]:
            logger.warning(f"  No persons found in face tracker output")
            continue

        updated_index = merge_faces_into_index(index_path, face_data)

        # Write updated index
        with open(index_path, "w") as f:
            json.dump(updated_index, f, indent=2)

        n_persons = len(face_data["persons"])
        n_segments_with_faces = sum(
            1 for seg in updated_index["segments"] if seg.get("persons")
        )
        logger.info(
            f"  {n_persons} persons, "
            f"{n_segments_with_faces}/{len(updated_index['segments'])} segments with faces"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
