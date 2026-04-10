#!/usr/bin/env python3
"""Re-index Chicago TV videos after adding whisper transcripts.

After running whisper on aristotle and copying the output MMIFs back,
this script merges the whisper views into the existing MMIF data,
re-indexes each video, runs SpaCy NER, and grounds entities.

Usage:
    python scripts/reindex_with_whisper.py
    python scripts/reindex_with_whisper.py --whisper-dir data/whisper_outputs
    python scripts/reindex_with_whisper.py --skip-grounding  # faster, NER only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_index import (
    get_video_index,
    extract_entities_from_transcripts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def merge_whisper_into_mmif(original_mmif_path: str, whisper_mmif_path: str) -> dict:
    """Merge whisper views from the whisper output into the original MMIF."""
    with open(original_mmif_path) as f:
        original = json.load(f)
    with open(whisper_mmif_path) as f:
        whisper = json.load(f)

    # Append whisper views to the original MMIF
    original_views = original.get("views", [])
    whisper_views = whisper.get("views", [])

    # Only add views from whisper-wrapper (not duplicate existing views)
    existing_apps = {
        v.get("metadata", {}).get("app", "") for v in original_views
    }
    added = 0
    for view in whisper_views:
        app = view.get("metadata", {}).get("app", "")
        if "whisper" in app.lower() and app not in existing_apps:
            original_views.append(view)
            added += 1

    original["views"] = original_views
    logger.info(f"Merged {added} whisper view(s) into {Path(original_mmif_path).name}")
    return original


def main():
    parser = argparse.ArgumentParser(description="Re-index videos with whisper transcripts")
    parser.add_argument(
        "--whisper-dir",
        default="data/whisper_outputs",
        help="Directory with whisper output MMIFs (default: data/whisper_outputs)",
    )
    parser.add_argument(
        "--original-dir",
        default="data/qwen_outputs",
        help="Directory with original (SWT+qwen) MMIFs (default: data/qwen_outputs)",
    )
    parser.add_argument(
        "--skip-grounding",
        action="store_true",
        help="Skip entity grounding (just NER, faster)",
    )
    args = parser.parse_args()

    whisper_dir = Path(args.whisper_dir)
    original_dir = Path(args.original_dir)

    if not whisper_dir.exists():
        logger.error(f"Whisper output directory not found: {whisper_dir}")
        logger.info("Run whisper on aristotle first, then copy results here:")
        logger.info(f"  scp aristotle:/home/kmlynch/chicago_tv/whisper_batch/output/*.mmif {whisper_dir}/")
        sys.exit(1)

    # Find whisper output files
    whisper_files = sorted(whisper_dir.glob("*_whisper.mmif"))
    if not whisper_files:
        logger.error(f"No *_whisper.mmif files found in {whisper_dir}")
        sys.exit(1)

    logger.info(f"Found {len(whisper_files)} whisper output files")

    index = get_video_index()
    results = []

    for whisper_mmif_path in whisper_files:
        # Derive original MMIF filename: foo_caption_whisper.mmif -> foo_caption.mmif
        name = whisper_mmif_path.stem.replace("_whisper", "")
        original_mmif_path = original_dir / f"{name}.mmif"

        if not original_mmif_path.exists():
            logger.warning(f"Original MMIF not found: {original_mmif_path}, skipping")
            continue

        # Derive video_id from filename: foo_caption -> foo
        video_id = name.replace("_caption", "").replace("_ocr", "")

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_id}")
        logger.info(f"  Original: {original_mmif_path}")
        logger.info(f"  Whisper:  {whisper_mmif_path}")

        # Step 1: Merge whisper views into original MMIF
        merged_mmif = merge_whisper_into_mmif(
            str(original_mmif_path), str(whisper_mmif_path)
        )

        # Save merged MMIF
        merged_dir = Path("data/merged_outputs")
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / f"{video_id}_merged.mmif"
        with open(merged_path, "w") as f:
            json.dump(merged_mmif, f)
        logger.info(f"  Saved merged MMIF: {merged_path}")

        # Step 2: Re-index with merged MMIF
        video_path = merged_mmif.get("documents", [{}])[0].get("properties", {}).get("location", "")
        if video_path.startswith("file://"):
            video_path = video_path[7:]

        try:
            index.build_from_mmif(
                video_id=video_id,
                video_path=video_path,
                mmif_dict=merged_mmif,
                source_mmif_path=str(merged_path),
            )
            logger.info(f"  Re-indexed: {video_id}")
        except Exception as e:
            logger.error(f"  Index failed: {e}")
            results.append({"video_id": video_id, "status": "index_failed", "error": str(e)})
            continue

        # Step 3: Run SpaCy NER on transcripts
        try:
            entity_count = extract_entities_from_transcripts(video_id)
            logger.info(f"  NER: extracted {entity_count} entities")
        except Exception as e:
            logger.warning(f"  NER failed: {e}")
            entity_count = 0

        # Step 4: Ground entities via Wikidata/Wikipedia
        grounded_count = 0
        if not args.skip_grounding:
            try:
                grounded = index.enrich_entities(video_id)
                grounded_count = sum(1 for v in grounded.values() if v)
                logger.info(f"  Grounded: {grounded_count}/{len(grounded)} entities")
            except Exception as e:
                logger.warning(f"  Grounding failed: {e}")

        results.append({
            "video_id": video_id,
            "status": "ok",
            "entities": entity_count,
            "grounded": grounded_count,
        })

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Processed: {ok}/{len(results)} videos")
    for r in results:
        status = r["status"]
        if status == "ok":
            print(f"  {r['video_id']}: {r['entities']} entities, {r['grounded']} grounded")
        else:
            print(f"  {r['video_id']}: {status} — {r.get('error', '')}")


if __name__ == "__main__":
    main()
