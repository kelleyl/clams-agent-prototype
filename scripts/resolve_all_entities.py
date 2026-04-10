#!/usr/bin/env python3
"""Run entity resolution (variant merging) across all indexed videos.

Merges variant spellings like "Woodruff" + "Judy Woodruff" → "Judy Woodruff",
preferring chyron/OCR spellings over ASR-derived ones.

Usage:
    python scripts/resolve_all_entities.py
    python scripts/resolve_all_entities.py --videos cpb-aacip-507-1v5bc3tf81
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_index import get_video_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Resolve entity variants across all indexes")
    parser.add_argument("--videos", nargs="*", help="Only process these video IDs")
    args = parser.parse_args()

    index = get_video_index()
    all_videos = index.get_indexed_videos()

    if args.videos:
        all_videos = [v for v in all_videos if v in args.videos]

    print(f"Resolving entity variants for {len(all_videos)} videos...")

    total_merged = 0
    total_entities = 0

    for i, video_id in enumerate(all_videos, 1):
        try:
            result = index.resolve_entity_variants(video_id)
            merged = result.get("merged", 0)
            total = result.get("total", 0)
            total_merged += merged
            total_entities += total
            if merged > 0:
                print(f"  [{i}/{len(all_videos)}] {video_id[-40:]}: {merged} variants merged ({total} entities)")
        except Exception as e:
            print(f"  [{i}/{len(all_videos)}] {video_id[-40:]}: ERROR: {e}")

    print(f"\nDone. {total_merged} variants merged across {len(all_videos)} videos ({total_entities} total entities)")


if __name__ == "__main__":
    main()
