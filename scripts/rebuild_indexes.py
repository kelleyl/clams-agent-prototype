#!/usr/bin/env python3
"""Rebuild video indexes from combined MMIF outputs.

Uses the v2 multi-layer schema: each CLAMS tool's output becomes its own
timeline layer with independent time boundaries.

After the full pipeline runs on aristotle (SWT + Parakeet ASR + TransNet +
Qwen VLM OCR/caption + Credits OCR + Diarization), copy the final MMIFs
back and run this script to rebuild the indexes with SpaCy NER and entity
grounding.

Usage:
    python scripts/rebuild_indexes.py
    python scripts/rebuild_indexes.py --mmif-dir data/combined_outputs
    python scripts/rebuild_indexes.py --skip-grounding   # NER only, faster
    python scripts/rebuild_indexes.py --skip-ner          # just rebuild indexes
"""

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Rebuild video indexes from combined MMIFs (v2 layered schema)")
    parser.add_argument(
        "--mmif-dir",
        default="data/combined_outputs",
        help="Directory with final combined MMIFs (default: data/combined_outputs)",
    )
    parser.add_argument(
        "--diarization-dir",
        default="data/diarization",
        help="Directory with diarization JSON files (default: data/diarization)",
    )
    parser.add_argument(
        "--skip-ner",
        action="store_true",
        help="Skip SpaCy NER extraction",
    )
    parser.add_argument(
        "--skip-grounding",
        action="store_true",
        help="Skip entity grounding (Wikidata/Wikipedia)",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Only process these video IDs (default: all MMIFs in dir)",
    )
    args = parser.parse_args()

    mmif_dir = Path(args.mmif_dir)
    diar_dir = Path(args.diarization_dir)

    if not mmif_dir.exists():
        logger.error(f"MMIF directory not found: {mmif_dir}")
        logger.info("Copy combined outputs from aristotle first:")
        logger.info(f"  scp aristotle:~/clams-agent-prototype/data/combined_outputs/*.mmif {mmif_dir}/")
        sys.exit(1)

    mmif_files = sorted(mmif_dir.glob("*.mmif"))
    if not mmif_files:
        logger.error(f"No .mmif files found in {mmif_dir}")
        sys.exit(1)

    # Filter to specific videos if requested
    if args.videos:
        mmif_files = [f for f in mmif_files if f.stem in args.videos]

    logger.info(f"Found {len(mmif_files)} MMIF files in {mmif_dir}")
    if diar_dir.exists():
        diar_count = len(list(diar_dir.glob("*.json")))
        logger.info(f"Found {diar_count} diarization files in {diar_dir}")

    index = get_video_index()
    results = []

    for mmif_path in mmif_files:
        video_id = mmif_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {video_id}")

        # Load MMIF
        with open(mmif_path) as f:
            mmif_dict = json.load(f)

        # Log what views are present
        for v in mmif_dict.get("views", []):
            app = v.get("metadata", {}).get("app", "?")
            n_ann = len(v.get("annotations", []))
            app_short = app.split("/")[-2] if "/" in app else app
            logger.info(f"  View: {app_short} ({n_ann} annotations)")

        # Extract video path from MMIF
        video_path = ""
        for doc in mmif_dict.get("documents", []):
            if "VideoDocument" in doc.get("@type", ""):
                video_path = doc.get("properties", {}).get("location", "")
                if video_path.startswith("file://"):
                    video_path = video_path[7:]
                break

        # Check for diarization data
        diar_path = diar_dir / f"{video_id}.json"
        diar_path_str = str(diar_path) if diar_path.exists() else None
        if diar_path_str:
            logger.info(f"  Diarization: found")

        # Step 1: Build layered index from MMIF + diarization
        layer_counts = {}
        try:
            index_doc = index.build_layers_from_mmif(
                video_id=video_id,
                video_path=video_path,
                mmif_dict=mmif_dict,
                source_mmif_path=str(mmif_path),
                diarization_path=diar_path_str,
            )
            layer_counts = {
                k: len(v.get("items", []))
                for k, v in index_doc.get("layers", {}).items()
            }
            logger.info(f"  Layers: {layer_counts}")
        except Exception as e:
            logger.error(f"  Index failed: {e}", exc_info=True)
            results.append({"video_id": video_id, "status": "index_failed", "error": str(e)})
            continue

        # Step 2: SpaCy NER on all text layers
        entity_count = 0
        if not args.skip_ner:
            try:
                entity_count = index.extract_entities_from_layers(video_id)
                logger.info(f"  NER: {entity_count} entities extracted")
            except Exception as e:
                logger.warning(f"  NER failed: {e}")

        # Step 3: Entity resolution (merge variant spellings)
        merged_count = 0
        if not args.skip_ner:
            try:
                result = index.resolve_entity_variants_v2(video_id)
                merged_count = result.get("merged", 0)
                if merged_count:
                    logger.info(f"  Entity resolution: {merged_count} variants merged")
            except Exception as e:
                logger.warning(f"  Entity resolution failed: {e}")

        # Step 4: Relation extraction (SVO triples)
        relation_count = 0
        if not args.skip_ner:
            try:
                relation_count = index.extract_relations_from_layers(video_id)
                logger.info(f"  Relations: {relation_count} triples extracted")
            except Exception as e:
                logger.warning(f"  Relation extraction failed: {e}")

        # Step 5: Ground entities via Wikidata + Wikipedia
        grounded_count = 0
        if not args.skip_ner and not args.skip_grounding:
            try:
                grounded = index.enrich_entities_v2(video_id)
                grounded_count = sum(1 for v in grounded.values() if v)
                total = len(grounded)
                logger.info(f"  Grounded: {grounded_count}/{total} entities")
            except Exception as e:
                logger.warning(f"  Grounding failed: {e}")

        results.append({
            "video_id": video_id,
            "status": "ok",
            "layers": layer_counts,
            "entities": entity_count,
            "merged": merged_count,
            "relations": relation_count,
            "grounded": grounded_count,
        })

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    ok = sum(1 for r in results if r["status"] == "ok")
    total_ents = sum(r.get("entities", 0) for r in results if r["status"] == "ok")
    total_grounded = sum(r.get("grounded", 0) for r in results if r["status"] == "ok")
    total_rels = sum(r.get("relations", 0) for r in results if r["status"] == "ok")
    print(f"Processed: {ok}/{len(results)} videos")
    print(f"Total entities: {total_ents}")
    print(f"Total relations: {total_rels}")
    print(f"Total grounded: {total_grounded}")
    print()
    for r in results:
        if r["status"] == "ok":
            layers = r.get("layers", {})
            layer_str = " ".join(f"{k}:{v}" for k, v in layers.items())
            print(f"  {r['video_id'][:45]:45s} {layer_str}  ent:{r['entities']:4d}  rel:{r['relations']:4d}  grnd:{r['grounded']:3d}")
        else:
            print(f"  {r['video_id'][:45]:45s} FAILED: {r.get('error', '')}")


if __name__ == "__main__":
    main()
