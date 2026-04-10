#!/usr/bin/env python3
"""Incrementally update a single video's captions, index, and QA.

Recaptions one video with the latest VLM model, rebuilds its layered index,
and regenerates QA questions — without touching other videos.

Usage:
    # Update one video
    python scripts/update_video.py cpb-aacip-507-154dn40c26

    # Update with a specific captioner
    python scripts/update_video.py cpb-aacip-507-154dn40c26 --captioner clams
    python scripts/update_video.py cpb-aacip-507-154dn40c26 --captioner simple

    # Update multiple videos
    python scripts/update_video.py cpb-aacip-507-154dn40c26 cpb-aacip-507-1v5bc3tf81

    # Just rebuild index + QA (skip captioning, e.g. if captions were already updated)
    python scripts/update_video.py cpb-aacip-507-154dn40c26 --skip-caption

    # List all videos
    python scripts/update_video.py --list
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.video_index import get_video_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

COMBINED_DIR = Path("data/combined_outputs")
INDEX_DIR = Path("data/video_indexes")
QA_FILE = Path("qa-data/raw/llm_comprehension_qa.jsonl")


def find_transnet_mmif(video_id):
    """Find the TransNet MMIF for a video across all batch directories."""
    search_dirs = [
        Path.home() / "newshour_pipeline" / "step3_transnet_raw",
        Path.home() / "newshour_batch2" / "step3_transnet_raw",
        Path.home() / "newshour_batch3" / "step3_transnet_raw",
        Path.home() / "multicollection_batch" / "step3_transnet_raw",
        Path.home() / "fuzzymemories_rerun" / "step3_transnet_raw",
        Path("data/pipeline_missing/step3_transnet"),
    ]
    for d in search_dirs:
        for pattern in [f"{video_id}_transnet.mmif", f"{video_id}.mmif"]:
            p = d / pattern
            if p.exists():
                return p
    return None


def recaption_clams(video_id, transnet_mmif, output_mmif):
    """Run captioning via the CLAMS app CLI."""
    app_dir = Path.home() / "clams_apps" / "app-qwen3vl-captioner"
    cmd = [
        "python3", str(app_dir / "cli.py"),
        "--config", "config/transnet_shots.yaml",
        str(transnet_mmif), str(output_mmif),
    ]
    logger.info(f"Running CLAMS captioner: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(app_dir), capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"CLAMS captioner failed: {result.stderr[-500:]}")
        return False
    return True


def recaption_simple(video_id, transnet_mmif, output_mmif):
    """Run captioning via the simple captioner (bypasses CLAMS framework)."""
    captioner = Path("/tmp/simple_captioner.py")
    if not captioner.exists():
        logger.error("simple_captioner.py not found at /tmp/simple_captioner.py")
        return False
    cmd = ["python3", str(captioner), str(transnet_mmif), str(output_mmif)]
    logger.info(f"Running simple captioner (2.5)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        logger.error(f"Simple captioner failed: {result.stderr[-500:]}")
        return False
    return True


def recaption_simple_35(video_id, transnet_mmif, output_mmif):
    """Run captioning via the Qwen 3.5 simple captioner."""
    captioner = Path("/tmp/simple_captioner_35.py")
    if not captioner.exists():
        logger.error("simple_captioner_35.py not found at /tmp/simple_captioner_35.py")
        return False
    cmd = ["python3", str(captioner), str(transnet_mmif), str(output_mmif)]
    logger.info(f"Running simple captioner (3.5)")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        logger.error(f"Simple captioner failed: {result.stderr[-500:]}")
        return False
    return True


def merge_captions(video_id, caption_mmif):
    """Replace visual caption views in the combined MMIF with new ones."""
    combined_file = COMBINED_DIR / f"{video_id}.mmif"
    if not combined_file.exists():
        logger.error(f"No combined MMIF for {video_id}")
        return False

    with open(caption_mmif) as f:
        cap = json.load(f)
    with open(combined_file) as f:
        combined = json.load(f)

    # Find new captioner view
    new_view = None
    for v in cap.get("views", []):
        if "captioner" in v.get("metadata", {}).get("app", "").lower():
            tds = [a for a in v.get("annotations", []) if "TextDocument" in a.get("@type", "")]
            if tds:
                new_view = v

    if not new_view:
        logger.error("No captioner view in caption MMIF")
        return False

    # Find the TransNet view ID in the combined MMIF
    transnet_view_id = None
    for v in combined.get("views", []):
        app = v.get("metadata", {}).get("app", "")
        if "transnet" in app.lower():
            transnet_view_id = v.get("id")
            break

    # Find the TransNet view ID in the caption MMIF (source of origin refs)
    cap_transnet_id = None
    for v in cap.get("views", []):
        app = v.get("metadata", {}).get("app", "")
        if "transnet" in app.lower():
            cap_transnet_id = v.get("id")
            break

    # Remap origin references if view IDs differ
    if cap_transnet_id and transnet_view_id and cap_transnet_id != transnet_view_id:
        logger.info(f"Remapping origins: {cap_transnet_id} -> {transnet_view_id}")
        for ann in new_view.get("annotations", []):
            props = ann.get("properties", {})
            origin = props.get("origin", "")
            if origin.startswith(f"{cap_transnet_id}:"):
                props["origin"] = origin.replace(f"{cap_transnet_id}:", f"{transnet_view_id}:", 1)
            source = props.get("source", "")
            if source.startswith(f"{cap_transnet_id}:"):
                props["source"] = source.replace(f"{cap_transnet_id}:", f"{transnet_view_id}:", 1)

    # Remove old visual caption views
    kept = []
    for v in combined.get("views", []):
        app = v.get("metadata", {}).get("app", "")
        if "captioner" in app.lower():
            pm = v.get("metadata", {}).get("appConfiguration", {}).get("promptMap", {})
            if not pm or isinstance(pm, dict):
                continue  # Skip old visual caption views
        kept.append(v)

    # Add new view
    max_id = max((int(v.get("id", "v_0").replace("v_", "")) for v in kept), default=0)
    new_view["id"] = f"v_{max_id + 1}"
    kept.append(new_view)
    combined["views"] = kept

    with open(combined_file, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Merged new captions into {combined_file}")
    return True


def rebuild_index(video_id):
    """Rebuild the v2 layered index for one video."""
    combined_file = COMBINED_DIR / f"{video_id}.mmif"
    if not combined_file.exists():
        logger.error(f"No combined MMIF for {video_id}")
        return False

    with open(combined_file) as f:
        mmif_dict = json.load(f)

    video_path = ""
    for doc in mmif_dict.get("documents", []):
        if "VideoDocument" in doc.get("@type", ""):
            video_path = doc.get("properties", {}).get("location", "").replace("file://", "")

    index = get_video_index()
    index.build_layers_from_mmif(
        video_id=video_id,
        video_path=video_path,
        mmif_dict=mmif_dict,
        source_mmif_path=str(combined_file),
    )

    # NER + entity resolution + relations
    index.extract_entities_from_layers(video_id)
    index.resolve_entity_variants_v2(video_id)
    index.extract_relations_from_layers(video_id)

    summary = index.get_video_summary_v2(video_id)
    layers = summary.get("layers", {})
    layer_str = ' '.join(f'{k}:{v["count"]}' for k, v in layers.items())
    logger.info(f"Rebuilt index: {layer_str}")
    return True


def regenerate_qa(video_id):
    """Regenerate QA questions for one video, replacing old ones in the JSONL."""
    # Remove old questions for this video
    if QA_FILE.exists():
        with open(QA_FILE) as f:
            lines = f.readlines()
        kept = [l for l in lines if json.loads(l).get("video_id") != video_id]
    else:
        kept = []

    # Generate new questions
    sys.path.insert(0, str(Path(__file__).parent.parent / "qa-data"))
    from generate_llm_comprehension import generate_for_video

    index_file = INDEX_DIR / f"{video_id}.json"
    if not index_file.exists():
        logger.warning(f"No index for {video_id}, skipping QA")
        return 0

    with open(index_file) as f:
        index_data = json.load(f)

    new_questions = generate_for_video(index_data)
    logger.info(f"Generated {len(new_questions)} questions for {video_id}")

    # Write back: kept old + new
    with open(QA_FILE, "w") as f:
        for line in kept:
            f.write(line)
        for q in new_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return len(new_questions)


def update_video(video_id, captioner="simple", skip_caption=False, skip_qa=False):
    """Full incremental update for one video."""
    logger.info(f"{'='*60}")
    logger.info(f"Updating: {video_id}")
    t0 = time.time()

    if not skip_caption:
        transnet_mmif = find_transnet_mmif(video_id)
        if not transnet_mmif:
            logger.error(f"No TransNet MMIF found for {video_id}")
            return False

        caption_out = Path(f"/tmp/{video_id}_updated_captions.mmif")
        if captioner == "clams":
            ok = recaption_clams(video_id, transnet_mmif, caption_out)
        elif captioner == "simple35":
            ok = recaption_simple_35(video_id, transnet_mmif, caption_out)
        else:
            ok = recaption_simple(video_id, transnet_mmif, caption_out)

        if not ok:
            return False

        merge_captions(video_id, caption_out)

    rebuild_index(video_id)

    qa_count = 0
    if not skip_qa:
        try:
            qa_count = regenerate_qa(video_id)
        except Exception as e:
            logger.warning(f"QA generation failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"Done: {video_id} ({elapsed:.0f}s, {qa_count} questions)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Incrementally update video captions, index, and QA")
    parser.add_argument("videos", nargs="*", help="Video IDs to update")
    parser.add_argument("--captioner", choices=["clams", "simple", "simple35"], default="simple35",
                        help="Captioner to use (default: simple35 = Qwen 3.5)")
    parser.add_argument("--skip-caption", action="store_true",
                        help="Skip captioning, just rebuild index + QA")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip QA regeneration")
    parser.add_argument("--list", action="store_true",
                        help="List all indexed videos")
    args = parser.parse_args()

    if args.list:
        for p in sorted(INDEX_DIR.glob("*.json")):
            with open(p) as f:
                d = json.load(f)
            v = d.get("schema_version", 1)
            layers = d.get("layers", {})
            caps = len(layers.get("visual_captions", {}).get("items", []))
            print(f"  {p.stem:60s} v{v} caps={caps}")
        return

    if not args.videos:
        parser.print_help()
        return

    for vid in args.videos:
        update_video(vid, captioner=args.captioner,
                     skip_caption=args.skip_caption, skip_qa=args.skip_qa)


if __name__ == "__main__":
    main()
