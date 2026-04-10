#!/usr/bin/env python3
"""
Batch video processing pipeline.

Processes multiple videos through CLAMS apps, runs ASR transcription,
indexes them, runs NER + entity enrichment, and generates QA evaluation pairs.

Usage:
    # Process all videos in data/videos/
    python batch_pipeline.py

    # Process specific videos
    python batch_pipeline.py --videos data/videos/video1.mp4 data/videos/video2.mp4

    # Custom CLAMS pipeline
    python batch_pipeline.py --apps swt-detection doctr-wrapper

    # Skip already-indexed videos
    python batch_pipeline.py --skip-indexed

    # Just index existing MMIF outputs (no CLAMS app execution)
    python batch_pipeline.py --index-only

    # Run ASR transcription (faster-whisper)
    python batch_pipeline.py --asr
    python batch_pipeline.py --asr --asr-model medium  # better accuracy
    python batch_pipeline.py --asr-only                # just ASR, no CLAMS apps

    # Generate QA after indexing
    python batch_pipeline.py --qa --qa-mc

    # Full pipeline: scene detection + ASR + NER + enrichment + QA
    python batch_pipeline.py --asr --qa --qa-mc
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.clams_executor import UnifiedCLAMSExecutor
from utils.video_index import VideoIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default pipeline: pyscenedetect is CPU-friendly and fast
DEFAULT_APPS = ["pyscenedetect-wrapper"]

# Extended pipeline when GPU is available
GPU_APPS = ["swt-detection", "doctr-wrapper"]

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def discover_videos(video_dir: Path) -> list[Path]:
    """Find all video files in a directory."""
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()
    )
    return videos


def video_id_from_path(video_path: Path) -> str:
    """Derive a video_id from the file path (stem, sanitized)."""
    return video_path.stem.replace(" ", "_")


def run_pipeline_for_video(
    executor: UnifiedCLAMSExecutor,
    video_path: Path,
    apps: list[str],
    app_params: dict[str, dict],
    output_dir: Path,
    timeout: int = 600,
) -> dict | None:
    """Run CLAMS apps on a single video, return final MMIF dict or None on failure."""
    video_id = video_id_from_path(video_path)
    logger.info(f"=== Processing: {video_path.name} ({video_path.stat().st_size / 1e6:.0f} MB) ===")

    # Check for existing MMIF output
    mmif_path = output_dir / f"{video_id}.mmif"
    if mmif_path.exists():
        logger.info(f"  Found existing MMIF: {mmif_path}")
        with open(mmif_path) as f:
            return json.load(f)

    # Run pipeline
    t0 = time.time()
    results = executor.execute_pipeline(
        apps=apps,
        video_path=str(video_path),
        parameters=app_params,
    )
    elapsed = time.time() - t0

    # Check results
    final_mmif = None
    for r in results:
        status = "OK" if r.success else "FAILED"
        logger.info(f"  {r.app_name}: {status} ({r.execution_time_seconds:.1f}s, {r.annotations_created} annotations)")
        if r.success and r.output_mmif:
            final_mmif = r.output_mmif
        elif not r.success:
            logger.error(f"  Error: {r.error}")

    if final_mmif:
        with open(mmif_path, "w") as f:
            json.dump(final_mmif, f)
        logger.info(f"  Saved MMIF: {mmif_path} ({elapsed:.1f}s total)")
    else:
        logger.error(f"  Pipeline failed for {video_path.name} — no output MMIF")

    return final_mmif


# ============================================================================
# ASR Transcription (faster-whisper)
# ============================================================================

def transcribe_video(video_path: Path, model, index_dir: Path) -> dict:
    """Transcribe a video and update its index with ASR text per segment.

    Returns dict with stats: {segments_updated, total_words, duration_s}.
    """
    video_id = video_id_from_path(video_path)
    index_path = index_dir / f"{video_id}.json"

    if not index_path.exists():
        logger.warning(f"  No index found for {video_id}, skipping ASR")
        return {"segments_updated": 0}

    with open(index_path) as f:
        index_doc = json.load(f)

    segments = index_doc.get("segments", [])
    if not segments:
        return {"segments_updated": 0}

    # Check if ASR already done
    has_asr = sum(1 for s in segments if s.get("asr_transcript", "").strip())
    if has_asr > 0:
        logger.info(f"  ASR already present ({has_asr}/{len(segments)} segments), skipping")
        return {"segments_updated": has_asr, "skipped": True}

    # Transcribe full video
    t0 = time.time()
    whisper_segments, info = model.transcribe(
        str(video_path),
        beam_size=5,
        vad_filter=True,  # skip silence for speed
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # Collect all whisper segments with timestamps
    asr_chunks = []
    for ws in whisper_segments:
        asr_chunks.append({
            "start_ms": int(ws.start * 1000),
            "end_ms": int(ws.end * 1000),
            "text": ws.text.strip(),
        })

    transcribe_time = time.time() - t0
    total_words = sum(len(c["text"].split()) for c in asr_chunks)
    logger.info(f"  Whisper: {len(asr_chunks)} chunks, {total_words} words, "
                f"lang={info.language} ({transcribe_time:.1f}s)")

    # Assign ASR chunks to index segments by time overlap
    updated = 0
    for seg in segments:
        seg_start = seg["start_ms"]
        seg_end = seg["end_ms"]
        # Find overlapping ASR chunks
        overlapping_texts = []
        for chunk in asr_chunks:
            # Check overlap: chunk overlaps segment if chunk doesn't end before
            # segment starts AND chunk doesn't start after segment ends
            if chunk["end_ms"] > seg_start and chunk["start_ms"] < seg_end:
                overlapping_texts.append(chunk["text"])
        if overlapping_texts:
            seg["asr_transcript"] = " ".join(overlapping_texts)
            updated += 1

    # Regenerate summaries with ASR content
    for seg in segments:
        from utils.video_index import SegmentEntry
        entry = SegmentEntry.from_dict(seg)
        seg["summary"] = entry.to_summary()

    # Save updated index
    with open(index_path, "w") as f:
        json.dump(index_doc, f, indent=2)

    logger.info(f"  Updated {updated}/{len(segments)} segments with ASR")

    return {
        "segments_updated": updated,
        "total_words": total_words,
        "asr_chunks": len(asr_chunks),
        "duration_s": transcribe_time,
        "language": info.language,
    }


def run_asr_batch(
    videos: list[Path],
    index_dir: Path,
    model_name: str = "small",
) -> list[dict]:
    """Run ASR on multiple videos, updating their indexes."""
    from faster_whisper import WhisperModel

    logger.info(f"Loading faster-whisper model: {model_name}")
    t0 = time.time()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    results = []
    for i, video_path in enumerate(videos, 1):
        logger.info(f"\n[ASR {i}/{len(videos)}] {video_path.name}")
        try:
            stats = transcribe_video(video_path, model, index_dir)
            stats["video"] = video_path.name
            results.append(stats)
        except Exception as e:
            logger.error(f"  ASR failed: {e}")
            results.append({"video": video_path.name, "error": str(e)})

    return results


# ============================================================================
# Indexing & QA
# ============================================================================

def index_video(
    index: VideoIndex,
    video_path: Path,
    mmif_dict: dict,
    mmif_path: str = "",
    run_ner: bool = True,
    run_enrichment: bool = True,
) -> dict | None:
    """Index a video from its MMIF, optionally run NER and entity enrichment."""
    video_id = video_id_from_path(video_path)

    t0 = time.time()
    index_doc = index.build_from_mmif(
        video_id=video_id,
        video_path=str(video_path),
        mmif_dict=mmif_dict,
        source_mmif_path=mmif_path,
    )
    n_segments = len(index_doc.get("segments", []))
    logger.info(f"  Indexed: {n_segments} segments ({time.time() - t0:.1f}s)")

    if run_ner:
        try:
            t0 = time.time()
            n_entities = index.extract_entities_from_transcripts(video_id)
            logger.info(f"  NER: {n_entities} entities ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"  NER failed: {e}")

    if run_enrichment:
        try:
            t0 = time.time()
            grounded = index.enrich_entities(video_id)
            logger.info(f"  Enrichment: {len(grounded)} entities grounded ({time.time() - t0:.1f}s)")
        except Exception as e:
            logger.warning(f"  Enrichment failed: {e}")

    return index_doc


def generate_qa(index_dir: Path, multiple_choice: bool = False):
    """Generate QA evaluation pairs from all indexed videos."""
    qa_dir = Path(__file__).parent / "qa-data"
    sys.path.insert(0, str(qa_dir))
    from generate_index_qa import IndexQAGenerator

    generator = IndexQAGenerator(index_dir=index_dir, multiple_choice=multiple_choice)
    entries = generator.generate_all()

    suffix = "_mc" if multiple_choice else ""
    out_path = qa_dir / "raw" / f"index_qa{suffix}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for entry in entries:
            d = entry.to_dict() if hasattr(entry, "to_dict") else entry
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info(f"QA generation: {len(entries)} entries → {out_path}")
    return entries


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch process videos through CLAMS pipeline, index, and generate QA."
    )
    parser.add_argument(
        "--videos", nargs="+", type=Path,
        help="Specific video files to process. Default: all videos in data/videos/"
    )
    parser.add_argument(
        "--video-dir", type=Path,
        default=Path(__file__).parent / "data" / "videos",
        help="Directory containing videos (default: data/videos/)"
    )
    parser.add_argument(
        "--apps", nargs="+", default=DEFAULT_APPS,
        help=f"CLAMS apps to run in sequence (default: {DEFAULT_APPS})"
    )
    parser.add_argument(
        "--skip-indexed", action="store_true",
        help="Skip videos that already have an index JSON"
    )
    parser.add_argument(
        "--index-only", action="store_true",
        help="Skip CLAMS app execution; just index existing MMIF files"
    )
    # ASR options
    parser.add_argument(
        "--asr", action="store_true",
        help="Run ASR transcription (faster-whisper) after indexing"
    )
    parser.add_argument(
        "--asr-only", action="store_true",
        help="Only run ASR on already-indexed videos (skip CLAMS apps and re-indexing)"
    )
    parser.add_argument(
        "--asr-model", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)"
    )
    # Post-processing options
    parser.add_argument(
        "--no-ner", action="store_true",
        help="Skip spaCy NER extraction"
    )
    parser.add_argument(
        "--no-enrichment", action="store_true",
        help="Skip Wikidata/Wikipedia entity enrichment"
    )
    parser.add_argument(
        "--qa", action="store_true",
        help="Generate free-form QA pairs after indexing"
    )
    parser.add_argument(
        "--qa-mc", action="store_true",
        help="Generate multiple-choice QA pairs after indexing"
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="Per-app timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running"
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent
    output_dir = project_root / "data" / "pipeline_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    index_dir = project_root / "data" / "video_indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir = project_root / "data" / "chroma_db"

    # Discover videos
    if args.videos:
        videos = [v.resolve() for v in args.videos]
    else:
        videos = discover_videos(args.video_dir)

    if not videos:
        logger.error(f"No videos found in {args.video_dir}")
        sys.exit(1)

    # Filter already-indexed (only for CLAMS pipeline, not ASR)
    if args.skip_indexed and not args.asr_only:
        already = set()
        for v in videos:
            vid = video_id_from_path(v)
            if (index_dir / f"{vid}.json").exists():
                already.add(v)
                logger.info(f"Skipping (already indexed): {v.name}")
        videos = [v for v in videos if v not in already]

    logger.info(f"Videos to process: {len(videos)}")
    for v in videos:
        size_mb = v.stat().st_size / 1e6
        logger.info(f"  {v.name} ({size_mb:.0f} MB)")

    if args.dry_run:
        if not args.asr_only:
            logger.info(f"Pipeline: {' → '.join(args.apps)}")
        if args.asr or args.asr_only:
            logger.info(f"ASR: faster-whisper ({args.asr_model})")
        logger.info("Dry run — exiting.")
        return

    # ── ASR-only mode ──
    if args.asr_only:
        asr_results = run_asr_batch(videos, index_dir, args.asr_model)

        # Re-run NER on updated transcripts
        if not args.no_ner:
            logger.info("\nRunning NER on updated transcripts...")
            index = VideoIndex(index_dir=str(index_dir), chroma_dir=str(chroma_dir))
            for v in videos:
                vid = video_id_from_path(v)
                try:
                    t0 = time.time()
                    n = index.extract_entities_from_transcripts(vid)
                    logger.info(f"  {vid}: {n} entities ({time.time() - t0:.1f}s)")
                except Exception as e:
                    logger.warning(f"  {vid}: NER failed: {e}")

        # Re-upsert to ChromaDB with updated summaries
        logger.info("\nUpdating ChromaDB embeddings...")
        index = VideoIndex(index_dir=str(index_dir), chroma_dir=str(chroma_dir))
        for v in videos:
            vid = video_id_from_path(v)
            json_path = index_dir / f"{vid}.json"
            if json_path.exists():
                with open(json_path) as f:
                    doc = json.load(f)
                from utils.video_index import SegmentEntry
                segs = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]
                index._upsert_segments(vid, segs)
                logger.info(f"  {vid}: {len(segs)} segments re-embedded")

        # Summary
        total_words = sum(r.get("total_words", 0) for r in asr_results)
        total_updated = sum(r.get("segments_updated", 0) for r in asr_results)
        logger.info(f"\nASR complete: {total_words} words across {total_updated} segments")

        if args.qa:
            logger.info("\nGenerating free-form QA pairs...")
            generate_qa(index_dir, multiple_choice=False)
        if args.qa_mc:
            logger.info("\nGenerating multiple-choice QA pairs...")
            generate_qa(index_dir, multiple_choice=True)
        return

    # ── Full pipeline mode ──
    executor = UnifiedCLAMSExecutor()
    index = VideoIndex(index_dir=str(index_dir), chroma_dir=str(chroma_dir))

    total_t0 = time.time()
    results_summary = []

    for i, video_path in enumerate(videos, 1):
        video_id = video_id_from_path(video_path)
        logger.info(f"\n[{i}/{len(videos)}] {video_path.name}")

        mmif_dict = None
        mmif_path = output_dir / f"{video_id}.mmif"

        # Step 1: Run CLAMS apps (unless --index-only)
        if not args.index_only:
            mmif_dict = run_pipeline_for_video(
                executor=executor,
                video_path=video_path,
                apps=args.apps,
                app_params={},
                output_dir=output_dir,
                timeout=args.timeout,
            )
        elif mmif_path.exists():
            with open(mmif_path) as f:
                mmif_dict = json.load(f)
        else:
            logger.warning(f"  No MMIF found for --index-only: {mmif_path}")
            results_summary.append({"video": video_path.name, "status": "no_mmif"})
            continue

        if mmif_dict is None:
            results_summary.append({"video": video_path.name, "status": "pipeline_failed"})
            continue

        # Step 2: Index
        index_doc = index_video(
            index=index,
            video_path=video_path,
            mmif_dict=mmif_dict,
            mmif_path=str(mmif_path),
            run_ner=not args.no_ner,
            run_enrichment=not args.no_enrichment,
        )

        n_seg = len(index_doc.get("segments", [])) if index_doc else 0
        results_summary.append({
            "video": video_path.name,
            "status": "ok",
            "segments": n_seg,
        })

    # Summary
    total_elapsed = time.time() - total_t0
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {len(videos)} videos in {total_elapsed:.0f}s")
    for r in results_summary:
        seg_info = f", {r['segments']} segments" if "segments" in r else ""
        logger.info(f"  {r['video']}: {r['status']}{seg_info}")

    # Step 3: ASR (after indexing, so segments exist)
    if args.asr:
        logger.info("\n" + "="*60)
        logger.info("Running ASR transcription...")
        asr_results = run_asr_batch(videos, index_dir, args.asr_model)

        # Re-run NER on transcripts
        if not args.no_ner:
            logger.info("\nRunning NER on ASR transcripts...")
            for v in videos:
                vid = video_id_from_path(v)
                try:
                    n = index.extract_entities_from_transcripts(vid)
                    logger.info(f"  {vid}: {n} entities")
                except Exception as e:
                    logger.warning(f"  {vid}: NER failed: {e}")

        # Re-upsert to ChromaDB with updated summaries
        logger.info("\nUpdating ChromaDB embeddings...")
        for v in videos:
            vid = video_id_from_path(v)
            json_path = index_dir / f"{vid}.json"
            if json_path.exists():
                with open(json_path) as f:
                    doc = json.load(f)
                from utils.video_index import SegmentEntry
                segs = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]
                index._upsert_segments(vid, segs)

    # Step 4: QA generation
    if args.qa:
        logger.info("\nGenerating free-form QA pairs...")
        generate_qa(index_dir, multiple_choice=False)

    if args.qa_mc:
        logger.info("\nGenerating multiple-choice QA pairs...")
        generate_qa(index_dir, multiple_choice=True)


if __name__ == "__main__":
    main()
