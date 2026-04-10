#!/usr/bin/env python3
"""Run Whisper ASR on videos and add asr_whisper layer to indexes.

The indexes already have Parakeet ASR in the 'asr' layer. This adds a
'asr_whisper' layer with Whisper large-v3 output for comparison, enabling
model-aware tool selection during GRPO training.

Usage:
    CUDA_VISIBLE_DEVICES=3 python training_data/run_whisper_asr.py \
        --index-dir data/video_indexes \
        --model large-v3 \
        --max-videos 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def run_whisper(video_path, model, language="en"):
    """Run Whisper on a video file, return list of ASR segments."""
    segments, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
        word_timestamps=False,
    )

    items = []
    for i, seg in enumerate(segments):
        items.append({
            "id": f"whisper_{i}",
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "text": seg.text.strip(),
        })

    return items, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--model", type=str, default="large-v3",
                        help="Whisper model size: tiny, base, small, medium, large-v3")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip indexes that already have asr_whisper layer")
    args = parser.parse_args()

    print(f"Loading Whisper {args.model}...")
    whisper = WhisperModel(args.model, device=args.device, compute_type="float16")

    index_files = sorted(args.index_dir.glob("*.json"))
    print(f"Found {len(index_files)} indexes")

    processed = 0
    skipped = 0
    errors = 0

    for idx_path in index_files:
        if args.max_videos and processed >= args.max_videos:
            break

        idx = json.load(open(idx_path))
        video_path = idx.get("video_path", "")

        # Skip if already has whisper ASR
        if args.skip_existing and "asr_whisper" in idx.get("layers", {}):
            skipped += 1
            continue

        if not video_path or not os.path.exists(video_path):
            print(f"  SKIP {idx_path.stem}: video not found at {video_path}")
            skipped += 1
            continue

        print(f"[{processed+1}] {idx_path.stem}")
        t0 = time.time()

        try:
            items, info = run_whisper(video_path, whisper)
            elapsed = time.time() - t0
            duration_s = idx.get("duration_ms", 0) / 1000

            # Add whisper layer to index
            if "layers" not in idx:
                idx["layers"] = {}
            idx["layers"]["asr_whisper"] = {
                "source": "faster-whisper",
                "model": args.model,
                "language": info.language if hasattr(info, "language") else "en",
                "items": items,
            }

            # Write back
            with open(idx_path, "w") as f:
                json.dump(idx, f, ensure_ascii=False, indent=2)

            ratio = elapsed / max(1, duration_s)
            print(f"  {len(items)} segments, {elapsed:.1f}s ({ratio:.2f}x realtime)")
            processed += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
