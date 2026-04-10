#!/usr/bin/env python3
"""Generate CLIP embeddings for each shot's representative frame.

Creates a FAISS index + metadata JSON per video for visual similarity search.

Output:
    data/clip_embeddings/{video_id}.faiss  - FAISS index
    data/clip_embeddings/{video_id}.json   - metadata (shot_id, start_ms, end_ms per embedding)

Usage:
    python scripts/generate_clip_embeddings.py
    python scripts/generate_clip_embeddings.py --videos vid1 vid2
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_clip_embeddings.py
"""
import argparse
import json
import os
import sys
import cv2
import numpy as np
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()

CLIP_MODEL = "clip-ViT-L-14"
BATCH_SIZE = 64


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", help="Specific video IDs")
    parser.add_argument("--index-dir", default="data/video_indexes")
    parser.add_argument("--output-dir", default="data/clip_embeddings")
    parser.add_argument("--model", default=CLIP_MODEL)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = Path(args.index_dir)

    # Load CLIP model
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading CLIP model: {args.model}")
    model = SentenceTransformer(args.model)
    logger.info("Model loaded")

    # Get video list
    if args.videos:
        video_ids = args.videos
    else:
        video_ids = sorted(f.stem for f in index_dir.glob("*.json"))

    import faiss

    count = 0
    for vid in video_ids:
        faiss_path = output_dir / f"{vid}.faiss"
        meta_path = output_dir / f"{vid}.json"

        if faiss_path.exists() and meta_path.exists():
            continue

        idx_path = index_dir / f"{vid}.json"
        if not idx_path.exists():
            continue

        with open(idx_path) as f:
            doc = json.load(f)

        video_path = doc.get("video_path", "")
        if not video_path or not os.path.exists(video_path):
            logger.warning(f"{vid}: no video file")
            continue

        # Get shot boundaries
        if doc.get("schema_version", 1) >= 2:
            shots = doc.get("layers", {}).get("shots", {}).get("items", [])
        else:
            shots = doc.get("segments", [])

        if not shots:
            logger.warning(f"{vid}: no shots")
            continue

        # Extract representative frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
        images = []
        metadata = []

        for shot in shots:
            start_ms = shot.get("start_ms", 0)
            end_ms = shot.get("end_ms", start_ms)
            mid_ms = (start_ms + end_ms) // 2
            frame_num = int(mid_ms / 1000 * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB for CLIP
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                img = Image.fromarray(rgb)
                images.append(img)
                metadata.append({
                    "shot_id": shot.get("id", f"shot_{len(metadata)}"),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "frame_num": frame_num,
                })

        cap.release()

        if not images:
            logger.warning(f"{vid}: no frames extracted")
            continue

        # Encode in batches
        all_embeddings = []
        for i in range(0, len(images), BATCH_SIZE):
            batch = images[i:i + BATCH_SIZE]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(emb)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        index.add(embeddings)

        # Save
        faiss.write_index(index, str(faiss_path))
        with open(meta_path, "w") as f:
            json.dump({"video_id": vid, "model": args.model, "dim": dim,
                        "count": len(metadata), "shots": metadata}, f, indent=2)

        count += 1
        logger.info(f"[{count}] {vid}: {len(metadata)} embeddings ({dim}d)")

    logger.info(f"Done: {count} videos embedded")


if __name__ == "__main__":
    main()
