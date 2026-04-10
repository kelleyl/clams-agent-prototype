#!/usr/bin/env python3
"""Serve the multi-layer video index viewer with video streaming and LLM proxy.

Usage:
    # Set up SSH tunnel for vLLM first:
    ssh -L 8101:localhost:8101 aristotle

    # Run the server:
    python serve_viewer.py

    # Open http://localhost:8765/layer_viewer.html

Environment variables:
    VLLM_URL       - vLLM API base (default: http://localhost:8101)
    VLLM_MODEL     - Model name (default: Qwen/Qwen3.5-9B)
    VIDEO_HOST     - SSH host for video streaming (default: aristotle)
"""

import json
import os
import subprocess
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8101")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
VIDEO_HOST = os.environ.get("VIDEO_HOST", "aristotle")

# ============================================================
# Static file serving
# ============================================================

@app.route("/")
def index():
    return send_from_directory(".", "layer_viewer.html")


@app.route("/data/video_indexes/")
def list_indexes():
    """Serve directory listing of video indexes as HTML with links."""
    idx_dir = Path("data/video_indexes")
    files = sorted(f.name for f in idx_dir.glob("*.json"))
    links = "\n".join(f'<a href="{f}">{f}</a>' for f in files)
    return f"<html><body>{links}</body></html>"


@app.route("/api/save_review", methods=["POST"])
def save_review():
    """Save QA review decisions."""
    review_dir = Path("qa-data/reviews")
    review_dir.mkdir(parents=True, exist_ok=True)
    data = request.get_json()
    with open(review_dir / "v2_review.json", "w") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status": "ok", "count": len(data)})


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(".", path)


# ============================================================
# Video streaming via SSH
# ============================================================

@app.route("/api/video/<video_id>")
def stream_video(video_id):
    """Stream video file from remote host via SSH.

    Looks up the video_path from the index JSON, then streams it from
    the remote host. Supports Range requests for seeking.
    """
    index_path = Path("data/video_indexes") / f"{video_id}.json"
    if not index_path.exists():
        return jsonify({"error": "Video index not found"}), 404

    with open(index_path) as f:
        doc = json.load(f)

    video_path = doc.get("video_path", "")
    if not video_path:
        return jsonify({"error": "No video path in index"}), 404

    # Get file size first
    try:
        result = subprocess.run(
            ["ssh", VIDEO_HOST, f"stat -c %s '{video_path}' 2>/dev/null || stat -f %z '{video_path}'"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return jsonify({"error": f"Video file not found on {VIDEO_HOST}: {video_path}"}), 404
        file_size = int(result.stdout.strip())
    except Exception as e:
        return jsonify({"error": f"Could not stat video: {e}"}), 500

    # Handle Range requests for seeking
    range_header = request.headers.get("Range")
    start = 0
    end = file_size - 1

    if range_header:
        parts = range_header.replace("bytes=", "").split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1

    length = end - start + 1

    def generate():
        try:
            # Use dd via ssh for byte-range streaming
            cmd = ["ssh", VIDEO_HOST, f"dd if='{video_path}' bs=65536 skip={start // 65536} count={(length // 65536) + 2} 2>/dev/null"]
            if start % 65536 == 0 and length > 65536:
                # Aligned read — use dd directly
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=65536)
            else:
                # Use head/tail for precise byte ranges
                cmd = ["ssh", VIDEO_HOST, f"dd if='{video_path}' bs=1 skip={start} count={length} 2>/dev/null"]
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=65536)

            bytes_sent = 0
            while bytes_sent < length:
                chunk_size = min(65536, length - bytes_sent)
                chunk = proc.stdout.read(chunk_size)
                if not chunk:
                    break
                bytes_sent += len(chunk)
                yield chunk
            proc.stdout.close()
            proc.wait()
        except Exception as e:
            logger.error(f"Video stream error: {e}")

    headers = {
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
        "Content-Range": f"bytes {start}-{end}/{file_size}",
    }

    status = 206 if range_header else 200
    return Response(generate(), status=status, headers=headers)


# ============================================================
# LLM Summarization proxy
# ============================================================

@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Summarize a time range of a video using vLLM.

    Expects JSON body:
    {
        "video_id": "...",
        "start_ms": 0,
        "end_ms": 100000,
        "layers": { ... gathered layer data ... },
        "prompt": "optional custom prompt"
    }
    """
    import requests as req

    data = request.get_json()
    video_id = data.get("video_id", "")
    start_ms = data.get("start_ms", 0)
    end_ms = data.get("end_ms", 0)
    layer_context = data.get("context", "")
    custom_prompt = data.get("prompt", "")

    if not layer_context:
        return jsonify({"error": "No context provided"}), 400

    system_prompt = """You are an expert at analyzing broadcast video content. You will be given structured evidence from multiple analysis layers of a video segment. Provide a clear, informative summary that synthesizes the information across modalities (speech, visual, on-screen text, speakers, entities)."""

    user_prompt = custom_prompt or f"""Summarize what is happening in this section of the video ({_format_time(start_ms)} - {_format_time(end_ms)}):

{layer_context}

Provide a concise but comprehensive summary that:
1. Describes the main topic/event being covered
2. Notes who is speaking and what they're saying
3. Describes relevant visuals and on-screen text
4. Highlights any cross-modal connections (e.g., chyron identifying a speaker, visual supporting narration)"""

    try:
        resp = req.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]

        # Strip thinking tags if present (Qwen3.5 thinking mode)
        if "<think>" in content:
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        return jsonify({"summary": content})

    except req.ConnectionError:
        return jsonify({"error": "Cannot connect to vLLM. Set up SSH tunnel: ssh -L 8101:localhost:8101 aristotle"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# CLIP similarity search
# ============================================================

@app.route("/api/similar/<video_id>/<shot_idx>")
def find_similar(video_id, shot_idx):
    """Find visually similar shots using CLIP embeddings.

    Returns top-k similar shots from the same video (or across all videos).
    """
    import numpy as np

    shot_idx = int(shot_idx)
    top_k = int(request.args.get("k", 20))
    cross_video = request.args.get("cross_video", "false") == "true"

    embed_dir = Path("data/clip_embeddings")
    faiss_path = embed_dir / f"{video_id}.faiss"
    meta_path = embed_dir / f"{video_id}.json"

    if not faiss_path.exists() or not meta_path.exists():
        return jsonify({"error": f"No CLIP embeddings for {video_id}"}), 404

    try:
        import faiss
    except ImportError:
        return jsonify({"error": "faiss not installed"}), 500

    # Load query video's index
    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)

    if shot_idx >= meta["count"]:
        return jsonify({"error": f"Shot index {shot_idx} out of range (max {meta['count']-1})"}), 400

    # Get query embedding
    query = index.reconstruct(shot_idx).reshape(1, -1)

    results = []

    if cross_video:
        # Search across all videos
        for emb_file in sorted(embed_dir.glob("*.faiss")):
            vid = emb_file.stem
            vid_index = faiss.read_index(str(emb_file))
            with open(embed_dir / f"{vid}.json") as f:
                vid_meta = json.load(f)

            k = min(top_k, vid_index.ntotal)
            scores, indices = vid_index.search(query, k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                shot = vid_meta["shots"][idx]
                results.append({
                    "video_id": vid,
                    "shot_id": shot["shot_id"],
                    "shot_idx": int(idx),
                    "start_ms": shot["start_ms"],
                    "end_ms": shot["end_ms"],
                    "score": float(score),
                })
        results.sort(key=lambda x: -x["score"])
        results = results[:top_k]
    else:
        # Same video only
        k = min(top_k + 1, index.ntotal)
        scores, indices = index.search(query, k)

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx == shot_idx:
                continue
            shot = meta["shots"][idx]
            results.append({
                "video_id": video_id,
                "shot_id": shot["shot_id"],
                "shot_idx": int(idx),
                "start_ms": shot["start_ms"],
                "end_ms": shot["end_ms"],
                "score": float(score),
            })

    return jsonify({"query_shot": meta["shots"][shot_idx], "results": results[:top_k]})


@app.route("/api/text_search_clip/<video_id>")
def text_search_clip(video_id):
    """Search shots by text query using CLIP text-image matching."""
    import numpy as np

    query_text = request.args.get("q", "")
    top_k = int(request.args.get("k", 20))

    if not query_text:
        return jsonify({"error": "No query text"}), 400

    embed_dir = Path("data/clip_embeddings")
    faiss_path = embed_dir / f"{video_id}.faiss"
    meta_path = embed_dir / f"{video_id}.json"

    if not faiss_path.exists():
        return jsonify({"error": f"No CLIP embeddings for {video_id}"}), 404

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return jsonify({"error": "faiss or sentence-transformers not installed"}), 500

    # Load model (cached after first call)
    if not hasattr(text_search_clip, '_model'):
        text_search_clip._model = SentenceTransformer("clip-ViT-L-14")

    index = faiss.read_index(str(faiss_path))
    with open(meta_path) as f:
        meta = json.load(f)

    # Encode text query
    text_emb = text_search_clip._model.encode([query_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(text_emb)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(text_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        shot = meta["shots"][idx]
        results.append({
            "video_id": video_id,
            "shot_id": shot["shot_id"],
            "shot_idx": int(idx),
            "start_ms": shot["start_ms"],
            "end_ms": shot["end_ms"],
            "score": float(score),
        })

    return jsonify({"query": query_text, "results": results})


@app.route("/api/clip_status")
def clip_status():
    """Check which videos have CLIP embeddings."""
    embed_dir = Path("data/clip_embeddings")
    if not embed_dir.exists():
        return jsonify({"count": 0, "videos": []})

    videos = []
    for f in sorted(embed_dir.glob("*.json")):
        with open(f) as fh:
            meta = json.load(fh)
        videos.append({"video_id": meta["video_id"], "count": meta["count"]})

    return jsonify({"count": len(videos), "videos": videos})


def _format_time(ms):
    s = ms // 1000
    m = s // 60
    h = m // 60
    return f"{h}:{m%60:02d}:{s%60:02d}" if h else f"{m}:{s%60:02d}"


if __name__ == "__main__":
    print("Layer Viewer Server")
    print(f"  vLLM: {VLLM_URL} (model: {VLLM_MODEL})")
    print(f"  Video host: {VIDEO_HOST}")
    print(f"  Open: http://localhost:8765/layer_viewer.html")
    print()
    print("Ensure SSH tunnel is running:")
    print(f"  ssh -L 8101:localhost:8101 {VIDEO_HOST}")
    print()
    app.run(host="0.0.0.0", port=8769, debug=True)
