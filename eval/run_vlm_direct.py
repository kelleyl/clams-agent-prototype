#!/usr/bin/env python3
"""Answer QA benchmark questions using a VLM directly on video frames.

Baseline comparison: instead of using our structured index, feed raw
video frames (+ optionally ASR transcript) directly to a VLM.

Conditions:
- frames_only_1:    1 midpoint frame, no transcript
- frames_asr_1:     1 midpoint frame + ASR transcript
- frames_only_5:    5 evenly sampled frames, no transcript
- frames_asr_5:     5 frames + ASR transcript
- frames_asr_10:    10 frames + ASR transcript

Usage:
    python eval/run_vlm_direct.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --condition frames_asr_5 \
        --output eval/results/vlm_direct_frames_asr_5.jsonl \
        --model qwen2.5-vl:7b \
        --index-dir data/video_indexes
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import requests
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8200")
API_BACKEND = os.environ.get("VLM_API_BACKEND", "ollama")  # "ollama" or "vllm"

CONDITIONS = {
    "frames_only_1":  {"n_frames": 1,  "include_asr": False},
    "frames_asr_1":   {"n_frames": 1,  "include_asr": True},
    "frames_only_5":  {"n_frames": 5,  "include_asr": False},
    "frames_asr_5":   {"n_frames": 5,  "include_asr": True},
    "frames_asr_10":  {"n_frames": 10, "include_asr": True},
}


def extract_frames(video_path, start_ms, end_ms, n_frames, tmp_dir):
    """Extract n_frames evenly spaced between start_ms and end_ms."""
    duration_s = (end_ms - start_ms) / 1000.0
    if duration_s <= 0:
        return []

    frames = []
    for i in range(n_frames):
        if n_frames == 1:
            t = (start_ms + end_ms) / 2 / 1000.0
        else:
            t = start_ms / 1000.0 + (i / (n_frames - 1)) * duration_s

        out_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(t),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ], capture_output=True, timeout=15, check=True)
            if os.path.exists(out_path):
                frames.append(out_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    return frames


def encode_image_base64(path):
    """Read image file and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def answer_with_vlm(question, frame_paths, asr_text, model, mc_options=None):
    """Send frames + question to VLM via Ollama or vLLM."""
    # Build prompt
    prompt_parts = []
    if asr_text:
        prompt_parts.append(f"Speech transcript from this video segment:\n{asr_text[:800]}")
    prompt_parts.append(f"\nQuestion: {question}")

    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in mc_options.items())
        prompt_parts.append(f"\nOptions:\n{opts}")
        prompt_parts.append("\nReturn a JSON object with \"answer\" (the letter A/B/C/D), "
                          "\"reasoning\" (brief explanation).")
    else:
        prompt_parts.append("\nAnswer based on what you see in the frame(s) and any transcript provided. "
                          "Return a JSON object with \"answer\" (your answer) and \"reasoning\" (brief explanation).")

    prompt = "\n".join(prompt_parts)

    try:
        if API_BACKEND == "vllm":
            # vLLM OpenAI-compatible vision API
            content_parts = []
            for p in frame_paths:
                b64 = encode_image_base64(p)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            content_parts.append({"type": "text", "text": prompt})

            resp = requests.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": content_parts}],
                    "temperature": 0.3,
                    "max_tokens": 300,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        else:
            # Ollama API
            images = [encode_image_base64(p) for p in frame_paths]
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt, "images": images}
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3, "num_predict": 300},
                },
                timeout=120,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"answer": content, "reasoning": "parse_error"}
    except Exception as e:
        return {"answer": f"Error: {e}", "reasoning": "api_error"}


def get_segment_asr(video_id, start_ms, end_ms, index_data):
    """Get ASR transcript for a time range from the index."""
    parts = []
    for seg in index_data.get("segments", []):
        seg_start = seg.get("start_ms", 0)
        seg_end = seg.get("end_ms", 0)
        # Check overlap
        if seg_end > start_ms and seg_start < end_ms:
            asr = seg.get("asr_transcript", "")
            if asr:
                parts.append(asr)
    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--condition", type=str, required=True,
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen2.5-vl:7b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--retrieval", choices=["oracle", "uniform"], default="oracle",
                        help="oracle=frames from ground-truth segment, uniform=evenly spaced across full video")
    parser.add_argument("--api", choices=["ollama", "vllm"], default="ollama",
                        help="API backend: ollama or vllm (OpenAI-compatible)")
    args = parser.parse_args()

    global API_BACKEND
    API_BACKEND = args.api

    condition = CONDITIONS[args.condition]
    n_frames = condition["n_frames"]
    include_asr = condition["include_asr"]

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Pre-load indexes (for video paths and ASR)
    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"VLM Direct Evaluation: {args.condition} (retrieval={args.retrieval})")
    print(f"  Frames: {n_frames}, ASR: {include_asr}, Retrieval: {args.retrieval}")
    print(f"  Model: {args.model}")
    print(f"  Questions: {len(questions)}")

    stats = {"total": 0, "answered": 0, "frame_errors": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            question_text = q["question"]
            fmt = q.get("format", "free_text")

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                continue

            video_path = idx.get("video_path", "")
            if not video_path or not os.path.exists(video_path):
                continue

            # Get time range based on retrieval mode
            if args.retrieval == "oracle":
                # Oracle: use ground-truth segment times
                source_times = q.get("source_segment_times", [])
                if source_times:
                    start_ms = min(t["start_ms"] for t in source_times)
                    end_ms = max(t["end_ms"] for t in source_times)
                else:
                    start_ms = 0
                    end_ms = idx.get("duration_ms", 60000)
            else:
                # Uniform: sample across the full video
                start_ms = 0
                end_ms = idx.get("duration_ms", 60000)

            # Extract frames
            with tempfile.TemporaryDirectory() as tmp_dir:
                frame_paths = extract_frames(video_path, start_ms, end_ms, n_frames, tmp_dir)

                if not frame_paths:
                    stats["frame_errors"] += 1
                    continue

                # Get ASR based on retrieval mode
                asr_text = ""
                if include_asr:
                    if args.retrieval == "oracle":
                        asr_text = get_segment_asr(video_id, start_ms, end_ms, idx)
                    else:
                        # Uniform: give full video transcript (truncated)
                        all_asr = []
                        for seg in idx.get("segments", []):
                            asr = seg.get("asr_transcript", "")
                            if asr:
                                all_asr.append(asr)
                        asr_text = " ".join(all_asr)[:2000]  # Truncate to fit context

                # Query VLM
                prediction = answer_with_vlm(
                    question_text, frame_paths, asr_text, args.model, mc_options
                )

            predicted_answer = prediction.get("answer", "")
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                for letter in "ABCD":
                    if predicted_answer.strip().upper().startswith(letter):
                        predicted_answer = letter
                        break

            stats["total"] += 1
            stats["answered"] += 1

            result = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted_answer,
                "condition": args.condition,
                "retrieval": args.retrieval,
                "n_frames": len(frame_paths),
                "include_asr": include_asr,
                "reasoning": prediction.get("reasoning", ""),
                "reasoning_type": q.get("reasoning_type", ""),
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(questions)}] answered", flush=True)

            time.sleep(0.5)

    print(f"\nDone. {stats['answered']} answered, {stats['frame_errors']} frame extraction errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
