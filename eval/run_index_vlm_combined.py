#!/usr/bin/env python3
"""Combined Index + VLM evaluation: structured retrieval + visual verification.

Mirrors the Microsoft Deep Video Discovery approach:
1. Search the structured index to find relevant segments (text-based)
2. Extract frames from top candidate segments
3. Send frames + index context to a VLM for visual verification + answer

This combines the grounding advantage of the index (segment citations,
quoted evidence) with the visual accuracy of a VLM (can read chyrons,
interpret b-roll, understand graphics directly from frames).

Usage:
    python eval/run_index_vlm_combined.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --output eval/results/combined_index_vlm.jsonl \
        --text-model Qwen/Qwen3.5-9B \
        --vlm-model Qwen/Qwen3-VL-8B-Instruct
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
TEXT_VLLM_URL = os.environ.get("TEXT_VLLM_URL", "http://localhost:8100")
VLM_VLLM_URL = os.environ.get("VLM_VLLM_URL", "http://localhost:8200")


def search_index(question, index_data, top_k=5):
    """Find relevant segments via keyword matching."""
    segments = index_data.get("segments", [])
    question_words = set(question.lower().split())
    stops = {"the", "a", "an", "is", "was", "what", "how", "did", "does",
             "who", "where", "when", "why", "which", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this",
             "based", "according", "between", "during", "about"}
    question_words -= stops

    scored = []
    for seg in segments:
        text = " ".join([
            seg.get("asr_transcript", ""),
            seg.get("ocr_text", "") or "",
            seg.get("visual_caption", "") or "",
            " ".join(e["text"] for e in seg.get("named_entities", [])),
        ]).lower()
        text_words = set(text.split())
        overlap = question_words & text_words
        if overlap:
            scored.append((len(overlap), seg))

    scored.sort(key=lambda x: -x[0])
    return [seg for _, seg in scored[:top_k]]


def format_index_context(segments, index_data):
    """Format retrieved segments as text context."""
    speaker_names = index_data.get("speaker_names", {})
    parts = []
    for seg in segments:
        start_s = seg.get("start_ms", 0) / 1000
        end_s = seg.get("end_ms", 0) / 1000
        lines = [f"[Segment {start_s:.1f}s - {end_s:.1f}s]"]

        speaker = speaker_names.get(seg.get("primary_speaker", ""), "")
        if speaker:
            lines.append(f"Speaker: {speaker}")
        if seg.get("asr_transcript"):
            lines.append(f"Speech: {seg['asr_transcript'][:400]}")
        if seg.get("ocr_text") and not (seg["ocr_text"] or "").startswith("The image"):
            lines.append(f"On-screen text: {(seg['ocr_text'] or '')[:200]}")
        ents = seg.get("named_entities", [])
        if ents:
            lines.append(f"Entities: {', '.join(e['text'] for e in ents[:10])}")
        grounded = seg.get("grounded_entities", {})
        if grounded:
            gr_strs = [f"{k}: {v[:80]}" for k, v in grounded.items() if v][:3]
            if gr_strs:
                lines.append(f"Entity knowledge: {'; '.join(gr_strs)}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def extract_frame(video_path, timestamp_s, tmp_dir):
    """Extract a single frame."""
    out_path = os.path.join(tmp_dir, f"frame_{timestamp_s:.0f}.jpg")
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(timestamp_s),
            "-i", video_path, "-frames:v", "1", "-q:v", "2",
            "-vf", "scale=640:-1", out_path,
        ], capture_output=True, timeout=15, check=True)
        if os.path.exists(out_path):
            return out_path
    except Exception:
        pass
    return None


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


COMBINED_PROMPT = """You are answering a question about a broadcast video. You have TWO sources of evidence:

1. STRUCTURED INDEX DATA (text extracted by specialized tools):
{index_context}

2. VIDEO FRAMES from the most relevant segments (you can see these directly).

Question: {question}
{mc_section}
Use BOTH the structured text data AND what you see in the frames to answer. The text data provides speech transcripts, entity names, and on-screen text extracted by OCR. The frames let you verify visual details directly.

Return a JSON object with:
- "answer": your answer{mc_note}
- "cited_segments": list of segment time ranges you used
- "speech_evidence": relevant speech quote (or null)
- "visual_evidence": what you see in the frames that supports the answer (or null)
- "reasoning": how you combined text and visual evidence"""


def answer_combined(question, index_context, frame_paths, mc_options=None):
    """Send index context + frames to VLM."""
    mc_section = ""
    mc_note = ""
    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in mc_options.items())
        mc_section = f"\nOptions:\n{opts}\n"
        mc_note = " (the letter A/B/C/D)"

    prompt = COMBINED_PROMPT.format(
        index_context=index_context,
        question=question,
        mc_section=mc_section,
        mc_note=mc_note,
    )

    # Build VLM request with images
    content_parts = []
    for p in frame_paths:
        b64 = encode_image_base64(p)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content_parts.append({"type": "text", "text": prompt})

    try:
        resp = requests.post(
            f"{VLM_VLLM_URL}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "messages": [{"role": "user", "content": content_parts}],
                "temperature": 0.3,
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"answer": content, "cited_segments": [], "reasoning": "parse_error"}
    except Exception as e:
        return {"answer": f"Error: {e}", "cited_segments": [], "reasoning": "api_error"}


def verify_grounding(prediction, index_data):
    """Verify cited evidence exists in the index."""
    speech_ev = (prediction.get("speech_evidence") or "")
    visual_ev = (prediction.get("visual_evidence") or "")

    segments = index_data.get("segments", [])
    verification = {
        "speech_verified": False,
        "visual_from_frame": bool(visual_ev),
        "any_verified": False,
        "multi_modal_citation": False,
    }

    for seg in segments:
        asr = (seg.get("asr_transcript") or "").lower()
        if speech_ev and speech_ev[:40].lower() in asr:
            verification["speech_verified"] = True

    verification["any_verified"] = verification["speech_verified"] or verification["visual_from_frame"]
    verification["multi_modal_citation"] = verification["speech_verified"] and verification["visual_from_frame"]

    return verification


def main():
    parser = argparse.ArgumentParser(
        description="Combined Index + VLM evaluation")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--frames-per-segment", type=int, default=1,
                        help="Frames to extract per retrieved segment")
    parser.add_argument("--top-k-segments", type=int, default=3,
                        help="Number of segments to retrieve and visualize")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Combined Index + VLM Evaluation")
    print(f"  Questions: {len(questions)}")
    print(f"  Top-K segments: {args.top_k_segments}")
    print(f"  Frames per segment: {args.frames_per_segment}")
    print(f"  VLM: Qwen3-VL-8B-Instruct (vLLM @ {VLM_VLLM_URL})")

    mc_count = sum(1 for q in questions if q.get("format") == "multiple_choice")
    ft_count = sum(1 for q in questions if q.get("format") == "free_text")
    print(f"  Format: {mc_count} MC + {ft_count} free-text")

    stats = {"total": 0, "answered": 0, "grounding_verified": 0,
             "multi_modal": 0, "frame_errors": 0}

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
                # Fall back to text-only if no video file
                relevant = search_index(question_text, idx, args.top_k_segments)
                context = format_index_context(relevant, idx)
                # Can't do combined without video, skip
                continue

            # Step 1: Search index for relevant segments
            relevant = search_index(question_text, idx, args.top_k_segments)
            context = format_index_context(relevant, idx)

            # Step 2: Extract frames from top segments
            with tempfile.TemporaryDirectory() as tmp_dir:
                frame_paths = []
                for seg in relevant[:args.top_k_segments]:
                    mid_s = (seg.get("start_ms", 0) + seg.get("end_ms", 0)) / 2000
                    frame = extract_frame(video_path, mid_s, tmp_dir)
                    if frame:
                        frame_paths.append(frame)

                if not frame_paths:
                    stats["frame_errors"] += 1
                    continue

                # Step 3: Send index context + frames to VLM
                prediction = answer_combined(
                    question_text, context, frame_paths, mc_options
                )

            predicted_answer = prediction.get("answer", "")
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                for letter in "ABCD":
                    if predicted_answer.strip().upper().startswith(letter):
                        predicted_answer = letter
                        break

            verification = verify_grounding(prediction, idx)

            stats["total"] += 1
            stats["answered"] += 1
            if verification["any_verified"]:
                stats["grounding_verified"] += 1
            if verification["multi_modal_citation"]:
                stats["multi_modal"] += 1

            result = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted_answer,
                "segments_retrieved": len(relevant),
                "frames_used": len(frame_paths),
                "reasoning_type": q.get("reasoning_type", ""),
                "grounding": {
                    "cited_segments": prediction.get("cited_segments", []),
                    "speech_evidence": prediction.get("speech_evidence"),
                    "visual_evidence": prediction.get("visual_evidence"),
                    "reasoning": prediction.get("reasoning", ""),
                },
                "verification": verification,
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                verified_pct = stats["grounding_verified"] / max(1, stats["answered"])
                print(f"  [{i+1}/{len(questions)}] answered, "
                      f"grounding: {verified_pct:.0%}", flush=True)

            time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Combined Index + VLM Results")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"Frame errors: {stats['frame_errors']}")
    print(f"Grounding verified: {stats['grounding_verified']}/{stats['answered']} "
          f"({stats['grounding_verified']/max(1,stats['answered']):.0%})")
    print(f"Multi-modal citations: {stats['multi_modal']}/{stats['answered']} "
          f"({stats['multi_modal']/max(1,stats['answered']):.0%})")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
