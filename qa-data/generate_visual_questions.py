#!/usr/bin/env python3
"""Generate visual-grounded questions requiring actual frame images.

Questions that require SEEING a video frame + reading the ASR transcript
to answer. The visual caption is used to GENERATE questions but is NOT
given to the model being evaluated.

Categories:
- b-roll + transcript: Action footage provides context for narration
- map + transcript: Geographic display relates to the story
- graphic + transcript: Chart/data/diagram supports the discussion
- headline + transcript: On-screen text adds info not in speech

Usage:
    python qa-data/generate_visual_questions.py \
        --output-dir qa-data/benchmark \
        --index-dir data/video_indexes
"""
import argparse
import json
import os
import subprocess
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")


VISUAL_QA_PROMPT = """You are generating questions about broadcast video content that REQUIRE looking at a specific video frame AND reading the speech transcript to answer.

You will be given:
- A description of what a video frame shows (the evaluated model will SEE the actual frame, not this description)
- The speech transcript from the same moment
- Any on-screen text (OCR)

Generate 1-2 questions where:
1. The question CANNOT be answered from the transcript alone (requires seeing the frame)
2. The question CANNOT be answered from the frame alone (requires the transcript context)
3. The answer combines specific visual details with speech context

GOOD question patterns:
- "What type of [vehicles/equipment/building] is shown during the report on [topic from speech]?"
- "What geographic area is highlighted on the map while the reporter discusses [event]?"
- "What data trend does the chart show that relates to the anchor's claim about [topic]?"
- "What action is taking place in the footage while [person] discusses [topic]?"

BAD questions (DO NOT generate):
- Questions answerable from transcript alone ("What did the reporter say about X?")
- Questions answerable from frame alone ("What color is the building?")
- Questions about generic studio settings ("What does the newsroom look like?")
- Questions about show branding or test patterns

Each question must have a specific, verifiable answer grounded in both the visual and speech content.

Return a JSON object with a "questions" key containing an array. Each question object:
- "question": the question text
- "answer": the answer (combining visual + speech evidence)
- "visual_evidence": what in the frame supports the answer (specific detail)
- "speech_evidence": what in the transcript supports the answer (specific quote)
- "reasoning": why both frame AND transcript are needed
"""


def call_ollama(prompt, system=VISUAL_QA_PROMPT):
    """Call Ollama chat API."""
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.7, "num_predict": 2000},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def classify_visual_content(caption):
    """Classify a visual caption into content category."""
    caption_lower = caption.lower()

    if any(x in caption_lower for x in ["map", "globe", "geographic", "world map",
                                         "showing countries", "highlighted region"]):
        return "map"
    if any(x in caption_lower for x in ["chart", "graph", "statistics", "bar chart",
                                         "line graph", "pie chart", "table", "data",
                                         "numbers on screen", "percentage"]):
        return "graphic"
    if any(x in caption_lower for x in ["headline", "banner", "breaking",
                                         "news graphic", "title text"]):
        return "headline"
    if any(x in caption_lower for x in ["tank", "soldier", "military", "flood",
                                         "damage", "protest", "march", "building",
                                         "bridge", "crowd", "vehicle", "plane",
                                         "ship", "fire", "smoke", "factory",
                                         "hospital", "ceremony", "parade", "rally",
                                         "concert", "audience", "aerial", "skyline",
                                         "landscape", "street", "road", "farm",
                                         "construction", "demolition", "wreckage",
                                         "ruins", "debris", "explosion"]):
        return "broll"
    # Person doing something interesting (not just at desk)
    if any(x in caption_lower for x in ["holding", "pointing", "showing",
                                         "demonstrating", "walking", "speaking to",
                                         "shaking hands", "signing", "waving"]):
        return "broll"
    return None


def find_visual_segments(index_data, max_segments=8):
    """Find segments with rich visual content suitable for frame-grounded questions."""
    segs = index_data.get("segments", [])
    scored = []

    for seg in segs:
        caption = seg.get("visual_caption", "") or ""
        asr = seg.get("asr_transcript", "") or ""
        ocr = seg.get("ocr_text", "") or ""

        # Need both visual content and speech
        if len(caption) < 40 or len(asr) < 30:
            continue

        # Skip boring visuals
        if any(skip in caption.lower() for skip in (
            "completely black", "black screen", "test pattern", "color bar",
            "countdown", "no visible content", "currently empty",
            "fuzzymemories", "memories.tv", "macneil/lehrer",
            "newshour", "title card", "purple background",
            "gradient background", "globe graphic", "streaking lights",
        )):
            continue

        # Skip anchor-at-desk (not interesting visually)
        if all(x in caption.lower() for x in ["person", "desk"]) and \
           not any(x in caption.lower() for x in ["map", "chart", "holding", "pointing"]):
            continue

        category = classify_visual_content(caption)
        if not category:
            continue

        # Score by visual richness
        score = 0
        if category == "map":
            score += 5
        elif category == "graphic":
            score += 5
        elif category == "broll":
            score += 4
        elif category == "headline":
            score += 3

        # Boost if OCR adds info not in speech
        if ocr and len(ocr) > 10:
            ocr_words = set(ocr.lower().split())
            asr_words = set(asr.lower().split())
            unique_ratio = len(ocr_words - asr_words) / max(1, len(ocr_words))
            if unique_ratio > 0.3:
                score += 3

        # Boost longer captions (more visual detail)
        if len(caption) > 100:
            score += 1
        if len(caption) > 200:
            score += 1

        scored.append((score, category, seg))

    scored.sort(key=lambda x: -x[0])
    return [(cat, seg) for _, cat, seg in scored[:max_segments]]


def extract_frame(video_path, timestamp_ms, output_path):
    """Extract a single frame from video at the given timestamp."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    timestamp_s = timestamp_ms / 1000.0
    try:
        subprocess.run([
            "ffmpeg", "-y", "-ss", str(timestamp_s),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ], capture_output=True, timeout=30, check=True)
        return os.path.exists(output_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def parse_qa_response(text):
    """Parse LLM JSON response into question objects."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            if "questions" in parsed:
                return parsed["questions"]
            if "question" in parsed:
                return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual-grounded questions requiring frame images")
    parser.add_argument("--output-dir", type=Path, default=Path("qa-data/benchmark"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--max-per-video", type=int, default=3)
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    out_path = args.output_dir / "visual_questions.jsonl"

    videos = sorted(args.index_dir.glob("*.json"))
    print(f"Scanning {len(videos)} video indexes for visual-grounded segments...")

    all_results = []
    category_counts = defaultdict(int)
    q_id = 0

    with open(out_path, "w") as fout:
        for i, f in enumerate(videos):
            index_data = json.load(open(f))
            vid = index_data["video_id"]

            # Get video path from index
            video_path = index_data.get("video_path", "")
            if not video_path:
                continue

            visual_segs = find_visual_segments(index_data, max_segments=6)
            if not visual_segs:
                continue

            print(f"  [{i+1}/{len(videos)}] {vid[-35:]}: {len(visual_segs)} visual segments...",
                  end=" ", flush=True)

            video_questions = 0
            for category, seg in visual_segs:
                if video_questions >= args.max_per_video:
                    break
                if args.max_questions and len(all_results) >= args.max_questions:
                    break

                caption = seg.get("visual_caption", "")
                asr = seg.get("asr_transcript", "")
                ocr = seg.get("ocr_text", "") or ""
                start_ms = seg.get("start_ms", 0)
                end_ms = seg.get("end_ms", 0)
                mid_ms = (start_ms + end_ms) // 2
                seg_id = seg.get("segment_id", f"{vid}_{start_ms}_{end_ms}")

                # Extract frame
                frame_path = str(frames_dir / vid / f"frame_{seg_id}.jpg")
                extracted = extract_frame(video_path, mid_ms, frame_path)
                if not extracted:
                    continue

                # Generate questions
                prompt = f"""Video: {vid}
Visual category: {category}

FRAME DESCRIPTION (what the frame shows — NOT given to the model):
{caption[:500]}

SPEECH TRANSCRIPT (concurrent with this frame):
{asr[:500]}

{"ON-SCREEN TEXT: " + ocr[:200] if ocr else ""}

Generate 1-2 questions that require SEEING this frame AND reading the transcript to answer."""

                try:
                    response = call_ollama(prompt)
                    items = parse_qa_response(response)
                except Exception as e:
                    print(f"LLM error: {e}", file=sys.stderr)
                    continue

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    if not q or not a:
                        continue
                    if video_questions >= args.max_per_video:
                        break

                    q_id += 1
                    result = {
                        "id": f"vq-{q_id:04d}",
                        "format": "visual_grounded",
                        "question": q,
                        "answer": a,
                        "video_id": vid,
                        "frame_path": frame_path,
                        "frame_timestamp_ms": mid_ms,
                        "asr_context": asr[:500],
                        "visual_category": category,
                        "modalities_required": ["visual_frame", "speech"],
                        "grounding": {
                            "visual_reference": item.get("visual_evidence", ""),
                            "speech_excerpt": item.get("speech_evidence", ""),
                            "reasoning": item.get("reasoning", ""),
                        },
                        "source_segment_times": [
                            {"start_ms": start_ms, "end_ms": end_ms}
                        ],
                    }
                    all_results.append(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    video_questions += 1
                    category_counts[category] += 1

                time.sleep(1)

            print(f"{video_questions} questions", flush=True)

            if args.max_questions and len(all_results) >= args.max_questions:
                break

    # Summary
    print(f"\n{'='*60}")
    print(f"Visual-Grounded Questions Summary")
    print(f"{'='*60}")
    print(f"Total: {len(all_results)} questions")
    print(f"\nBy category:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat:15s} {count:4d}")
    print(f"\nWritten to {out_path}")

    # Count unique videos
    unique_vids = len(set(r["video_id"] for r in all_results))
    print(f"Across {unique_vids} videos")

    # Check frame extraction success
    total_frames = sum(1 for r in all_results if os.path.exists(r["frame_path"]))
    print(f"Frames extracted: {total_frames}/{len(all_results)}")


if __name__ == "__main__":
    main()
