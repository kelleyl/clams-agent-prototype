#!/usr/bin/env python3
"""LLM-based coreference resolution and narrative threading for broadcast videos.

For each video, processes the transcript in overlapping windows and asks
a small LLM to:
1. Resolve pronouns/references to named entities
2. Identify story threads (groups of segments about the same topic)
3. Detect narrative structure (intro, detail, resumption, commercial break)

Uses Ollama with a small model (Qwen 3.5 7B) for efficiency.
"""
import json
import os
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:7b")

COREF_PROMPT = """Analyze this broadcast news transcript. Return JSON with:

- "story_thread": short label for the main topic being discussed (e.g., "budget debate", "Middle East peace", "medical errors", "commercial", "program intro")

- "coreferences": list of pronoun/vague references resolved to full names. Each: {"reference": "he/the senator/the bill/etc", "resolves_to": "full name or description", "context": "brief quote"}. Only include confident resolutions.

- "narrative_marker": MUST be one of:
  - "topic_introduction" if the anchor previews upcoming stories ("Tonight we'll look at...", "On the NewsHour tonight...", "Coming up...")
  - "topic_resumption" if returning to a previous topic ("As we reported...", "Returning to...", "We continue with...")
  - "topic_transition" if switching to a new story ("Now turning to...", "In other news...", "Next tonight...")
  - "commercial_adjacent" if near a commercial break (sponsor messages, "we'll be right back", "welcome back")
  - "none" if regular content with no structural marker

- "introduced_topics": list of story labels previewed (ONLY when narrative_marker is "topic_introduction")
"""


def call_ollama(prompt, system=COREF_PROMPT):
    """Call Ollama chat API with JSON format."""
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
            "options": {
                "temperature": 0.3,  # Low temp for factual extraction
                "num_predict": 1000,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def parse_response(text):
    """Parse LLM JSON response."""
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def build_segment_context(seg, speaker_names):
    """Build a text representation of a segment for the LLM."""
    parts = []
    speaker = speaker_names.get(seg.get("primary_speaker", ""), "")
    if speaker:
        parts.append(f"[Speaker: {speaker}]")

    asr = seg.get("asr_transcript", "")
    if asr:
        parts.append(asr[:500])

    # Include entity context so LLM knows who's being discussed
    ents = seg.get("named_entities", [])
    persons = [e["text"] for e in ents if e["type"] == "PERSON" and len(e["text"]) > 3]
    orgs = [e["text"] for e in ents if e["type"] == "ORG" and len(e["text"]) > 3]
    if persons:
        parts.append(f"[People mentioned: {', '.join(persons[:8])}]")
    if orgs:
        parts.append(f"[Organizations: {', '.join(orgs[:5])}]")

    return " ".join(parts)


def process_video(index_path):
    """Run coref + narrative threading on one video."""
    doc = json.load(open(index_path))
    segs = doc.get("segments", [])
    speaker_names = doc.get("speaker_names", {})
    vid = doc["video_id"]

    if len(segs) < 3:
        return doc

    # Process in overlapping windows of 3 segments
    # This gives the LLM enough context to resolve cross-segment references
    window_size = 3
    results = [None] * len(segs)

    for i in range(0, len(segs), 2):  # stride of 2 for overlap
        window = segs[i:i + window_size]
        if not any(s.get("asr_transcript") for s in window):
            continue

        # Build context for this window
        context_parts = []
        for j, seg in enumerate(window):
            ctx = build_segment_context(seg, speaker_names)
            if ctx.strip():
                ts = f"{seg['start_ms'] // 1000 // 60}:{seg['start_ms'] // 1000 % 60:02d}"
                context_parts.append(f"[Segment {i + j + 1}, at {ts}]\n{ctx}")

        if not context_parts:
            continue

        prompt = "\n\n".join(context_parts)

        try:
            response = call_ollama(prompt)
            parsed = parse_response(response)
            if parsed:
                # Apply results to the middle segment (or first if at edge)
                target_idx = min(i + 1, len(segs) - 1) if i > 0 else i
                if results[target_idx] is None:
                    results[target_idx] = parsed
        except Exception as e:
            print(f"  Error at segment {i}: {e}", file=sys.stderr)

        time.sleep(0.3)  # Rate limit

    # Apply results to segments
    total_corefs = 0
    story_threads = defaultdict(list)

    for i, seg in enumerate(segs):
        if results[i]:
            r = results[i]
            seg["story_thread"] = r.get("story_thread", "")
            seg["narrative_marker"] = r.get("narrative_marker", "none")
            seg["introduced_topics"] = r.get("introduced_topics", [])

            corefs = r.get("coreferences", [])
            if isinstance(corefs, list):
                seg["coreferences"] = corefs
                total_corefs += len(corefs)

            thread = r.get("story_thread", "")
            if thread:
                story_threads[thread].append(i)

    # Build story thread summary
    doc["story_threads"] = {
        thread: {
            "segments": indices,
            "span_seconds": round(
                (segs[indices[-1]]["end_ms"] - segs[indices[0]]["start_ms"]) / 1000
            ) if indices else 0,
        }
        for thread, indices in story_threads.items()
    }

    doc["segments"] = segs

    with open(index_path, "w") as f:
        json.dump(doc, f, indent=2)

    return doc, total_corefs, len(story_threads)


def main():
    index_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/video_indexes")

    videos = sorted(index_dir.glob("*.json"))
    print(f"Processing {len(videos)} videos with {MODEL}...")

    total_corefs = 0
    total_threads = 0

    for i, f in enumerate(videos):
        vid = f.stem
        doc = json.load(open(f))
        n_segs = len(doc.get("segments", []))
        asr_segs = sum(1 for s in doc.get("segments", []) if s.get("asr_transcript"))

        if asr_segs < 3:
            print(f"  [{i + 1}/{len(videos)}] {vid[-30:]}: skip (only {asr_segs} ASR segments)")
            continue

        print(f"  [{i + 1}/{len(videos)}] {vid[-30:]}: {n_segs} segs...", end=" ", flush=True)
        try:
            doc, n_corefs, n_threads = process_video(str(f))
            total_corefs += n_corefs
            total_threads += n_threads
            print(f"{n_corefs} corefs, {n_threads} threads")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal: {total_corefs} coreferences, {total_threads} story threads")


if __name__ == "__main__":
    main()
