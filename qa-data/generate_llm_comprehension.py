#!/usr/bin/env python3
"""Generate cross-modal comprehension questions using an LLM.

For each video, identifies segments rich in multi-modal evidence
(ASR + visual + OCR + entities + speaker) and asks an LLM to generate
questions that require combining information across modalities.

Uses Ollama (Qwen 3.5 or Qwen 3) on aristotle.
"""
import json
import sys
import os
import time
import requests
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:14b")

SYSTEM_PROMPT = """You are an expert at generating video comprehension questions for evaluating AI systems that process broadcast news videos.

You will be given structured evidence from a video segment including:
- Speech transcript (what was said)
- Visual description (what was shown on screen)
- On-screen text/graphics (OCR)
- Named entities mentioned
- Speaker identity (if known)
- Relations extracted (who did what)

Generate questions that REQUIRE combining information from MULTIPLE modalities to answer correctly. The questions should test genuine comprehension, not simple retrieval.

GOOD question patterns (prioritize these):

Pattern 1 - ON-SCREEN DATA + SPEECH CONTEXT: Information appears on screen (chyron with a person's name/title, a headline graphic, a map label, a data figure) that is NOT spoken aloud but is meaningful in context of what IS being said.
- "What organization does Dr. Krim represent according to her on-screen title?" (name chyron + speech context)
- "What headline was displayed while the anchor discussed currency markets?" (headline graphic adds new info)
- "What statistics were shown on the chart during the healthcare discussion?" (data on screen not in speech)

Pattern 2 - B-ROLL FOOTAGE + NARRATION: Visual footage shows action, locations, or events (soldiers marching, flood damage, a protest, a building) while narration provides context about what/where/why.
- "What was shown on screen while the reporter described the Soviet advance into Kabul?" (visual shows tanks, speech names the city)
- "What damage was visible during the report on Hurricane Katrina?" (visual shows flooding, speech provides context)

Pattern 3 - SPEAKER IDENTIFICATION: A chyron or on-screen label identifies WHO is speaking, while the speech contains their actual argument or claim.
- "What position did the person identified as 'Anti-AIDS Citizens Group' representative take on the issue?"

Pattern 4 - EXTERNAL KNOWLEDGE + VIDEO CONTENT: When entity knowledge from Wikidata/Wikipedia is provided, generate questions that require combining that external knowledge with what was said or shown in the video.
- "Given that [person] is described as [Wikidata description], how does their professional background relate to the topic they're discussing?"
- "The broadcast mentions [organization] — based on its description as [Wikidata info], why would they be relevant to this story?"
- "What expertise does [speaker] bring to this discussion, based on their identified role and the topic being covered?"

BAD questions (DO NOT generate these):
- "Who is mentioned in the video?" (just entity listing)
- "What happens at 5:30?" (timestamp grounded)
- "What is shown on screen?" (single modality)
- "How many times is X mentioned?" (counting query)
- Questions about "test patterns", "color bars", "black screens", "countdown sequences", show branding, or title cards
- Questions where the visual is just "a person talking at a desk" or "news studio" with no meaningful visual information
- Questions where the visual/OCR is unrelated to the speech — do NOT force a cross-modal connection where none exists
- Questions whose answer is "does not directly relate" or "is unrelated"
- Questions about show branding ("MacNeil/Lehrer", "NewsHour" logo) — these are not content

KEY PRINCIPLE: The best questions are ones where REMOVING either the visual/OCR OR the speech would make the question unanswerable. If you can answer the question from speech alone, it's not cross-modal. If you can answer from the visual alone, it's not cross-modal.

Examples of truly cross-modal information:
- A chyron says "CLAUDIE LINDSEY, Anti-AIDS Citizens Group" — you need the chyron to know WHO is speaking, and the speech to know WHAT they said
- A headline graphic says "EGYPTIAN CABINET RESIGNS" while the anchor discusses a different story — the graphic provides separate news information
- B-roll shows flooding while narration says "the worst damage was in the French Quarter" — visual shows WHAT, speech says WHERE

Each question must:
1. Require at least 2 modalities (speech+visual, speech+OCR, visual+entities, etc.)
2. Have a specific, verifiable answer grounded in the evidence
3. Test understanding, not just recall
4. Be naturally phrased as a human would ask

CRITICAL: For each question you MUST provide grounding — exact excerpts from the evidence that support the question and answer. If you cannot point to specific excerpts, do not generate the question.

Return a JSON object with a "questions" key containing an array. Each question object must have:
- "question": the question text
- "answer": the answer grounded in the evidence
- "modalities_required": list of modalities needed (e.g., ["speech", "visual", "entities"])
- "reasoning_type": one of "factual", "inferential", "comparative", "causal", "cross-modal"
- "grounding": object with:
  - "speech_excerpt": the exact quote from the speech transcript that supports the answer (copy verbatim, or null if speech not used)
  - "visual_reference": the relevant part of the visual description (copy verbatim, or null if visual not used)
  - "ocr_reference": the relevant on-screen text (copy verbatim, or null if OCR not used)
  - "reasoning": brief explanation of why multiple modalities are needed to answer this question
"""


def _overlaps(a_start, a_end, b_start, b_end):
    """Check if two time ranges overlap."""
    return a_start < b_end and a_end > b_start


def _gather_window_data(layers, start_ms, end_ms):
    """Gather all layer data overlapping a time window.

    Returns a dict with aggregated text/entities/relations for scoring.
    """
    asr_texts = []
    captions = []
    ocr_texts = []
    entities = []
    relations = []
    speakers = []

    for name, layer in layers.items():
        for item in layer.get("items", []):
            if not _overlaps(item.get("start_ms", 0), item.get("end_ms", 0), start_ms, end_ms):
                continue

            if name == "asr":
                asr_texts.append(item.get("text", ""))
            elif name == "visual_captions":
                captions.append(item.get("text", ""))
            elif name == "ocr":
                ocr_texts.append(item.get("text", ""))
            elif name == "entities":
                entities.append(item)
            elif name == "relations":
                relations.append(item)
            elif name == "speakers":
                speakers.append(item)

    return {
        "start_ms": start_ms,
        "end_ms": end_ms,
        "asr": " ".join(asr_texts),
        "captions": " ".join(captions),
        "ocr": " ".join(ocr_texts),
        "entities": entities,
        "relations": relations,
        "speakers": speakers,
    }


def _score_window(data):
    """Score a time window by multi-modal richness. Returns (score, data) or None."""
    score = 0
    asr = data.get("asr", "")
    caption = data.get("captions", "")
    ocr = data.get("ocr", "")
    ents = data.get("entities", [])
    rels = data.get("relations", [])
    speakers = data.get("speakers", [])

    # Score based on modality richness
    if len(asr) > 50:
        score += 2
    if (caption and len(caption) > 30
            and not any(skip in caption.lower() for skip in (
                "completely black", "black screen", "test pattern",
                "color bar", "countdown", "displaying the number",
                "currently empty", "no visible content",
                "scenic view of a mountain", "fuel gauge",
                "macneil/lehrer", "newshour", "title card",
                "purple background", "gradient background",
                "globe graphic", "streaking lights",
                "fuzzymemories",
            ))):
        score += 2
    if (ocr and len(ocr) > 10
            and not ocr.startswith(("The image", "The video", "None", "00:"))
            and not all(c in "0123456789:.- \n" for c in ocr.strip())):
        score += 3  # OCR is especially valuable for cross-modal
    if len(ents) > 2:
        score += 1
    if len(rels) > 1:
        score += 1
    named_speakers = [s for s in speakers if s.get("speaker_name")]
    if named_speakers:
        score += 1

    if score < 3:
        return None

    # Boost segments with substantive OCR (chyrons, headlines, data)
    if ocr and len(ocr) > 15:
        ocr_words = set(ocr.lower().split())
        asr_words = set(asr.lower().split()) if asr else set()
        unique_ratio = len(ocr_words - asr_words) / max(1, len(ocr_words))
        if unique_ratio > 0.3:
            score += 8  # OCR with unique info is the gold standard
        else:
            score += 3

    return (score, data)


def find_rich_segments(index_data, max_segments=10):
    """Find time windows with the most multi-modal evidence, spread across the video.

    Supports both v1 (flat segments) and v2 (layered) indexes.
    Divides the video timeline into equal windows and picks the richest
    content from each window, ensuring coverage throughout the video.
    """
    # Detect schema version
    if index_data.get("schema_version", 1) >= 2:
        return _find_rich_windows_v2(index_data, max_segments)
    else:
        return _find_rich_segments_v1(index_data, max_segments)


def _find_rich_windows_v2(index_data, max_segments=10):
    """Find rich time windows from v2 layered index."""
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    if not duration_ms:
        return []

    # Use shot boundaries as candidate windows (natural segmentation)
    shots = layers.get("shots", {}).get("items", [])
    if not shots:
        # Fallback: create 30-second windows
        shots = [
            {"start_ms": t, "end_ms": min(t + 30000, duration_ms)}
            for t in range(0, duration_ms, 30000)
        ]

    # Score each shot as a candidate window
    scored = []
    for shot in shots:
        data = _gather_window_data(layers, shot["start_ms"], shot["end_ms"])
        result = _score_window(data)
        if result:
            scored.append(result)

    if not scored:
        return []

    # Time-bucketed selection: divide timeline into max_segments windows
    num_windows = max_segments
    window_ms = duration_ms / num_windows

    selected = []
    for w in range(num_windows):
        window_start = w * window_ms
        window_end = (w + 1) * window_ms

        in_window = [
            (score, data) for score, data in scored
            if data["start_ms"] >= window_start
            and data["start_ms"] < window_end
        ]

        if in_window:
            in_window.sort(key=lambda x: -x[0])
            selected.append(in_window[0])

    # Fill remaining slots
    if len(selected) < max_segments:
        selected_times = {(d["start_ms"], d["end_ms"]) for _, d in selected}
        remaining = [
            (score, data) for score, data in scored
            if (data["start_ms"], data["end_ms"]) not in selected_times
        ]
        remaining.sort(key=lambda x: -x[0])
        for item in remaining:
            if len(selected) >= max_segments:
                break
            selected.append(item)

    selected.sort(key=lambda x: x[1]["start_ms"])
    return [data for _, data in selected]


def _find_rich_segments_v1(index_data, max_segments=10):
    """Find rich segments from v1 flat index (backward compat)."""
    segs = index_data.get("segments", [])
    speaker_names = index_data.get("speaker_names", {})

    scored = []
    for seg in segs:
        # Convert v1 segment to window-like data dict
        data = {
            "start_ms": seg.get("start_ms", 0),
            "end_ms": seg.get("end_ms", 0),
            "asr": seg.get("asr_transcript", ""),
            "captions": seg.get("visual_caption", ""),
            "ocr": seg.get("ocr_text", ""),
            "entities": seg.get("named_entities", []),
            "relations": seg.get("relations", []),
            "speakers": [{"speaker_name": speaker_names.get(seg.get("primary_speaker", ""), "")}] if seg.get("primary_speaker") else [],
            "_v1_seg": seg,  # Keep reference for format_segment_context
        }
        result = _score_window(data)
        if result:
            scored.append(result)

    if not scored:
        return []

    duration_ms = max(seg.get("end_ms", 0) for seg in segs) if segs else 0
    if duration_ms == 0:
        scored.sort(key=lambda x: -x[0])
        return [data for _, data in scored[:max_segments]]

    num_windows = max_segments
    window_ms = duration_ms / num_windows

    selected = []
    for w in range(num_windows):
        window_start = w * window_ms
        window_end = (w + 1) * window_ms

        in_window = [
            (score, data) for score, data in scored
            if data["start_ms"] >= window_start
            and data["start_ms"] < window_end
        ]
        if in_window:
            in_window.sort(key=lambda x: -x[0])
            selected.append(in_window[0])

    if len(selected) < max_segments:
        selected_times = {(d["start_ms"], d["end_ms"]) for _, d in selected}
        remaining = [
            (score, data) for score, data in scored
            if (data["start_ms"], data["end_ms"]) not in selected_times
        ]
        remaining.sort(key=lambda x: -x[0])
        for item in remaining:
            if len(selected) >= max_segments:
                break
            selected.append(item)

    selected.sort(key=lambda x: x[1]["start_ms"])
    return [data for _, data in selected]


def format_segment_context(data, speaker_names_unused=None, index_data=None):
    """Format a segment/window's multi-modal evidence for the LLM.

    Works with both v2 window dicts (from _gather_window_data) and
    v1 segment dicts (backward compat).
    """
    # If this is a v2 window data dict
    parts = []

    # Speaker info
    speakers = data.get("speakers", [])
    named = [s.get("speaker_name") for s in speakers if s.get("speaker_name")]
    if named:
        parts.append(f"SPEAKER: {', '.join(set(named))}")

    asr = data.get("asr", "")
    if asr:
        parts.append(f"SPEECH TRANSCRIPT:\n{asr[:600]}")

    caption = data.get("captions", "")
    if caption and not caption.startswith("The image is completely black"):
        parts.append(f"VISUAL DESCRIPTION:\n{caption[:400]}")

    ocr = data.get("ocr", "")
    if ocr and len(ocr) > 5 and not ocr.startswith(("The image", "The video", "None")):
        parts.append(f"ON-SCREEN TEXT/GRAPHICS:\n{ocr[:300]}")

    ents = data.get("entities", [])
    if ents:
        seen = set()
        unique = []
        for e in ents:
            text = e.get("text", "")
            key = text.lower()
            if key not in seen and len(text) > 2:
                seen.add(key)
                unique.append(f'{text} ({e.get("type", "?")})')
        if unique:
            parts.append(f"NAMED ENTITIES: {', '.join(unique[:15])}")

    rels = data.get("relations", [])
    if rels:
        rel_strs = []
        seen_rels = set()
        for r in rels:
            if isinstance(r, dict):
                key = (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
                if key not in seen_rels:
                    seen_rels.add(key)
                    rel_strs.append(f"{key[0]} -> {key[1]} -> {key[2]}")
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                key = tuple(r[:3])
                if key not in seen_rels:
                    seen_rels.add(key)
                    rel_strs.append(f"{r[0]} -> {r[1]} -> {r[2]}")
        if rel_strs:
            parts.append(f"RELATIONS: {'; '.join(rel_strs[:8])}")

    # Entity grounding (v2: on entity items; v1: on grounded_entities dict)
    grounded_strs = []
    for e in ents:
        if e.get("grounding"):
            grounded_strs.append(f"{e.get('text', '')}: {e['grounding'][:80]}")
    if not grounded_strs:
        # v1 fallback
        grounded = data.get("grounded_entities", {})
        if grounded:
            grounded_strs = [f"{k}: {v[:80]}" for k, v in grounded.items() if v]
    if grounded_strs:
        parts.append(f"ENTITY KNOWLEDGE: {'; '.join(grounded_strs[:5])}")

    return "\n\n".join(parts)


def call_ollama(prompt, system=SYSTEM_PROMPT, temperature=0.7):
    """Call Ollama chat API with JSON format to disable thinking mode."""
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
                "temperature": temperature,
                "num_predict": 3000,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def parse_qa_response(text):
    """Parse LLM response into QA objects."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        parsed = json.loads(text)
        # Handle {"questions": [...]} format
        if isinstance(parsed, dict):
            if "questions" in parsed:
                return parsed["questions"]
            # Single question object — wrap in list
            if "question" in parsed:
                return [parsed]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find array brackets
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return []


def generate_for_video(index_data):
    """Generate cross-modal questions for one video."""
    vid = index_data["video_id"]
    speaker_names = index_data.get("speaker_names", {})
    # v2: speaker_names may be nested in the speakers layer
    if not speaker_names and index_data.get("schema_version", 1) >= 2:
        speaker_names = index_data.get("layers", {}).get("speakers", {}).get("speaker_names", {})
    broadcast_date = index_data.get("broadcast_date", "")

    rich_segs = find_rich_segments(index_data)
    if not rich_segs:
        return []

    results = []

    # Process segments/windows in batches of 2-3 for context
    for i in range(0, len(rich_segs), 2):
        batch = rich_segs[i:i + 2]

        context_parts = []
        if broadcast_date:
            context_parts.append(f"BROADCAST DATE: {broadcast_date}")
        context_parts.append(f"VIDEO: {vid}")

        for j, seg in enumerate(batch):
            ctx = format_segment_context(seg)
            context_parts.append(f"\n--- SEGMENT {j + 1} ---\n{ctx}")

        context = "\n".join(context_parts)

        prompt = f"""Here is multi-modal evidence from a broadcast news video:

{context}

Generate 3-5 comprehension questions that require combining information from multiple modalities (speech + visual, speech + on-screen text, visual + entities, etc.) to answer.

Focus on questions about:
- What was being discussed and what was shown simultaneously
- What information the on-screen graphics/text add to the spoken content
- How the visual content relates to what's being said
- What the speaker's position or argument was, supported by visual evidence

Return a JSON object with a "questions" key containing an array of question objects."""

        try:
            response = call_ollama(prompt)
            items = parse_qa_response(response)
            for item in items:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "")
                a = item.get("answer", "")
                if not q or not a:
                    continue
                grounding = item.get("grounding", {})

                # Verify grounding: check that excerpts actually appear in the context
                speech_ex = grounding.get("speech_excerpt") or ""
                visual_ref = grounding.get("visual_reference") or ""
                ocr_ref = grounding.get("ocr_reference") or ""

                # At least one grounding field must be non-empty
                if not speech_ex and not visual_ref and not ocr_ref:
                    continue  # skip ungrounded questions

                # Verify excerpts exist in context (fuzzy: check first 30 chars)
                grounded_ok = False
                if speech_ex and speech_ex[:30].lower() in context.lower():
                    grounded_ok = True
                if visual_ref and visual_ref[:30].lower() in context.lower():
                    grounded_ok = True
                if ocr_ref and ocr_ref[:20].lower() in context.lower():
                    grounded_ok = True

                if not grounded_ok:
                    continue  # skip hallucinated grounding

                # Reject questions about technical artifacts
                all_grounding_text = f"{speech_ex} {visual_ref} {ocr_ref} {q} {a}".lower()
                if any(artifact in all_grounding_text for artifact in (
                    "test pattern", "color bar", "black screen", "completely black",
                    "countdown", "is unrelated", "does not directly relate",
                    "no visible content",
                )):
                    continue

                results.append({
                    "question": q,
                    "answer": a,
                    "modalities_required": item.get("modalities_required", []),
                    "reasoning_type": item.get("reasoning_type", "cross-modal"),
                    "grounding": {
                        "speech_excerpt": speech_ex or None,
                        "visual_reference": visual_ref or None,
                        "ocr_reference": ocr_ref or None,
                        "reasoning": grounding.get("reasoning", ""),
                    },
                    "video_id": vid,
                    "broadcast_date": broadcast_date,
                    "difficulty": "L3" if len(item.get("modalities_required", [])) <= 2 else "L4",
                    "source_evidence": context,
                    "source_segment_times": [
                        {"start_ms": seg.get("start_ms", 0), "end_ms": seg.get("end_ms", 0)}
                        for seg in batch
                    ],
                })
        except Exception as e:
            print(f"  Error for {vid}: {e}", file=sys.stderr)

        time.sleep(1)  # Rate limit

    return results


def main():
    index_dir = Path("data/video_indexes")
    all_results = []

    videos = sorted(index_dir.glob("*.json"))
    out = Path("qa-data/raw/llm_comprehension_qa.jsonl")

    # Load existing results to skip already-processed videos
    existing_vids = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    q = json.loads(line)
                    existing_vids.add(q.get("video_id", ""))
                    all_results.append(q)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(all_results)} existing questions from {len(existing_vids)} videos", flush=True)

    print(f"Processing {len(videos)} videos with {MODEL}...", flush=True)

    # Append mode — skip already-processed videos
    with open(out, "a" if existing_vids else "w") as fout:
        for i, f in enumerate(videos):
            index_data = json.load(open(f))
            vid = index_data["video_id"]

            if vid in existing_vids:
                continue

            rich = find_rich_segments(index_data)
            if not rich:
                print(f"  [{i + 1}/{len(videos)}] {vid[-25:]}: no rich segments, skipping", flush=True)
                continue

            print(f"  [{i + 1}/{len(videos)}] {vid[-25:]}: {len(rich)} rich segments...", end=" ", flush=True)
            results = generate_for_video(index_data)
            all_results.extend(results)
            # Write immediately
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"{len(results)} questions (total: {len(all_results)})", flush=True)

    # Stats
    by_type = defaultdict(int)
    for r in all_results:
        by_type[r.get("reasoning_type", "unknown")] += 1

    print(f"\nTotal: {len(all_results)} questions")
    print(f"By reasoning type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print(f"\nWritten to {out}")

    # Show samples
    import random
    random.seed(42)
    print("\n--- Samples ---")
    for q in random.sample(all_results, min(8, len(all_results))):
        print(f"  Q: {q['question'][:120]}")
        print(f"  A: {q['answer'][:120]}")
        print(f"  Modalities: {q.get('modalities_required', [])}")
        print()


if __name__ == "__main__":
    main()
