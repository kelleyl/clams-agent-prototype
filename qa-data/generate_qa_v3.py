#!/usr/bin/env python3
"""Generate cross-modal QA v3: feed full index to Gemini, let it find interesting questions.

Instead of the complex entity-seed pipeline, this simply formats the entire video
index as a readable document and asks Gemini to generate cross-modal questions from it.

Key advantages:
- Gemini sees the full video context, not just entity seeds
- Questions naturally span the full broadcast
- No front-loading bias
- No entity-seed noise
- Much simpler code

Usage:
    export GOOGLE_API_KEY="..."
    python qa-data/generate_qa_v3.py \
        --index-dir data/video_indexes \
        --output qa-data/raw/qa_v3.jsonl \
        --max-questions-per-video 10
"""
import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are generating cross-modal comprehension questions for evaluating
AI systems that process archival broadcast video.

You will receive a full annotated transcript of a broadcast video, including:
- Speech transcripts (what was said, with timestamps)
- On-screen text / chyrons (names, titles, graphics shown on screen)
- Visual descriptions (what was shown on screen)
- Speaker identification (who said what)
- Chapter structure (topic segments)

Generate questions that REQUIRE combining information from at least two different
annotation types to answer. The best questions combine:
- WHO is speaking (from chyron/speaker ID) with WHAT they said (from speech)
- WHAT is shown visually (from visual description) with WHAT is being discussed (from speech)
- ON-SCREEN TEXT (from OCR) with the speech context around it

RULES:
1. Do NOT name people whose names appear in chyrons. Use descriptive references:
   "the guest discussing unemployment" not "John Jacob"
2. Keep questions SHORT and NATURAL (under 30 words, sound like a human would ask)
3. Do NOT reference annotation layers ("the chyron", "the ASR", "the visual description")
4. Each question must be unanswerable from speech alone OR visual alone
5. Answers must come directly from the provided evidence
6. Generate questions from THROUGHOUT the broadcast, not just the beginning
7. Avoid questions about test patterns, bars, slates, credits, or show branding
8. Avoid trivial visual questions (hair color, suit color) unless they serve identification

Return a JSON object with a "questions" array. Each question has:
- "question": the question text
- "answer": the answer (specific term, name, or phrase from the evidence)
- "modalities_required": list of annotation types needed (e.g., ["speech", "ocr"])
- "reasoning_type": "cross-modal", "inferential", "factual", or "comparative"
- "source_time": approximate timestamp (e.g., "15:43")
- "grounding": {
    "speech_excerpt": exact quote from speech (or null),
    "visual_reference": visual description reference (or null),
    "ocr_reference": on-screen text reference (or null),
    "reasoning": why multiple sources are needed
  }"""


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "for", "with",
    "from", "by", "is", "was", "were", "are", "be", "this", "that", "these",
    "those", "it", "its", "their", "his", "her",
}

INTERSTITIAL_PATTERNS = {
    "test pattern", "color bar", "countdown", "station identification", "show branding",
    "opening credits", "closing credits", "funding credits", "promotional spot",
    "commercial break", "commercial", "promo", "bumper", "logo", "slate", "bars",
    "underwriter", "sponsor",
}

LOW_VALUE_VISUAL_PATTERNS = {
    "hair color", "suit color", "shirt color", "wearing glasses", "glasses",
    "tie color", "standing", "sitting", "smiling", "older man", "older woman",
    "white hair", "gray hair", "dark hair",
}

OCR_PREMISE_PATTERNS = (
    "on-screen text", "on screen text", "on-screen title", "graphic", "caption",
    "lower-third", "lower third", "title on screen", "text on screen", "logo",
)


def format_index_as_document(index_data):
    """Format the full video index as a readable annotated transcript."""
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)
    video_id = index_data.get("video_id", "")

    lines = [f"VIDEO: {video_id}"]
    lines.append(f"Duration: {duration_ms/60000:.0f} minutes")
    lines.append("")

    # Chapters overview
    chapters = layers.get("chapters", {}).get("items", [])
    if chapters:
        lines.append("CHAPTER STRUCTURE:")
        for ch in chapters:
            s = ch["start_ms"] / 1000
            lines.append(f"  [{int(s)//60}:{int(s)%60:02d}] {ch.get('title', '')}")
        lines.append("")

    # Build timeline: merge all layers by time
    events = []
    for item in layers.get("asr", {}).get("items", []):
        events.append((item["start_ms"], "SPEECH", item.get("text", "")))
    for item in layers.get("ocr", {}).get("items", []):
        label = item.get("scene_label", "")
        text = item.get("text", "") or ""
        if text and not text.startswith("The ") and text != "None":
            events.append((item["start_ms"], f"ON-SCREEN ({label})", text[:200]))
    for item in layers.get("visual_captions", {}).get("items", []):
        text = item.get("text", "")
        # Truncate verbose VLM captions
        if text:
            first_sentence = text.split(". ")[0] + "."
            if len(first_sentence) > 20:
                events.append((item["start_ms"], "VISUAL", first_sentence[:150]))
    for item in layers.get("speakers", {}).get("items", []):
        name = item.get("speaker_name", "")
        if name and name != "?":
            events.append((item["start_ms"], f"SPEAKER: {name}", item.get("text", "")[:150]))

    events.sort(key=lambda e: e[0])

    # Format as timeline
    lines.append("ANNOTATED TIMELINE:")
    lines.append("")
    for ms, label, text in events:
        m, s = divmod(int(ms / 1000), 60)
        pct = ms / max(1, duration_ms) * 100
        lines.append(f"[{m}:{s:02d} | {pct:.0f}%] {label}: {text}")

    return "\n".join(lines)


def normalize_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_terms(text):
    cleaned = re.sub(r"[^a-z0-9\s]", " ", normalize_text(text))
    return {t for t in cleaned.split() if t and t not in STOPWORDS}


def parse_source_time_to_ms(src_time):
    if not src_time:
        return None
    match = re.match(r"(\d+):(\d+)", src_time)
    if not match:
        return None
    return (int(match.group(1)) * 60 + int(match.group(2))) * 1000


def layer_texts_in_window(index_data, start_ms, end_ms):
    layers = index_data.get("layers", {})
    results = {}
    for layer_name in ["asr", "ocr", "visual_captions", "speakers"]:
        snippets = []
        for item in layers.get(layer_name, {}).get("items", []):
            item_start = item.get("start_ms", 0)
            item_end = item.get("end_ms", 0)
            if item_start < end_ms and item_end > start_ms:
                text = item.get("text", "")
                if text:
                    snippets.append(text)
        results[layer_name] = " ".join(snippets)
    return results


def answer_layers(answer_terms, layer_texts):
    found = []
    for layer_name, text in layer_texts.items():
        terms = text_terms(text)
        if answer_terms and len(answer_terms & terms) >= max(1, len(answer_terms) * 0.5):
            found.append(layer_name)
    return found


def question_has_ocr_premise(question_text):
    qlower = normalize_text(question_text)
    return any(p in qlower for p in OCR_PREMISE_PATTERNS)


def references_interstitial_content(question_text, grounding):
    haystack = " ".join([
        question_text or "",
        grounding.get("speech_excerpt", "") if isinstance(grounding, dict) else "",
        grounding.get("visual_reference", "") if isinstance(grounding, dict) else "",
        grounding.get("ocr_reference", "") if isinstance(grounding, dict) else "",
    ]).lower()
    return any(p in haystack for p in INTERSTITIAL_PATTERNS)


def is_trivial_visual_question(question_text, grounding):
    qlower = normalize_text(question_text)
    if not any(p in qlower for p in LOW_VALUE_VISUAL_PATTERNS):
        return False
    visual = normalize_text((grounding or {}).get("visual_reference", ""))
    return bool(visual)


def answer_present_in_grounding(answer_terms, grounding):
    if not isinstance(grounding, dict):
        return False, []
    found = []
    for field, layer_name in [
        ("speech_excerpt", "asr"),
        ("ocr_reference", "ocr"),
        ("visual_reference", "visual_captions"),
    ]:
        terms = text_terms(grounding.get(field, ""))
        if answer_terms and len(answer_terms & terms) >= max(1, len(answer_terms) * 0.5):
            found.append(layer_name)
    return bool(found), found


def generate_questions(client, model, index_doc, n_questions=10, video_id=""):
    """Ask Gemini to generate questions from the full index document."""
    prompt = f"""{index_doc}

Generate exactly {n_questions} cross-modal questions about this broadcast.
Ensure questions come from throughout the video (not just the beginning).
Each question should require combining information from at least 2 annotation types.

Return JSON: {{"questions": [...]}}"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=8000,
                    response_mime_type="application/json",
                ),
            )
            content = response.text or ""
            content = re.sub(r'^```json\s*', '', content)
            content = re.sub(r'\s*```$', '', content)

            result = json.loads(content)
            return result.get("questions", [])
        except json.JSONDecodeError as e:
            # Try multiple repair strategies
            # 1. Strip markdown fences
            cleaned = re.sub(r'^```json\s*', '', content.strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)
            try:
                result = json.loads(cleaned)
                return result.get("questions", [])
            except:
                pass

            # 2. Find outermost JSON object
            start = cleaned.find('{"questions"')
            if start < 0:
                start = cleaned.find('{')
            if start >= 0:
                # Try progressively truncating to find valid JSON
                for end_offset in range(len(cleaned), start, -1):
                    try:
                        result = json.loads(cleaned[start:end_offset])
                        return result.get("questions", [])
                    except:
                        continue

                # 3. Try closing truncated arrays/objects
                fragment = cleaned[start:]
                for suffix in [']}}', ']}', '}}', '}']:
                    try:
                        result = json.loads(fragment + suffix)
                        qs = result.get("questions", [])
                        if qs:
                            return qs
                    except:
                        continue

            print(f"  JSON error (attempt {attempt+1}): {e}")
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"  API unavailable, retrying in {5*(attempt+1)}s...")
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  Error: {e}")
                break

    return []


def validate_question(q, index_data):
    """Validate that a generated question is usable as cross-modal QA."""
    answer = (q.get("answer", "") or "").lower()
    question = q.get("question", "") or ""
    grounding = q.get("grounding", {}) or {}
    modalities = q.get("modalities_required", []) or []
    details = {"issues": [], "answer_layers_global": [], "answer_layers_local": [], "answer_layers_grounding": []}

    if len(answer) < 2:
        details["issues"].append("answer_too_short")
        return False, "answer_too_short", details

    answer_terms = set(re.sub(r'[^a-z0-9\s]', '', answer).split())
    answer_terms -= STOPWORDS

    if not answer_terms:
        details["issues"].append("no_answer_terms")
        return False, "no_answer_terms", details

    if references_interstitial_content(question, grounding):
        details["issues"].append("interstitial_content")
        return False, "interstitial_content", details

    if is_trivial_visual_question(question, grounding):
        details["issues"].append("trivial_visual")
        return False, "trivial_visual", details

    has_grounding = any(bool(grounding.get(k)) for k in ["speech_excerpt", "visual_reference", "ocr_reference"])
    if not has_grounding:
        details["issues"].append("null_grounding")
        return False, "null_grounding", details

    if question_has_ocr_premise(question) and not grounding.get("ocr_reference"):
        details["issues"].append("ocr_premise_without_ocr")
        return False, "ocr_premise_without_ocr", details

    grounding_ok, grounding_layers = answer_present_in_grounding(answer_terms, grounding)
    details["answer_layers_grounding"] = grounding_layers

    global_texts = layer_texts_in_window(index_data, 0, index_data.get("duration_ms", 0) or 0)
    global_layers = answer_layers(answer_terms, global_texts)
    details["answer_layers_global"] = global_layers
    if not global_layers:
        details["issues"].append("answer_not_in_index")
        return False, "answer_not_in_index", details

    local_layers = []
    source_ms = parse_source_time_to_ms(q.get("source_time", ""))
    if source_ms is not None:
        local_texts = layer_texts_in_window(index_data, max(0, source_ms - 15000), source_ms + 15000)
        local_layers = answer_layers(answer_terms, local_texts)
        details["answer_layers_local"] = local_layers

    if source_ms is not None and not local_layers and not grounding_ok:
        details["issues"].append("answer_not_near_source_time")
        return False, "answer_not_near_source_time", details

    substantive_modalities = [m for m in modalities if m not in {"entities", "relations"}]
    if len(substantive_modalities) >= 2:
        shortcut_layers = local_layers or grounding_layers or global_layers
        if len(set(shortcut_layers)) <= 1:
            details["issues"].append("single_source_shortcut")
            return False, "single_source_shortcut", details

    return True, "valid", details


def main():
    parser = argparse.ArgumentParser(description="Generate QA v3 with Gemini")
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, default=Path("qa-data/raw/qa_v3.jsonl"))
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max-questions-per-video", type=int, default=10)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--videos", type=str, nargs="*")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set")
        return

    client = genai.Client(api_key=api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load indexes
    indexes = {}
    for f in sorted(args.index_dir.glob("*.json")):
        indexes[f.stem] = json.load(open(f))

    video_ids = args.videos if args.videos else list(indexes.keys())
    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    print(f"QA Generation v3 (Gemini)")
    print(f"Model: {args.model}")
    print(f"Videos: {len(video_ids)}")
    print(f"Questions per video: {args.max_questions_per_video}")

    stats = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, vid in enumerate(video_ids):
            idx = indexes.get(vid)
            if not idx:
                continue

            print(f"[{i+1}/{len(video_ids)}] {vid[-40:]}")

            # Format full index
            doc = format_index_as_document(idx)

            # Generate questions
            questions = generate_questions(
                client, args.model, doc,
                n_questions=args.max_questions_per_video,
                video_id=vid
            )

            valid_count = 0
            for j, q in enumerate(questions):
                is_valid, reason, details = validate_question(q, idx)
                q["video_id"] = vid
                q["cross_modal_valid"] = is_valid
                q["validation_reason"] = reason
                q["validation_details"] = details
                q["pipeline_version"] = "v3"

                import hashlib
                vid_hash = hashlib.md5(vid.encode()).hexdigest()[:8]
                q["id"] = f"v3-{vid_hash}-{j:03d}"

                # Parse source_time into source_segment_times
                src_time = q.get("source_time", "")
                if src_time:
                    m = re.match(r'(\d+):(\d+)', src_time)
                    if m:
                        ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
                        q["source_segment_times"] = [{"start_ms": ms, "end_ms": ms + 30000}]

                if is_valid:
                    valid_count += 1
                stats["total"] += 1
                stats["valid" if is_valid else "invalid"] += 1

                fout.write(json.dumps(q, ensure_ascii=False) + "\n")
                fout.flush()

            print(f"  Generated: {len(questions)}, Valid: {valid_count}")
            time.sleep(0.5)

    print(f"\nDone.")
    print(f"Total: {stats['total']}, Valid: {stats['valid']}, Invalid: {stats['invalid']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
