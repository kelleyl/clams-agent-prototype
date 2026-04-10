#!/usr/bin/env python3
"""Generate cross-modal comprehension questions v2: evidence-aware + thinking model.

Key differences from v1 (generate_llm_comprehension.py):
1. Builds entity-modality matrix FIRST to find asymmetries
2. Uses asymmetries as question seeds (visual-only, OCR-only entities)
3. Shows LLM all evidence locations to avoid single-modality shortcuts
4. Refinement step to weave cross-modal context into question text
5. Programmatic cross-modality validation

Uses qwen3:30b (thinking model) on aristotle via Ollama.

Usage:
    python qa-data/generate_llm_comprehension_v2.py \
        --index-dir data/video_indexes \
        --output qa-data/raw/llm_comprehension_qa_v2.jsonl \
        --model qwen3:30b \
        --max-videos 5
"""
import argparse
import json
import os
import re
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b")


# ============================================================
# Step 1: Entity-modality matrix
# ============================================================

def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


# Noise entities to skip (generic visual descriptions, not real entities)
NOISE_PATTERNS = {
    "grey hair", "dark hair", "short hair", "brown hair", "blonde hair",
    "white hair", "curly hair", "black hair", "long hair",
    "background", "foreground", "draft", "refine", "the user",
    "left", "right", "center", "top", "bottom",
    # Network/show branding noise from visual captions
    "abc", "cbs", "nbc", "pbs", "cpb", "rbai", "wgbh", "wnet",
    "our news", "news hour", "the news hour", "newshour",
    "mcneil-lehrer", "macneil/lehrer", "macneil-lehrer",
    "fuzzymemories", "united s",
    # Generic visual description artifacts
    "caucasian", "african american", "asian",
}


def build_entity_modality_matrix(index_data):
    """Map each entity to its source modalities and time ranges."""
    entities_layer = index_data.get("layers", {}).get("entities", {})
    matrix = defaultdict(lambda: {"layers": set(), "appearances": [], "types": set()})

    for item in entities_layer.get("items", []):
        text = item.get("text", "").strip()
        text_lower = text.lower()
        if len(text) < 3 or text_lower in NOISE_PATTERNS:
            continue
        # Skip generic visual description artifacts
        if any(text_lower.startswith(p) for p in ("the ", "a ", "an ")):
            if item.get("type") not in ("PERSON", "ORG", "GPE", "FAC", "WORK_OF_ART", "EVENT"):
                continue

        src = item.get("source_layer", "unknown")
        matrix[text_lower]["layers"].add(src)
        matrix[text_lower]["types"].add(item.get("type", ""))
        matrix[text_lower]["appearances"].append({
            "layer": src,
            "start_ms": item.get("start_ms", 0),
            "end_ms": item.get("end_ms", 0),
            "text": text,
            "type": item.get("type", ""),
        })

    # Classify asymmetry type
    for entity, data in matrix.items():
        srcs = data["layers"]
        if srcs == {"visual_captions"}:
            data["asymmetry_type"] = "visual_only"
        elif srcs == {"ocr"}:
            data["asymmetry_type"] = "ocr_only"
        elif srcs == {"asr"}:
            data["asymmetry_type"] = "asr_only"
        else:
            data["asymmetry_type"] = "multi_modal"

    return dict(matrix)


def find_overlapping_text(layers, layer_name, start_ms, end_ms, max_chars=400):
    """Find text from a layer that overlaps a time range."""
    items = layers.get(layer_name, {}).get("items", [])
    texts = []
    for item in items:
        if _overlaps(item.get("start_ms", 0), item.get("end_ms", 0), start_ms, end_ms):
            texts.append(item.get("text", ""))
    return " ".join(texts)[:max_chars]


def find_question_seeds(matrix, index_data, max_seeds=20):
    """Find entity-modality asymmetries that make good question targets."""
    seeds = []
    layers = index_data.get("layers", {})

    for entity, data in matrix.items():
        # Skip entities with only one appearance (not enough context)
        if len(data["appearances"]) < 1:
            continue

        if data["asymmetry_type"] == "visual_only":
            for app in data["appearances"]:
                asr_text = find_overlapping_text(
                    layers, "asr", app["start_ms"] - 5000, app["end_ms"] + 5000)
                if len(asr_text) < 30:
                    continue
                ocr_text = find_overlapping_text(
                    layers, "ocr", app["start_ms"] - 5000, app["end_ms"] + 5000)
                visual_text = find_overlapping_text(
                    layers, "visual_captions", app["start_ms"], app["end_ms"])
                seeds.append({
                    "entity": entity,
                    "entity_display": app["text"],
                    "entity_type": app["type"],
                    "asymmetry_type": "visual_only",
                    "answer_modality": "visual",
                    "time_ms": (app["start_ms"], app["end_ms"]),
                    "visual_context": visual_text,
                    "asr_context": asr_text,
                    "ocr_context": ocr_text,
                    "priority": 3,
                })

        elif data["asymmetry_type"] == "ocr_only":
            for app in data["appearances"]:
                asr_text = find_overlapping_text(
                    layers, "asr", app["start_ms"] - 5000, app["end_ms"] + 5000)
                if len(asr_text) < 30:
                    continue
                visual_text = find_overlapping_text(
                    layers, "visual_captions", app["start_ms"], app["end_ms"])
                seeds.append({
                    "entity": entity,
                    "entity_display": app["text"],
                    "entity_type": app["type"],
                    "asymmetry_type": "ocr_only",
                    "answer_modality": "ocr",
                    "time_ms": (app["start_ms"], app["end_ms"]),
                    "ocr_context": app["text"],
                    "asr_context": asr_text,
                    "visual_context": visual_text,
                    "priority": 2,
                })

        elif data["asymmetry_type"] == "multi_modal":
            # Multi-modal entities with OCR+ASR are high value (chyron + speech)
            has_ocr = "ocr" in data["layers"]
            has_asr = "asr" in data["layers"]
            if has_asr:  # need at least ASR context
                # Find the best appearance (prefer OCR source for chyron-based seeds)
                ocr_apps = [a for a in data["appearances"] if a["layer"] == "ocr"]
                best = ocr_apps[0] if ocr_apps else max(
                    data["appearances"], key=lambda a: a["end_ms"] - a["start_ms"])
                asr_text = find_overlapping_text(
                    layers, "asr", best["start_ms"] - 10000, best["end_ms"] + 10000)
                ocr_text = find_overlapping_text(
                    layers, "ocr", best["start_ms"] - 5000, best["end_ms"] + 5000)
                visual_text = find_overlapping_text(
                    layers, "visual_captions", best["start_ms"], best["end_ms"])
                speaker_text = find_overlapping_text(
                    layers, "speakers", best["start_ms"] - 5000, best["end_ms"] + 5000)
                # Check for alternative evidence elsewhere in video
                alt_evidence = []
                for app in data["appearances"]:
                    if abs(app["start_ms"] - best["start_ms"]) > 30000:
                        alt_text = find_overlapping_text(
                            layers, app["layer"], app["start_ms"], app["end_ms"], 150)
                        alt_evidence.append({
                            "layer": app["layer"],
                            "time_ms": (app["start_ms"], app["end_ms"]),
                            "text": alt_text,
                        })

                # Priority: OCR+ASR (chyron+speech) > visual+ASR > other
                priority = 4 if has_ocr else 2

                seeds.append({
                    "entity": entity,
                    "entity_display": best["text"],
                    "entity_type": best["type"],
                    "asymmetry_type": "multi_modal",
                    "answer_modality": "cross_modal",
                    "layers": sorted(data["layers"]),
                    "time_ms": (best["start_ms"], best["end_ms"]),
                    "asr_context": asr_text,
                    "ocr_context": ocr_text,
                    "visual_context": visual_text,
                    "speaker_context": speaker_text,
                    "alternative_evidence": alt_evidence,
                    "priority": priority,
                })

    # Filter noise: skip visual-only seeds that look like VLM hallucinations
    filtered = []
    for s in seeds:
        if s["asymmetry_type"] == "visual_only":
            # Skip if entity type is not a substantive NER type
            if s.get("entity_type") not in ("PERSON", "ORG", "GPE", "FAC", "LOC",
                                             "WORK_OF_ART", "EVENT", "PRODUCT"):
                continue
            # Skip very short or generic entities
            if len(s["entity"]) < 4:
                continue
        filtered.append(s)

    # Deduplicate: keep highest priority per entity
    seen = set()
    unique_seeds = []
    for s in sorted(filtered, key=lambda x: -x["priority"]):
        if s["entity"] not in seen:
            seen.add(s["entity"])
            unique_seeds.append(s)

    # Chapter-aware sampling: use the video's chapter structure to ensure
    # questions come from throughout the video, not just the beginning.
    chapters = index_data.get("layers", {}).get("chapters", {}).get("items", [])
    duration_ms = index_data.get("duration_ms", 0)

    if not chapters and duration_ms > 60000:
        # No chapters -- create synthetic ones by dividing into ~5min segments
        seg_ms = 300000  # 5 minutes
        chapters = [{"start_ms": t, "end_ms": min(t + seg_ms, duration_ms),
                     "title": f"Segment {t//seg_ms + 1}"}
                    for t in range(0, duration_ms, seg_ms)]

    if chapters:
        # Skip intro/credits chapters
        content_chapters = [ch for ch in chapters
                           if ch.get("title", "").lower() not in
                           ("introduction", "closing and credits", "credits",
                            "closing", "commercial break")]
        if not content_chapters:
            content_chapters = chapters

        # Allocate seeds per chapter proportional to chapter duration
        total_content_ms = sum(ch["end_ms"] - ch["start_ms"] for ch in content_chapters)
        per_chapter = []
        for ch in content_chapters:
            ch_frac = (ch["end_ms"] - ch["start_ms"]) / max(1, total_content_ms)
            n_seeds = max(1, round(ch_frac * max_seeds))
            # Find seeds in this chapter
            ch_seeds = [s for s in unique_seeds
                       if s["time_ms"][0] >= ch["start_ms"]
                       and s["time_ms"][0] < ch["end_ms"]]
            ch_seeds.sort(key=lambda s: -s["priority"])
            per_chapter.append((ch, ch_seeds[:n_seeds]))

        # Collect all chapter-selected seeds
        selected = []
        used = set()
        for ch, ch_seeds in per_chapter:
            for s in ch_seeds:
                if s["entity"] not in used:
                    selected.append(s)
                    used.add(s["entity"])

        # Fill remaining from any chapter
        if len(selected) < max_seeds:
            remaining = [s for s in unique_seeds if s["entity"] not in used]
            remaining.sort(key=lambda s: -s["priority"])
            for s in remaining:
                if len(selected) >= max_seeds:
                    break
                selected.append(s)
                used.add(s["entity"])

        return selected[:max_seeds]

    return unique_seeds[:max_seeds]


# ============================================================
# Step 2: Format evidence for LLM
# ============================================================

def format_seed_context(seed):
    """Format a question seed with full evidence context for the LLM."""
    start_s = seed["time_ms"][0] / 1000
    end_s = seed["time_ms"][1] / 1000

    lines = [
        f"QUESTION SEED ({seed['asymmetry_type']})",
        f"Entity: \"{seed['entity_display']}\" (type: {seed['entity_type']})",
        f"Time range: {start_s:.0f}s - {end_s:.0f}s",
        f"Answer modality: {seed['answer_modality']}",
        "",
    ]

    if seed.get("asr_context"):
        lines.append(f"SPEECH TRANSCRIPT (at this time):\n{seed['asr_context']}")
        lines.append("")
    if seed.get("ocr_context"):
        lines.append(f"ON-SCREEN TEXT:\n{seed['ocr_context']}")
        lines.append("")
    if seed.get("visual_context"):
        lines.append(f"VISUAL DESCRIPTION:\n{seed['visual_context']}")
        lines.append("")

    if seed.get("alternative_evidence"):
        lines.append("ALTERNATIVE EVIDENCE (same entity elsewhere in video):")
        for alt in seed["alternative_evidence"][:3]:
            alt_s = alt["time_ms"][0] / 1000
            lines.append(f"  - {alt['layer']} at {alt_s:.0f}s: {alt['text'][:100]}")
        lines.append("")

    if seed["asymmetry_type"] == "visual_only":
        lines.append("NOTE: This entity appears ONLY in visual descriptions, never spoken.")
        lines.append("The question answer must come from the visual; the question context from speech.")
    elif seed["asymmetry_type"] == "ocr_only":
        lines.append("NOTE: This entity appears ONLY in on-screen text, never spoken.")
        lines.append("Do NOT name this entity in the question. Use a descriptive reference.")
    elif seed["asymmetry_type"] == "multi_modal":
        lines.append("NOTE: This entity appears in multiple layers. Design the question to")
        lines.append("require combining layers. Do NOT use the entity name if it's in a chyron.")

    return "\n".join(lines)


# ============================================================
# Step 3: LLM generation + refinement
# ============================================================

SYSTEM_PROMPT = """You are an expert at generating cross-modal video comprehension questions
for evaluating AI systems that process broadcast news and archival video.

You will receive QUESTION SEEDS showing entities or facts that appear in specific
annotation modalities (visual descriptions, on-screen text, speech transcripts) along
with context from other modalities at the same time range.

Your job is to write questions that REQUIRE combining information across modalities
to answer correctly.

QUESTION DESIGN PRINCIPLES:

1. VISUAL-ONLY seeds (entity seen but not spoken):
   Write questions where the answer comes from the visual but the question
   references what's being discussed in the speech.
   Example: "What geographic feature is visible on the map shown during
   the discussion about the historical artifact?"

2. OCR-ONLY seeds (text on screen, not spoken):
   Write questions where the answer is the on-screen text but you describe
   the context from speech. Do NOT name the entity if it appears in a chyron.
   Example: "What organization does the guest discussing unemployment
   statistics for Black Americans represent, according to the on-screen graphic?"

3. MULTI-MODAL seeds:
   Use descriptive references instead of names. Design questions requiring
   information from different parts of the evidence.

CRITICAL RULES:
- Do NOT name entities that appear in chyrons/on-screen text in the question
- Use descriptive references: "the guest", "the speaker who argued..."
- Keep questions SHORT and NATURAL (1-2 sentences, under 30 words ideal)
- Do NOT reference "the on-screen graphic", "the chyron", "the segment", or timestamps
- Do NOT quote speech excerpts in the question text
- Do NOT tell the model WHERE to find the answer -- just ask WHAT to find
- The question should sound like something a human viewer would naturally ask
- Each question must be unanswerable from any single annotation layer alone

GOOD examples:
- "What organization does the guest discussing unemployment for Black Americans represent?"
- "What geographic feature appears on the map during the historical discussion?"
- "What play is referenced in the Playbill shown during the essay about angels in pop culture?"
BAD examples:
- "According to the on-screen graphic shown at 947s during the segment mentioning 'personal income was up 6.3%'..."
- "What text appears in the lower-third chyron overlay during the interview..."

Return a JSON object:
{
  "questions": [
    {
      "question": "...",
      "answer": "...",
      "modalities_required": ["speech", "visual"],
      "reasoning_type": "cross-modal",
      "grounding": {
        "speech_excerpt": "exact quote from speech",
        "visual_reference": "exact quote from visual description",
        "ocr_reference": "exact on-screen text (or null)",
        "reasoning": "why multiple modalities are needed"
      }
    }
  ]
}"""


REFINEMENT_PROMPT = """Make this question shorter, more natural, and harder to answer
from a single information source.

DRAFT: {question}
ANSWER: {answer}

Rules:
- Under 30 words
- Sound like a natural human question
- Do NOT reference "on-screen graphic", "chyron", "segment", or timestamps
- Do NOT quote speech in the question
- Use enough context that both visual AND speech information are needed
- Do NOT name people whose names appear in on-screen labels

EXAMPLE:
- Draft: "According to the on-screen graphic shown during the segment mentioning personal income growth, what organization does the man represent?"
- Improved: "What organization does the guest discussing unemployment for Black Americans represent?"

Return JSON: {{"improved_question": "..."}}"""


PROVIDER = "ollama"  # "ollama" or "gemini"
GEMINI_CLIENT = None


def _get_gemini_client():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        GEMINI_CLIENT = genai.Client(api_key=api_key)
    return GEMINI_CLIENT


def call_llm(prompt, model, system=None, temperature=0.7, max_tokens=2000):
    """Call LLM via Ollama or Gemini API."""
    if PROVIDER == "gemini":
        return _call_gemini(prompt, model, system, temperature, max_tokens)
    return _call_ollama(prompt, model, system, temperature, max_tokens)


def _call_gemini(prompt, model, system=None, temperature=0.7, max_tokens=2000):
    """Call Gemini API."""
    from google.genai import types
    client = _get_gemini_client()

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )
    if system:
        config.system_instruction = system

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            break
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(5 * (attempt + 1))
                continue
            raise
    else:
        raise Exception("Gemini API unavailable after 3 retries")
    content = response.text or ""
    # Strip thinking tags, markdown fences
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    return content


def _call_ollama(prompt, model, system=None, temperature=0.7, max_tokens=2000):
    """Call Ollama API."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "format": "json",
    }

    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=180)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def generate_questions_for_seeds(seeds, model, video_id=""):
    """Generate questions from a batch of seeds."""
    # Format seeds into prompt
    seed_texts = []
    for i, seed in enumerate(seeds):
        seed_texts.append(f"--- SEED {i+1} ---\n{format_seed_context(seed)}")

    prompt = f"VIDEO: {video_id}\n\n"
    prompt += "\n\n".join(seed_texts)
    prompt += """

Generate 1-2 cross-modal questions per seed. Return JSON with a 'questions' array.

CRITICAL: Your questions and answers must come ONLY from the evidence provided above.
Do NOT use knowledge from other videos or external sources.
Every answer must be a specific term, name, or phrase that appears in the provided
speech transcript, visual description, or on-screen text above.
If the evidence is insufficient to write a good question, skip that seed."""

    try:
        content = call_llm(prompt, model, system=SYSTEM_PROMPT)
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to repair: find the outermost JSON object
            start = content.find('{')
            if start >= 0:
                # Find matching closing brace
                depth = 0
                for i in range(start, len(content)):
                    if content[i] == '{': depth += 1
                    elif content[i] == '}': depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(content[start:i+1])
                        except json.JSONDecodeError:
                            result = {}
                        break
                else:
                    # Truncated JSON -- try adding closing braces
                    truncated = content[start:] + ']}'
                    try:
                        result = json.loads(truncated)
                    except json.JSONDecodeError:
                        print(f"  LLM error: Could not parse JSON ({len(content)} chars)")
                        result = {}
            else:
                print(f"  LLM error: No JSON found in response")
                result = {}
        return result.get("questions", [])
    except Exception as e:
        print(f"  LLM error: {e}")
        return []


def refine_question(question_data, seed, model):
    """Refine a question to incorporate cross-modal context."""
    answer_evidence = seed.get(f"{seed['answer_modality']}_context", seed.get("visual_context", ""))
    context_evidence = seed.get("asr_context", "")
    if seed["answer_modality"] == "visual":
        answer_evidence = seed.get("visual_context", "")
    elif seed["answer_modality"] == "ocr":
        answer_evidence = seed.get("ocr_context", "")

    prompt = REFINEMENT_PROMPT.format(
        question=question_data["question"],
        answer=question_data["answer"],
        answer_modality=seed["answer_modality"],
        answer_evidence=answer_evidence[:300],
        context_evidence=context_evidence[:300],
    )

    try:
        content = call_llm(prompt, model, temperature=0.3, max_tokens=500)
        result = json.loads(content)
        improved = result.get("improved_question", "")
        if improved and len(improved) > 20:
            return improved
    except (json.JSONDecodeError, requests.RequestException):
        pass

    return question_data["question"]


# ============================================================
# Step 4: Cross-modality validation
# ============================================================

def normalize_terms(text):
    """Extract key terms for matching."""
    stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this"}
    words = set(re.sub(r'[^a-z0-9\s]', '', text.lower()).split())
    return words - stops


def validate_cross_modality(question_text, answer, index_data):
    """Verify the question truly requires multiple modalities."""
    answer_terms = normalize_terms(answer)
    if len(answer_terms) < 1:
        return True, "answer_too_short_to_validate"

    layers = index_data.get("layers", {})
    reachable_from = []

    for layer_name in ["asr", "ocr", "visual_captions", "speakers"]:
        items = layers.get(layer_name, {}).get("items", [])
        for item in items:
            text = item.get("text", "").lower()
            text_words = set(re.sub(r'[^a-z0-9\s]', '', text).split())
            # Fuzzy: match if >=50% of answer key terms appear in item
            # (down from 70%, allows partial matches like "US Navy" vs "United States Navy")
            matched = answer_terms & text_words
            if len(matched) >= max(1, len(answer_terms) * 0.5):
                reachable_from.append(layer_name)
                break

    if len(reachable_from) == 0:
        return False, "answer_not_found_in_index"
    # Single-layer reachability is only a hard fail for ASR and OCR,
    # since those are directly searchable text. Visual captions and speakers
    # contain the answer text but the QUESTION still requires ASR context
    # to know what to look for (e.g., "What feature is on the map?" --
    # caption has the feature but you need ASR to know which map matters
    # but you need ASR to know which segment matters).
    if len(reachable_from) == 1 and reachable_from[0] in ("asr", "ocr"):
        return False, f"answer_reachable_from_{reachable_from[0]}_alone"

    # Check: does the question relate to this video's actual content?
    q_terms = normalize_terms(question_text)
    all_text = ""
    for ln in ["asr", "ocr", "visual_captions", "speakers"]:
        for item in layers.get(ln, {}).get("items", []):
            all_text += " " + (item.get("text", "") or "")
    all_text_words = set(re.sub(r'[^a-z0-9\s]', '', all_text.lower()).split())
    # Remove generic question words
    content_words = q_terms - {"what", "which", "who", "how", "does", "did", "show",
                                "appear", "during", "segment", "guest", "speaker",
                                "discussing", "discussed", "featured", "represents",
                                "name", "organization", "title", "location", "color"}
    if content_words:
        matched = content_words & all_text_words
        if len(matched) < max(1, len(content_words) * 0.3):
            return False, f"question_not_grounded_in_video"

    # Check: does the question name a PERSON or ORG from OCR chyrons?
    # Only flag multi-word proper names, not common words
    ocr_names = set()
    for item in layers.get("ocr", {}).get("items", []):
        text = item.get("text", "") or ""
        # Extract likely proper names (capitalized words, multi-word)
        for line in text.split("\n"):
            line = line.strip()
            if line and line[0].isupper() and len(line) > 4:
                ocr_names.add(line.lower())
    q_lower = question_text.lower()
    for name in ocr_names:
        if name in q_lower and len(name.split()) >= 2:
            return False, f"question_names_ocr_entity:{name}"

    return True, "valid"


# ============================================================
# Main
# ============================================================

def generate_for_video(video_id, index_data, model, max_seeds=15):
    """Generate v2 questions for a single video."""
    matrix = build_entity_modality_matrix(index_data)
    seeds = find_question_seeds(matrix, index_data, max_seeds=max_seeds)

    if not seeds:
        return []

    # Stats
    by_type = defaultdict(int)
    for s in seeds:
        by_type[s["asymmetry_type"]] += 1
    print(f"  Seeds: {len(seeds)} ({dict(by_type)})")

    # Generate in batches of 3-4 seeds
    all_questions = []
    for batch_start in range(0, len(seeds), 3):
        batch = seeds[batch_start:batch_start + 3]
        questions = generate_questions_for_seeds(batch, model, video_id=video_id)

        for q in questions:
            # Find which seed this question relates to
            best_seed = batch[0]  # default
            answer_lower = q.get("answer", "").lower()
            question_lower = q.get("question", "").lower()
            combined = answer_lower + " " + question_lower
            best_overlap = 0
            for seed in batch:
                # Match by entity appearing in answer or question
                entity_words = set(seed["entity"].split())
                overlap = len(entity_words & set(combined.split()))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_seed = seed

            # Refine question
            original_q = q["question"]
            q["question"] = refine_question(q, best_seed, model)
            q["original_question"] = original_q

            # Validate cross-modality
            is_valid, reason = validate_cross_modality(
                q["question"], q.get("answer", ""), index_data)
            q["cross_modal_valid"] = is_valid
            q["validation_reason"] = reason

            # Add metadata
            q["video_id"] = video_id
            q["seed_entity"] = best_seed["entity"]
            q["seed_type"] = best_seed["asymmetry_type"]
            q["seed_time_ms"] = list(best_seed["time_ms"])
            q["source_segment_times"] = [{"start_ms": best_seed["time_ms"][0],
                                           "end_ms": best_seed["time_ms"][1]}]
            q["pipeline_version"] = "v2"

            # Difficulty
            mods = q.get("modalities_required", [])
            q["difficulty"] = "L4" if len(mods) > 2 else "L3"

            all_questions.append(q)

        if PROVIDER == "ollama":
            time.sleep(1)
        else:
            time.sleep(0.2)  # Gemini rate limiting is generous

    # Deduplicate by answer (keep first occurrence)
    seen_answers = set()
    deduped = []
    for q in all_questions:
        ans = q.get("answer", "").lower().strip()
        if ans and ans not in seen_answers:
            seen_answers.add(ans)
            deduped.append(q)

    return deduped


def main():
    parser = argparse.ArgumentParser(description="Generate v2 cross-modal QA")
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path,
                        default=Path("qa-data/raw/llm_comprehension_qa_v2.jsonl"))
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama",
                        help="LLM provider: ollama (local) or gemini (Google API)")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=15,
                        help="Max question seeds per video")
    parser.add_argument("--videos", type=str, nargs="*",
                        help="Specific video IDs to process")
    args = parser.parse_args()

    # Set provider
    global PROVIDER
    PROVIDER = args.provider
    if args.provider == "gemini" and not args.model.startswith("gemini"):
        args.model = "gemini-2.5-flash"

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load indexes
    indexes = {}
    for f in sorted(args.index_dir.glob("*.json")):
        indexes[f.stem] = json.load(open(f))

    if args.videos:
        video_ids = [v for v in args.videos if v in indexes]
    else:
        video_ids = list(indexes.keys())

    if args.max_videos:
        video_ids = video_ids[:args.max_videos]

    print(f"V2 QA Generation")
    print(f"Model: {args.model}")
    print(f"Videos: {len(video_ids)}")
    print(f"Max seeds per video: {args.max_seeds}")
    print()

    stats = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, video_id in enumerate(video_ids):
            print(f"[{i+1}/{len(video_ids)}] {video_id}")
            idx = indexes[video_id]

            questions = generate_for_video(video_id, idx, args.model, args.max_seeds)

            valid = sum(1 for q in questions if q.get("cross_modal_valid"))
            invalid = len(questions) - valid
            stats["total"] += len(questions)
            stats["valid"] += valid
            stats["invalid"] += invalid

            for j, q in enumerate(questions):
                # Use hash to avoid ID collisions from truncated video names
                import hashlib
                vid_hash = hashlib.md5(video_id.encode()).hexdigest()[:8]
                q["id"] = f"v2-{vid_hash}-{j:03d}"
                fout.write(json.dumps(q, ensure_ascii=False) + "\n")
                fout.flush()

            print(f"  Generated: {len(questions)} ({valid} valid, {invalid} invalid)")
            print()

    # Cross-video deduplication: remove questions with near-identical templates
    # reused across different videos (hallucinated question patterns)
    print(f"\nCross-video deduplication...")
    all_questions = [json.loads(l) for l in open(args.output)]

    def normalize_question(q_text):
        """Normalize question to detect template reuse."""
        # Replace proper nouns/specific terms with placeholder
        norm = re.sub(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", "X", q_text)
        # Also normalize numbers
        norm = re.sub(r'\d+', 'N', norm)
        return norm.strip()

    # Find templates used across 2+ different videos
    from collections import defaultdict as dd
    template_videos = dd(set)
    for q in all_questions:
        template = normalize_question(q["question"])
        template_videos[template].add(q["video_id"])

    # Templates appearing in 2+ videos are suspicious
    bad_templates = {t for t, vids in template_videos.items() if len(vids) >= 2}

    if bad_templates:
        # For each bad template, keep only the first occurrence
        kept = []
        removed = 0
        seen_templates = set()
        for q in all_questions:
            template = normalize_question(q["question"])
            if template in bad_templates:
                if template not in seen_templates:
                    kept.append(q)
                    seen_templates.add(template)
                else:
                    removed += 1
            else:
                kept.append(q)

        # Rewrite output
        with open(args.output, "w") as fout:
            for q in kept:
                fout.write(json.dumps(q, ensure_ascii=False) + "\n")

        print(f"  Removed {removed} cross-video duplicates")
        stats["total"] = len(kept)
        stats["valid"] = sum(1 for q in kept if q.get("cross_modal_valid"))
        stats["invalid"] = stats["total"] - stats["valid"]

    print(f"\nDone.")
    print(f"Total: {stats['total']}")
    print(f"Valid (cross-modal): {stats['valid']}")
    print(f"Invalid: {stats['invalid']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
