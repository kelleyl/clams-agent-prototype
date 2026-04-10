#!/usr/bin/env python3
"""Tag benchmark questions with quality filter flags.

Adds a 'quality_flags' field to each question with any issues found.
Questions are NOT removed -- flags are used downstream to filter during eval.

Usage:
    python qa-data/filter_benchmark.py \
        --input qa-data/benchmark/benchmark.jsonl \
        --output qa-data/benchmark/benchmark_tagged.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path


# Keywords that indicate the generator flagged its own output as invalid
SELF_FLAG_KEYWORDS = [
    "invalid", "problematic", "not valid", "cannot be answered",
    "unanswerable", "not answerable", "question is flawed",
    "no clear answer", "no correct answer",
]

# Hedging language in answers suggesting uncertainty
HEDGING_PHRASES = [
    "possibly", "unclear", "unknown", "not clear", "might be",
    "may be", "context is unclear", "cannot determine",
    "not enough information", "insufficient",
]

INTERSTITIAL_KEYWORDS = [
    "commercial", "promo", "bumper", "show branding", "logo", "credits",
    "test pattern", "slate", "station identification", "underwriter", "sponsor",
]

TRIVIAL_VISUAL_KEYWORDS = [
    "hair color", "suit color", "shirt color", "tie color", "glasses",
    "smiling", "standing", "sitting", "older man", "older woman",
]

OCR_PREMISE_PATTERNS = [
    "on-screen text", "on screen text", "on-screen title", "lower-third",
    "lower third", "graphic", "caption", "logo",
]


def check_self_flagged(q):
    """Check if generator flagged this question as invalid."""
    searchable = json.dumps(q.get("grounding", {})).lower()
    dr = q.get("distractor_rationale", "") or ""
    if isinstance(dr, list):
        dr = " ".join(str(x) for x in dr)
    searchable += " " + dr.lower()
    for kw in SELF_FLAG_KEYWORDS:
        if kw in searchable:
            return f"self_flagged:{kw}"
    return None


def check_modalities(q):
    """Check if question requires sufficient modalities."""
    mods = q.get("modalities_required", [])
    if len(mods) == 0:
        return "zero_modalities"
    if len(mods) == 1:
        return f"single_modality:{mods[0]}"
    return None


def check_vague_answer(q):
    """Check if the answer uses hedging/uncertain language."""
    answer = q.get("mc_correct_text", "") or q.get("reference_answer", "") or ""
    answer_lower = answer.lower()
    for phrase in HEDGING_PHRASES:
        if phrase in answer_lower:
            return f"vague_answer:{phrase}"
    return None


def check_null_grounding(q):
    """Check if grounding has no evidence at all."""
    g = q.get("grounding", {})
    if not g:
        return "null_grounding"
    has_speech = bool(g.get("speech_excerpt"))
    has_visual = bool(g.get("visual_reference"))
    has_ocr = bool(g.get("ocr_reference") or g.get("ocr_evidence"))
    if not has_speech and not has_visual and not has_ocr:
        return "null_grounding"
    return None


def check_premise_grounding_mismatch(q):
    """Check for question premises not supported by grounding."""
    question = (q.get("question", "") or "").lower()
    grounding = q.get("grounding", {}) or {}
    ocr = (grounding.get("ocr_reference") or grounding.get("ocr_evidence") or "").lower()
    visual = (grounding.get("visual_reference") or "").lower()
    reasoning = (grounding.get("reasoning") or "").lower()

    if any(p in question for p in OCR_PREMISE_PATTERNS):
        if not ocr or "no visible text" in ocr:
            return "premise_mismatch:ocr"

    if "on-screen visual" in question and not visual:
        return "premise_mismatch:visual"

    if "question is invalid" in reasoning or "on-screen text is missing" in reasoning:
        return "premise_mismatch:self_invalid"

    return None


def check_interstitial_or_trivial(q):
    """Check for low-value interstitial or trivial visual questions."""
    haystack = " ".join([
        q.get("question", "") or "",
        q.get("mc_correct_text", "") or q.get("reference_answer", "") or "",
        json.dumps(q.get("grounding", {})),
    ]).lower()

    if any(k in haystack for k in INTERSTITIAL_KEYWORDS):
        return "interstitial_content"
    if any(k in haystack for k in TRIVIAL_VISUAL_KEYWORDS):
        return "trivial_visual"
    return None


def check_generator_validation(q):
    """Carry forward generation-time validation failures when present."""
    if q.get("cross_modal_valid") is False:
        reason = q.get("validation_reason", "invalid")
        return f"generator_invalid:{reason}"
    return None


def check_single_source_shortcut(q):
    """Flag likely single-source questions based on modalities and grounding."""
    mods = q.get("modalities_required", []) or []
    grounding = q.get("grounding", {}) or {}
    available = [
        bool(grounding.get("speech_excerpt")),
        bool(grounding.get("visual_reference")),
        bool(grounding.get("ocr_reference") or grounding.get("ocr_evidence")),
    ]
    if len(mods) >= 2 and sum(available) <= 1:
        return "single_source_shortcut:grounding"
    return None


def check_blind_correct(q):
    """Check if blind LLM answers correctly (not index-dependent)."""
    adv = q.get("adversarial_result", "")
    if adv in ("blind_correct_conf85", "blind_correct_conf95"):
        return f"blind_correct:{adv}"
    return None


def check_near_duplicate(q, seen_questions):
    """Check if question is near-duplicate of a previous one from same video."""
    vid = q["video_id"]
    qtxt = q["question"]
    for prev_txt, prev_id in seen_questions.get(vid, []):
        if SequenceMatcher(None, qtxt, prev_txt).ratio() > 0.85:
            return f"near_duplicate:{prev_id}"
    return None


def tag_benchmark(questions):
    """Add quality_flags field to each question."""
    seen_questions = defaultdict(list)

    for q in questions:
        flags = []

        r = check_self_flagged(q)
        if r:
            flags.append(r)

        r = check_modalities(q)
        if r:
            flags.append(r)

        r = check_vague_answer(q)
        if r:
            flags.append(r)

        r = check_null_grounding(q)
        if r:
            flags.append(r)

        r = check_premise_grounding_mismatch(q)
        if r:
            flags.append(r)

        r = check_interstitial_or_trivial(q)
        if r:
            flags.append(r)

        r = check_generator_validation(q)
        if r:
            flags.append(r)

        r = check_single_source_shortcut(q)
        if r:
            flags.append(r)

        r = check_near_duplicate(q, seen_questions)
        if r:
            flags.append(r)

        r = check_blind_correct(q)
        if r:
            flags.append(r)

        q["quality_flags"] = flags
        seen_questions[q["video_id"]].append((q["question"], q["id"]))

    return questions


def main():
    parser = argparse.ArgumentParser(description="Tag benchmark questions with quality flags")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        questions = [json.loads(line) for line in f]

    print(f"Input: {len(questions)} questions")

    tagged = tag_benchmark(questions)

    # Write tagged benchmark
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for q in tagged:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Stats
    flagged = [q for q in tagged if q["quality_flags"]]
    clean = [q for q in tagged if not q["quality_flags"]]

    flag_counts = Counter()
    for q in tagged:
        for flag in q["quality_flags"]:
            category = flag.split(":")[0]
            flag_counts[category] += 1

    print(f"\nFlagged: {len(flagged)}")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count}")

    print(f"\nClean (no flags): {len(clean)}")
    fmt = Counter(q.get("format") for q in clean)
    print(f"  Formats: {dict(fmt)}")
    rt = Counter(q.get("reasoning_type") for q in clean)
    print(f"  Reasoning: {dict(rt)}")

    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
