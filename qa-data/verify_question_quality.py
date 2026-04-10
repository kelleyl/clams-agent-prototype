#!/usr/bin/env python3
"""Run structured question verification with accept/tag/reject decisions.

This is a stricter review pass than scalar quality scoring. It checks whether a
question is benchmark-worthy, borderline, or should be rejected entirely.

Usage:
    python qa-data/verify_question_quality.py \
        --questions qa-data/raw/qa_v3.jsonl \
        --output qa-data/raw/qa_v3_verified.jsonl \
        --index-dir data/video_indexes \
        --model qwen3:8b
"""
import argparse
import json
import os
import re
import time
import requests
from collections import Counter
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

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

VERIFY_PROMPT = """You are reviewing a benchmark question for a video QA dataset.

Your job is to decide whether the question should be ACCEPTED, ACCEPTED_BUT_TAGGED,
or REJECTED for benchmark use.

Question: {question}
Answer: {answer}
Modalities required: {modalities}
Reasoning type: {reasoning_type}

Grounding:
- Speech excerpt: {speech}
- Visual reference: {visual}
- OCR reference: {ocr}
- Reasoning note: {reasoning}

Local source window:
{source_window}

Return JSON with:
- "decision": one of ["accept", "accept_but_tag", "reject"]
- "premise_valid": boolean
- "answer_grounded": boolean
- "cross_modal_required": boolean
- "single_source_shortcut": one of ["none", "speech", "ocr", "visual", "unknown"]
- "program_content": one of ["main", "interstitial", "unclear"]
- "question_value": one of ["high", "medium", "low"]
- "ambiguity": one of ["low", "medium", "high"]
- "issues": list of short issue codes
- "rationale": 1-3 sentences

Be strict. Reject questions with invalid premises, missing evidence, low archival value,
or obvious single-source shortcuts."""


def call_llm(prompt, model, temperature=0.1, max_tokens=500):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=90,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def parse_source_window(question, index_doc):
    times = question.get("source_segment_times", []) or []
    if not times:
        return "No source window provided."

    snippets = []
    layers = index_doc.get("layers", {})
    for span in times[:2]:
        start_ms = span.get("start_ms", 0)
        end_ms = span.get("end_ms", start_ms + 30000)
        for layer_name in ["asr", "ocr", "visual_captions", "speakers"]:
            layer_snips = []
            for item in layers.get(layer_name, {}).get("items", []):
                if item.get("start_ms", 0) < end_ms and item.get("end_ms", 0) > start_ms:
                    text = item.get("text", "")
                    if text:
                        layer_snips.append(text[:180])
            if layer_snips:
                snippets.append(f"{layer_name}: {' | '.join(layer_snips[:3])}")
    return "\n".join(snippets) if snippets else "No local evidence found."


def heuristic_review(question, index_doc):
    grounding = question.get("grounding", {}) or {}
    qtext = (question.get("question", "") or "").lower()
    answer = (question.get("answer") or question.get("reference_answer") or question.get("mc_correct_text") or "").lower()
    modalities = question.get("modalities_required", []) or []
    speech = (grounding.get("speech_excerpt") or "").lower()
    visual = (grounding.get("visual_reference") or "").lower()
    ocr = (grounding.get("ocr_reference") or grounding.get("ocr_evidence") or "").lower()
    reasoning = (grounding.get("reasoning") or "").lower()

    issues = []
    premise_valid = True
    answer_grounded = True
    cross_modal_required = len(modalities) >= 2
    shortcut = "none"
    program_content = "main"
    question_value = "medium"
    ambiguity = "low"

    if any(k in " ".join([qtext, speech, visual, ocr, reasoning]) for k in INTERSTITIAL_KEYWORDS):
        program_content = "interstitial"
        issues.append("interstitial_content")

    if any(k in qtext for k in TRIVIAL_VISUAL_KEYWORDS):
        question_value = "low"
        issues.append("trivial_visual")

    if any(p in qtext for p in OCR_PREMISE_PATTERNS) and (not ocr or "no visible text" in ocr):
        premise_valid = False
        issues.append("ocr_premise_without_ocr")

    if "question is invalid" in reasoning or "on-screen text is missing" in reasoning:
        premise_valid = False
        issues.append("self_invalid")

    haystacks = {
        "speech": speech,
        "ocr": ocr,
        "visual": visual,
    }
    answer_tokens = {t for t in re.sub(r"[^a-z0-9\s]", " ", answer).split() if len(t) > 2}
    answer_hits = []
    if answer_tokens:
        for name, text in haystacks.items():
            text_tokens = set(re.sub(r"[^a-z0-9\s]", " ", text).split())
            if len(answer_tokens & text_tokens) >= max(1, len(answer_tokens) * 0.5):
                answer_hits.append(name)
    if not answer_hits:
        answer_grounded = False
        issues.append("answer_not_grounded")
    elif len(answer_hits) == 1:
        shortcut = answer_hits[0]
        if cross_modal_required:
            issues.append("single_source_shortcut")

    if "possibly" in answer or "likely" in answer or "unclear" in answer:
        ambiguity = "medium"
        issues.append("hedged_answer")

    if question.get("reasoning_type") in ("inferential", "comparative", "causal"):
        question_value = "high"
    elif "single_source_shortcut" in issues or "trivial_visual" in issues:
        question_value = "low"

    if not premise_valid or not answer_grounded:
        decision = "reject"
    elif "interstitial_content" in issues or "single_source_shortcut" in issues or "trivial_visual" in issues:
        decision = "accept_but_tag"
    else:
        decision = "accept"

    rationale_bits = []
    if not premise_valid:
        rationale_bits.append("Premise is not supported by grounding.")
    if not answer_grounded:
        rationale_bits.append("Answer is not clearly supported by local evidence.")
    if decision == "accept_but_tag":
        rationale_bits.append("Question is usable but has benchmark quality concerns.")
    if decision == "accept":
        rationale_bits.append("Question appears grounded and reasonably benchmark-worthy.")

    return {
        "decision": decision,
        "premise_valid": premise_valid,
        "answer_grounded": answer_grounded,
        "cross_modal_required": cross_modal_required,
        "single_source_shortcut": shortcut,
        "program_content": program_content,
        "question_value": question_value,
        "ambiguity": ambiguity,
        "issues": issues,
        "rationale": " ".join(rationale_bits) or "Heuristic review completed.",
        "review_method": "heuristic_fallback",
    }


def verify_question(question, index_doc, model):
    grounding = question.get("grounding", {}) or {}
    prompt = VERIFY_PROMPT.format(
        question=question.get("question", ""),
        answer=question.get("answer") or question.get("reference_answer") or question.get("mc_correct_text") or "",
        modalities=question.get("modalities_required", []),
        reasoning_type=question.get("reasoning_type", ""),
        speech=(grounding.get("speech_excerpt") or "none")[:400],
        visual=(grounding.get("visual_reference") or "none")[:300],
        ocr=(grounding.get("ocr_reference") or grounding.get("ocr_evidence") or "none")[:300],
        reasoning=(grounding.get("reasoning") or "none")[:400],
        source_window=parse_source_window(question, index_doc),
    )

    content = call_llm(prompt, model)
    result = json.loads(content)
    result.setdefault("decision", "reject")
    result.setdefault("issues", [])
    result.setdefault("review_method", "llm")
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify question quality with accept/tag/reject decisions")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for path in args.index_dir.glob("*.json"):
        indexes[path.stem] = json.load(open(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    decisions = Counter()
    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            idx = indexes.get(q.get("video_id", ""))
            if not idx:
                q["verification_review"] = {
                    "decision": "accept_but_tag",
                    "issues": ["missing_index"],
                    "rationale": "Video index not found for verification.",
                    "review_method": "missing_index",
                }
            else:
                try:
                    q["verification_review"] = verify_question(q, idx, args.model)
                except Exception as exc:
                    q["verification_review"] = heuristic_review(q, idx)
                    q["verification_review"]["issues"] = ["verification_error"] + q["verification_review"].get("issues", [])
                    q["verification_review"]["rationale"] = (
                        f"LLM verifier unavailable ({exc}). " + q["verification_review"]["rationale"]
                    )

            decisions[q["verification_review"]["decision"]] += 1
            fout.write(json.dumps(q, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(questions)}] {dict(decisions)}", flush=True)
            time.sleep(0.2)

    print("\nVerification summary:")
    for decision, count in sorted(decisions.items()):
        print(f"  {decision}: {count}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
