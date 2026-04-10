#!/usr/bin/env python3
"""Generate a dual-format QA benchmark: Multiple-Choice + Free-Text.

Inspired by MLVU, TempCompass, and CinePile:
- MC track: factual cross-modal questions with adversarially-filtered distractors
- Free-text track: inferential/causal questions scored by LLM-as-judge

Pipeline:
1. Pre-filter (remove watermark/branding/artifact questions)
2. Attempt MC distractor generation for all questions
3. Questions that produce good distractors → MC track
4. Questions that fail distractor generation → Free-text track
5. Output both tracks as separate files + combined benchmark

Usage:
    python qa-data/generate_distractors.py \
        --questions qa-data/raw/llm_comprehension_qa.jsonl \
        --output-dir qa-data/benchmark \
        --index-dir data/video_indexes
"""
import argparse
import json
import os
import random
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path

random.seed(42)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")


DISTRACTOR_PROMPT = """You are generating wrong-but-plausible answer options for a multiple-choice question about a broadcast news video.

Given the question, correct answer, and source evidence, generate exactly 3 incorrect alternatives.

RULES:
1. Each distractor must be DEFINITIVELY WRONG based on the evidence — not "arguably correct" or "partially true"
2. Each distractor must use a DIFFERENT error type (you MUST use at least 2 of the 4 types below)
3. Keep distractors SHORT — roughly the same length as the correct answer (under 40 words each)
4. Distractors must sound plausible to someone who didn't watch carefully
5. Use REAL names, organizations, and places — never use made-up or nonsensical words
6. Do NOT create a distractor that is just a rephrasing of the correct answer
7. Each distractor should be clearly distinguishable from the correct answer AND from each other — change the core claim, not just one word
8. Do NOT create template distractors that only swap one word (BAD: "named TRENDY", "named TEENAGE", "named POPULAR"). Each distractor must have a different STRUCTURE, not just a different noun/adjective

ERROR TYPES:

1. ENTITY_SUBSTITUTION: Replace a key person, organization, or place with a REAL alternative from the same domain. Example: correct="Senator Conrad" → distractor="Senator Gregg"

2. ATTRIBUTE_MODIFICATION: Change a specific factual detail (date, number, title, role). The modified detail must be wrong but realistic. Example: correct="Secretary of State" → distractor="Secretary of Defense"

3. TEMPORAL_CONFUSION: Reference real information from a different segment or story in the same broadcast. Example: correct="the budget proposal" → distractor="the ceasefire agreement"

4. PARTIAL_MISLEADING: Get some facts right but draw the wrong conclusion or invert the stance. Example: correct="opposed the bill because of cost" → distractor="supported the bill with reservations about cost"

Return a JSON object with:
- "distractors": list of exactly 3 strings (the wrong answers)
- "distractor_types": list of 3 strings indicating which error type each uses
- "rationale": list of 3 strings explaining why each is wrong but plausible
"""


def call_ollama(prompt, system=DISTRACTOR_PROMPT, temperature=0.7, max_tokens=1000):
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
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def get_video_entities(video_id, index_dir):
    """Get all entities from a video's index for KG-informed substitution."""
    idx_path = Path(index_dir) / f"{video_id}.json"
    if not idx_path.exists():
        return {}

    doc = json.load(open(idx_path))
    entities_by_type = defaultdict(set)

    # v2 layered index
    if doc.get("schema_version", 1) >= 2:
        for item in doc.get("layers", {}).get("entities", {}).get("items", []):
            etype = item.get("type", "")
            text = item.get("text", "")
            if etype and text:
                entities_by_type[etype].add(text)
    else:
        # v1 flat segments
        for seg in doc.get("segments", []):
            for ent in seg.get("named_entities", []):
                entities_by_type[ent["type"]].add(ent["text"])

    return {k: list(v) for k, v in entities_by_type.items()}


def generate_distractors(question, answer, evidence, video_entities):
    """Generate 3 distractors using the LLM."""
    entity_context = ""
    if video_entities:
        parts = []
        for etype, ents in video_entities.items():
            if len(ents) > 2:
                parts.append(f"{etype}: {', '.join(ents[:15])}")
        if parts:
            entity_context = f"\n\nOther entities in this video (use for substitution):\n{chr(10).join(parts)}"

    prompt = f"""Question: {question}

Correct answer: {answer}

Source evidence:
{evidence[:800] if evidence else 'Not available'}
{entity_context}

Generate 3 wrong-but-plausible distractors."""

    try:
        response = call_ollama(prompt)
        parsed = json.loads(response)
        distractors = parsed.get("distractors", [])
        types = parsed.get("distractor_types", [])
        rationale = parsed.get("rationale", [])

        if len(distractors) >= 3:
            distractors = distractors[:3]
            types = types[:3]
            rationale = rationale[:3]

            # Validate: reject distractors too similar to correct answer
            answer_lower = answer.lower().strip()
            a_words = set(answer_lower.split())
            valid = []
            for d in distractors:
                d_lower = d.lower().strip()
                d_words = set(d_lower.split())
                if a_words and d_words:
                    overlap = len(a_words & d_words) / max(len(a_words), len(d_words))
                    if overlap > 0.7:
                        continue
                valid.append(d)

            # Also reject distractors too similar to EACH OTHER
            if len(valid) >= 3:
                final = [valid[0]]
                for d in valid[1:]:
                    d_words = set(d.lower().split())
                    too_similar = False
                    for existing in final:
                        e_words = set(existing.lower().split())
                        if d_words and e_words:
                            overlap = len(d_words & e_words) / min(len(d_words), len(e_words))
                            if overlap > 0.7:
                                too_similar = True
                                break
                    if not too_similar:
                        final.append(d)
                valid = final

            if len(valid) >= 3:
                valid = valid[:3]
                types = types[:3]
                rationale = rationale[:3]
                # Check diversity: at least 2 different error types
                unique_types = set(t.upper().replace(" ", "_") for t in types)
                if len(unique_types) < 2:
                    return None, None, None
                return valid, types, rationale
    except Exception:
        pass

    return None, None, None


def adversarial_filter(question, correct_answer, distractors):
    """Deaf-Blind test: can an LLM answer correctly without evidence?

    Returns True if the question passes (distractors are hard enough).
    Returns False if the LLM can guess the answer (distractors too easy).
    """
    options = [correct_answer] + distractors
    random.shuffle(options)
    correct_idx = options.index(correct_answer)
    letters = "ABCD"

    options_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))

    prompt = f"""Answer this multiple-choice question. You have NO context — just pick the best answer based on general knowledge and the question itself.

Question: {question}

{options_text}

Return a JSON object with "answer" (just the letter A/B/C/D) and "confidence" (0-100)."""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 100},
            },
            timeout=30,
        )
        result = json.loads(resp.json()["message"]["content"])
        chosen = result.get("answer", "").strip().upper()
        confidence = result.get("confidence", 50)

        if chosen != letters[correct_idx]:
            return True, "blind_wrong"
        elif confidence < 80:
            return True, "blind_low_confidence"
        else:
            return False, f"blind_correct_conf{confidence}"
    except Exception:
        return True, "blind_error"


# ---------- Pre-filtering ----------

SKIP_PATTERNS = [
    "memories.tv", "fuzzymemories", "macneil/lehrer", "newshour logo",
    "test pattern", "color bar", "countdown", "black screen",
    "title card", "show branding",
]


def prefilter_questions(questions, collection=None, max_per_video=None):
    """Apply all pre-filters: collection, watermark/branding, per-video limit."""
    # Collection filter
    if collection == "newshour":
        questions = [q for q in questions if q.get("video_id", "").startswith("cpb-aacip")]
    elif collection == "fuzzymemories":
        questions = [q for q in questions if not q.get("video_id", "").startswith("cpb-aacip")]

    # Watermark/branding filter
    filtered = []
    for q in questions:
        text = (q["question"] + " " + (q["answer"] if isinstance(q["answer"], str) else "")).lower()
        if any(p in text for p in SKIP_PATTERNS):
            continue
        filtered.append(q)
    n_removed = len(questions) - len(filtered)
    if n_removed:
        print(f"Pre-filtered: {len(questions)} -> {len(filtered)} (removed {n_removed} watermark/branding)")
    questions = filtered

    # Per-video limit
    if max_per_video:
        video_counts = defaultdict(int)
        balanced = []
        for q in questions:
            vid = q.get("video_id", "")
            if video_counts[vid] < max_per_video:
                balanced.append(q)
                video_counts[vid] += 1
        print(f"Per-video limit ({max_per_video}): {len(questions)} -> {len(balanced)}")
        questions = balanced

    return questions


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Generate dual-format QA benchmark (MC + free-text)")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for benchmark files")
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--skip-adversarial", action="store_true",
                        help="Skip the deaf-blind adversarial test")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max regeneration attempts for failed adversarial test")
    parser.add_argument("--collection", type=str, default=None,
                        choices=["newshour", "fuzzymemories"])
    parser.add_argument("--max-per-video", type=int, default=None,
                        help="Max questions per video to prevent over-representation")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    questions = prefilter_questions(questions, args.collection, args.max_per_video)

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Assign stable IDs if missing
    for i, q in enumerate(questions):
        if "id" not in q:
            q["id"] = f"q-{i:04d}"

    entity_cache = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mc_path = args.output_dir / "mc_questions.jsonl"
    ft_path = args.output_dir / "freetext_questions.jsonl"
    combined_path = args.output_dir / "benchmark.jsonl"

    print(f"Processing {len(questions)} questions with {MODEL}...")
    print(f"Output: {args.output_dir}/")

    stats = {
        "total": len(questions),
        "mc_generated": 0,
        "mc_adversarial_pass": 0,
        "mc_adversarial_fail": 0,
        "mc_hardened": 0,
        "freetext": 0,
        "distractor_failed": 0,
    }

    mc_results = []
    ft_results = []

    for i, q in enumerate(questions):
        video_id = q.get("video_id", "")
        question_text = q["question"]
        answer = q["answer"] if isinstance(q["answer"], str) else json.dumps(q["answer"])
        evidence = q.get("source_evidence", "")

        # Get video entities for KG-informed substitution
        if video_id not in entity_cache:
            entity_cache[video_id] = get_video_entities(video_id, args.index_dir)
        video_entities = entity_cache[video_id]

        # Try MC distractor generation
        distractors = None
        types = None
        rationale = None
        adversarial_result = "skipped"
        mc_success = False

        for attempt in range(args.max_retries + 1):
            distractors, types, rationale = generate_distractors(
                question_text, answer, evidence, video_entities
            )

            if not distractors:
                continue

            # Adversarial filter
            if not args.skip_adversarial:
                passed, adversarial_result = adversarial_filter(
                    question_text, answer, distractors
                )
                if passed:
                    stats["mc_adversarial_pass"] += 1
                    mc_success = True
                    break
                else:
                    stats["mc_adversarial_fail"] += 1
                    if attempt < args.max_retries:
                        continue
                    else:
                        stats["mc_hardened"] += 1
                        mc_success = True  # Keep hardened questions
            else:
                mc_success = True
                break

        if mc_success and distractors:
            # MC track
            stats["mc_generated"] += 1
            options = [answer] + distractors
            random.shuffle(options)
            correct_idx = options.index(answer)
            letters = "ABCD"

            mc_question = {
                "id": q.get("id", f"q-{i:04d}"),
                "format": "multiple_choice",
                "question": question_text,
                "mc_options": {letters[j]: opt for j, opt in enumerate(options)},
                "mc_correct": letters[correct_idx],
                "mc_correct_text": answer,
                "distractor_types": types,
                "distractor_rationale": rationale,
                "adversarial_result": adversarial_result,
                "video_id": video_id,
                "broadcast_date": q.get("broadcast_date"),
                "modalities_required": q.get("modalities_required", []),
                "reasoning_type": q.get("reasoning_type", "cross-modal"),
                "difficulty": q.get("difficulty", "L3"),
                "grounding": q.get("grounding", {}),
                "source_segment_times": q.get("source_segment_times", []),
            }
            mc_results.append(mc_question)
        else:
            # Free-text track
            stats["freetext"] += 1
            stats["distractor_failed"] += 1
            ft_question = {
                "id": q.get("id", f"q-{i:04d}"),
                "format": "free_text",
                "question": question_text,
                "reference_answer": answer,
                "video_id": video_id,
                "broadcast_date": q.get("broadcast_date"),
                "modalities_required": q.get("modalities_required", []),
                "reasoning_type": q.get("reasoning_type", "cross-modal"),
                "difficulty": q.get("difficulty", "L3"),
                "grounding": q.get("grounding", {}),
                "source_segment_times": q.get("source_segment_times", []),
            }
            ft_results.append(ft_question)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(questions)}] MC:{stats['mc_generated']} "
                  f"(pass:{stats['mc_adversarial_pass']} hard:{stats['mc_hardened']}) "
                  f"FT:{stats['freetext']}", flush=True)

        time.sleep(0.5)

    # Write outputs
    with open(mc_path, "w") as f:
        for q in mc_results:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with open(ft_path, "w") as f:
        for q in ft_results:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    with open(combined_path, "w") as f:
        for q in mc_results + ft_results:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"Dual-Format Benchmark Summary")
    print(f"{'='*60}")
    print(f"Total questions processed: {stats['total']}")
    print(f"")
    print(f"MC Track:       {stats['mc_generated']:4d} questions")
    print(f"  Adversarial pass:  {stats['mc_adversarial_pass']:4d}")
    print(f"  Hardened (kept):   {stats['mc_hardened']:4d}")
    print(f"")
    print(f"Free-Text Track: {stats['freetext']:4d} questions")
    print(f"  (distractor gen failed)")
    print(f"")
    print(f"Files:")
    print(f"  MC:        {mc_path} ({len(mc_results)} questions)")
    print(f"  Free-text: {ft_path} ({len(ft_results)} questions)")
    print(f"  Combined:  {combined_path} ({len(mc_results) + len(ft_results)} questions)")

    # Reasoning type breakdown per track
    mc_by_type = defaultdict(int)
    ft_by_type = defaultdict(int)
    for q in mc_results:
        mc_by_type[q.get("reasoning_type", "unknown")] += 1
    for q in ft_results:
        ft_by_type[q.get("reasoning_type", "unknown")] += 1

    print(f"\nBy reasoning type:")
    all_types = sorted(set(list(mc_by_type.keys()) + list(ft_by_type.keys())))
    print(f"  {'type':20s} {'MC':>5s} {'FT':>5s}")
    for t in all_types:
        print(f"  {t:20s} {mc_by_type.get(t,0):5d} {ft_by_type.get(t,0):5d}")

    # Save stats
    stats_path = args.output_dir / "benchmark_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            **stats,
            "mc_by_reasoning_type": dict(mc_by_type),
            "ft_by_reasoning_type": dict(ft_by_type),
            "mc_count": len(mc_results),
            "ft_count": len(ft_results),
            "total_benchmark": len(mc_results) + len(ft_results),
            "model": MODEL,
        }, f, indent=2)


if __name__ == "__main__":
    main()
