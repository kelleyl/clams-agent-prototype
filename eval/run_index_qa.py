#!/usr/bin/env python3
"""Answer QA benchmark questions using our structured video index.

This is the CLAMS-Agent baseline — given a question, search the index
for relevant segments and use an LLM to synthesize an answer WITH
grounded evidence citations (segment timestamps + quoted evidence).

The model must:
1. Search the index for relevant segments
2. Cite specific segment time ranges
3. Quote supporting evidence from ASR/OCR/visual captions
4. Answer the question based on cited evidence

Usage:
    # Run on dual-format benchmark
    python eval/run_index_qa.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --output eval/results/index_baseline.jsonl \
        --model qwen3:8b

    # MC only
    python eval/run_index_qa.py \
        --benchmark qa-data/benchmark/mc_questions.jsonl \
        --output eval/results/index_mc.jsonl

    # Limit for testing
    python eval/run_index_qa.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --output eval/results/test.jsonl \
        --max-questions 10
"""
import argparse
import json
import os
import sys
import time
import requests
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8100")
API_BACKEND = os.environ.get("API_BACKEND", "ollama")  # "ollama" or "vllm"


def call_llm(prompt, model, temperature=0.3, max_tokens=500, json_format=True):
    """Call LLM via Ollama or vLLM OpenAI-compatible API."""
    if API_BACKEND == "vllm":
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_format:
            body["response_format"] = {"type": "json_object"}
        resp = requests.post(f"{VLLM_URL}/v1/chat/completions", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    else:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_format:
            body["format"] = "json"
        resp = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def search_index(question, index_data, top_k=5):
    """Find the most relevant segments/windows for a question using keyword matching.

    Supports both v1 (flat segments) and v2 (layered) indexes.
    """
    question_words = set(question.lower().split())
    stops = {"the", "a", "an", "is", "was", "what", "how", "did", "does",
             "who", "where", "when", "why", "which", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this",
             "based", "according", "between", "during", "about"}
    question_words -= stops

    if index_data.get("schema_version", 1) >= 2:
        return _search_index_v2(question_words, index_data, top_k)

    # v1: search flat segments
    segments = index_data.get("segments", [])
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
        score = len(overlap)
        if score > 0:
            scored.append((score, seg))

    scored.sort(key=lambda x: -x[0])
    return [seg for _, seg in scored[:top_k]]


def _search_index_v2(question_words, index_data, top_k):
    """Search v2 layered index by scoring shots by keyword overlap."""
    layers = index_data.get("layers", {})
    shots = layers.get("shots", {}).get("items", [])
    if not shots:
        return []

    scored = []
    for shot in shots:
        # Gather all text overlapping this shot
        texts = []
        for layer_name in ("asr", "ocr", "visual_captions"):
            for item in layers.get(layer_name, {}).get("items", []):
                if _overlaps(item["start_ms"], item["end_ms"], shot["start_ms"], shot["end_ms"]):
                    texts.append(item.get("text", ""))

        for item in layers.get("entities", {}).get("items", []):
            if _overlaps(item["start_ms"], item["end_ms"], shot["start_ms"], shot["end_ms"]):
                texts.append(item.get("text", ""))

        text = " ".join(texts).lower()
        text_words = set(text.split())
        overlap = question_words & text_words
        score = len(overlap)
        if score > 0:
            # Build a context dict for this shot
            window = _build_shot_context(shot, layers)
            scored.append((score, window))

    scored.sort(key=lambda x: -x[0])
    return [data for _, data in scored[:top_k]]


def _build_shot_context(shot, layers):
    """Build a flat context dict for a shot from layers."""
    s, e = shot["start_ms"], shot["end_ms"]
    asr_texts = []
    ocr_texts = []
    captions = []
    entities = []
    relations = []
    speakers = []

    for item in layers.get("asr", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            asr_texts.append(item.get("text", ""))
    for item in layers.get("ocr", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            ocr_texts.append(item.get("text", ""))
    for item in layers.get("visual_captions", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            captions.append(item.get("text", ""))
    for item in layers.get("entities", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            entities.append(item)
    for item in layers.get("relations", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            relations.append(item)
    for item in layers.get("speakers", {}).get("items", []):
        if _overlaps(item["start_ms"], item["end_ms"], s, e):
            speakers.append(item)

    return {
        "start_ms": s,
        "end_ms": e,
        "asr_transcript": " ".join(asr_texts),
        "ocr_text": " ".join(ocr_texts),
        "visual_caption": " ".join(captions),
        "named_entities": entities,
        "relations": relations,
        "speakers": speakers,
    }


def format_context(segments, index_data):
    """Format retrieved segments/windows as context for the LLM."""
    speaker_names = index_data.get("speaker_names", {})
    # v2: speaker_names in layers
    if not speaker_names and index_data.get("schema_version", 1) >= 2:
        speaker_names = index_data.get("layers", {}).get("speakers", {}).get("speaker_names", {})

    parts = []
    for seg in segments:
        start_s = seg.get("start_ms", 0) / 1000
        end_s = seg.get("end_ms", 0) / 1000
        lines = [f"[Segment {start_s:.1f}s - {end_s:.1f}s]"]

        # Chapter context (from multihop retrieval)
        chapter = seg.get("chapter", "")
        if chapter:
            lines.append(f"Chapter: {chapter}")

        # Speaker info (v1 or v2)
        speakers = seg.get("speakers", [])
        if speakers and isinstance(speakers, list) and isinstance(speakers[0], dict):
            named = [s.get("speaker_name") for s in speakers if s.get("speaker_name")]
            if named:
                lines.append(f"Speaker: {', '.join(set(named))}")
        else:
            speaker = speaker_names.get(seg.get("primary_speaker", ""), "")
            if speaker:
                lines.append(f"Speaker: {speaker}")

        asr = seg.get("asr_transcript", "") or seg.get("asr", "")
        if asr:
            lines.append(f"Speech: {asr[:400]}")
        ocr = seg.get("ocr_text", "") or seg.get("ocr", "")
        if ocr and not (ocr or "").startswith("The image"):
            lines.append(f"On-screen text: {(ocr or '')[:200]}")
        caption = seg.get("visual_caption", "") or seg.get("captions", "")
        if caption:
            lines.append(f"Visual: {caption[:250]}")

        ents = seg.get("named_entities", []) or seg.get("entities", [])
        if ents:
            ent_strs = [f"{e.get('text', '')} ({e.get('type', '?')})" for e in ents[:10]]
            lines.append(f"Entities: {', '.join(ent_strs)}")

        # Grounding (v1: grounded_entities dict, v2: on entity items)
        grounded = seg.get("grounded_entities", {})
        if grounded:
            gr_strs = [f"{k}: {v[:80]}" for k, v in grounded.items() if v][:3]
            if gr_strs:
                lines.append(f"Entity knowledge: {'; '.join(gr_strs)}")
        else:
            gr_strs = [f"{e.get('text', '')}: {e['grounding'][:80]}" for e in ents if e.get("grounding")][:3]
            if gr_strs:
                lines.append(f"Entity knowledge: {'; '.join(gr_strs)}")

        rels = seg.get("relations", [])
        if rels:
            rel_strs = []
            for r in rels[:5]:
                if isinstance(r, dict):
                    rel_strs.append(f"{r.get('subject', '')} -> {r.get('predicate', '')} -> {r.get('object', '')}")
                elif isinstance(r, (list, tuple)) and len(r) >= 3:
                    rel_strs.append(f"{r[0]} -> {r[1]} -> {r[2]}")
            if rel_strs:
                lines.append(f"Relations: {'; '.join(rel_strs)}")
        if lines:
            parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def search_index_multihop(question, index_data, top_k=5):
    """Multi-hop retrieval with entity expansion, chapter context, and cross-layer scoring.

    Steps:
    1. Initial keyword search across all layers (weighted by layer importance)
    2. Entity expansion: find mentioned entities, pull all their speaker turns
    3. Chapter context: include chapter info for each hit
    4. Temporal expansion: merge adjacent hits into wider windows
    5. Re-rank by combined evidence density
    """
    if index_data.get("schema_version", 1) < 2:
        return search_index(question, index_data, top_k)

    layers = index_data.get("layers", {})
    question_lower = question.lower()
    question_words = set(question_lower.split())
    stops = {"the", "a", "an", "is", "was", "what", "how", "did", "does",
             "who", "where", "when", "why", "which", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this",
             "based", "according", "between", "during", "about",
             "represent", "organization", "discuss", "discussing",
             "guest", "speaker", "shown", "appear", "segment"}
    q_words = question_words - stops

    if not q_words:
        return search_index(question, index_data, top_k)

    # Step 1: Score each layer item individually (not per-shot)
    layer_weights = {"ocr": 3, "entities": 2, "asr": 1.5, "visual_captions": 1, "speakers": 2}
    hits = []

    for layer_name, weight in layer_weights.items():
        for item in layers.get(layer_name, {}).get("items", []):
            text = (item.get("text", "") or "").lower()
            text_words = set(text.split())
            overlap = q_words & text_words
            if overlap:
                hits.append({
                    "score": len(overlap) * weight,
                    "start_ms": item.get("start_ms", 0),
                    "end_ms": item.get("end_ms", 0),
                    "layer": layer_name,
                    "matched": overlap,
                })

    if not hits:
        return search_index(question, index_data, top_k)

    # Step 2: Entity expansion -- find matching entities, pull their speaker turns
    entity_items = layers.get("entities", {}).get("items", [])
    speaker_items = layers.get("speakers", {}).get("items", [])

    for ent in entity_items:
        ent_text = (ent.get("text", "") or "").lower()
        ent_words = set(ent_text.split())
        if len(ent_words & q_words) >= max(1, len(ent_words) * 0.5):
            for spk in speaker_items:
                if _overlaps(spk["start_ms"], spk["end_ms"],
                            ent["start_ms"] - 5000, ent["end_ms"] + 5000):
                    hits.append({
                        "score": 3,
                        "start_ms": spk["start_ms"],
                        "end_ms": spk["end_ms"],
                        "layer": "speakers_expanded",
                        "matched": ent_words & q_words,
                    })

    # Step 3: Cluster overlapping hits into time windows (merge within 5s)
    hits.sort(key=lambda h: h["start_ms"])
    clusters = []
    for h in hits:
        merged = False
        for c in clusters:
            if _overlaps(h["start_ms"], h["end_ms"],
                        c["start_ms"] - 5000, c["end_ms"] + 5000):
                c["start_ms"] = min(c["start_ms"], h["start_ms"])
                c["end_ms"] = max(c["end_ms"], h["end_ms"])
                c["score"] += h["score"]
                c["layers"].add(h["layer"])
                c["matched"] |= h["matched"]
                merged = True
                break
        if not merged:
            clusters.append({
                "start_ms": h["start_ms"],
                "end_ms": h["end_ms"],
                "score": h["score"],
                "layers": {h["layer"]},
                "matched": set(h["matched"]),
            })

    # Bonus: multi-layer clusters score higher
    for c in clusters:
        c["score"] *= (1 + 0.3 * len(c["layers"]))

    # Step 4: Add chapter context
    chapters = layers.get("chapters", {}).get("items", [])
    for c in clusters:
        for ch in chapters:
            if _overlaps(c["start_ms"], c["end_ms"], ch["start_ms"], ch["end_ms"]):
                c["chapter"] = ch.get("title", "")
                break

    # Step 5: Rank and build context (cap window to 60s max)
    clusters.sort(key=lambda c: -c["score"])
    results = []
    for c in clusters[:top_k]:
        # Cap window size to prevent context flooding
        start = c["start_ms"]
        end = min(c["end_ms"], start + 60000)
        window = _build_shot_context(
            {"start_ms": start, "end_ms": end},
            layers
        )
        window["chapter"] = c.get("chapter", "")
        results.append(window)

    return results


GROUNDED_ANSWER_PROMPT = """You are answering questions about a broadcast video using evidence from a structured video index. You MUST ground your answer in specific evidence.

Evidence from video index:
{context}

Question: {question}
{mc_section}
Return a JSON object with:
- "answer": your answer to the question{mc_answer_note}
- "cited_segments": list of segment time ranges you used (e.g., ["3.0s-15.0s", "42.0s-55.0s"])
- "speech_evidence": exact quote from the speech transcript that supports your answer (or null)
- "visual_evidence": description from the visual field that supports your answer (or null)
- "ocr_evidence": on-screen text that supports your answer (or null)
- "reasoning": brief explanation of how you combined multiple evidence sources

Answer based ONLY on the evidence provided. For multiple-choice questions, you MUST select one of the provided options (A, B, C, or D) even if the evidence is limited -- pick the best answer. For free-text questions, if the evidence is truly insufficient, say "insufficient evidence"."""


def answer_from_index(question, context, model, mc_options=None):
    """Use LLM to answer question with grounded evidence citations."""
    mc_section = ""
    mc_answer_note = ""
    if mc_options:
        opts = "\n".join(f"  {k}. {v}" for k, v in mc_options.items())
        mc_section = f"\nOptions:\n{opts}\n"
        mc_answer_note = " (for multiple choice, provide BOTH the letter AND the full text)"

    prompt = GROUNDED_ANSWER_PROMPT.format(
        context=context,
        question=question,
        mc_section=mc_section,
        mc_answer_note=mc_answer_note,
    )

    try:
        content = call_llm(prompt, model, temperature=0.3, max_tokens=500)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"answer": content, "cited_segments": [], "reasoning": "parse_error"}
    except Exception as e:
        return {"answer": f"Error: {e}", "cited_segments": [], "reasoning": "api_error"}


def verify_grounding(prediction, index_data):
    """Verify that cited evidence actually exists in the index segments."""
    cited_times = prediction.get("cited_segments", [])
    speech_ev = prediction.get("speech_evidence") or ""
    visual_ev = prediction.get("visual_evidence") or ""
    ocr_ev = prediction.get("ocr_evidence") or ""
    # Ensure string types (LLM sometimes returns lists)
    if not isinstance(speech_ev, str): speech_ev = str(speech_ev)
    if not isinstance(visual_ev, str): visual_ev = str(visual_ev)
    if not isinstance(ocr_ev, str): ocr_ev = str(ocr_ev)

    verification = {
        "segments_cited": len(cited_times),
        "speech_verified": False,
        "visual_verified": False,
        "ocr_verified": False,
        "any_verified": False,
    }

    # Build searchable text from either v1 segments or v2 layers
    if index_data.get("schema_version", 1) >= 2:
        layers = index_data.get("layers", {})
        all_asr = " ".join(i.get("text", "") for i in layers.get("asr", {}).get("items", [])).lower()
        all_cap = " ".join(i.get("text", "") for i in layers.get("visual_captions", {}).get("items", [])).lower()
        all_ocr = " ".join(i.get("text", "") for i in layers.get("ocr", {}).get("items", [])).lower()

        if speech_ev and speech_ev[:40].lower() in all_asr:
            verification["speech_verified"] = True
        if visual_ev and visual_ev[:40].lower() in all_cap:
            verification["visual_verified"] = True
        if ocr_ev and ocr_ev[:25].lower() in all_ocr:
            verification["ocr_verified"] = True
    else:
        segments = index_data.get("segments", [])
        for seg in segments:
            asr = (seg.get("asr_transcript") or "").lower()
            cap = (seg.get("visual_caption") or "").lower()
            ocr = (seg.get("ocr_text") or "").lower()

            if speech_ev and speech_ev[:40].lower() in asr:
                verification["speech_verified"] = True
            if visual_ev and visual_ev[:40].lower() in cap:
                verification["visual_verified"] = True
            if ocr_ev and ocr_ev[:25].lower() in ocr:
                verification["ocr_verified"] = True

    verification["any_verified"] = (
        verification["speech_verified"] or
        verification["visual_verified"] or
        verification["ocr_verified"]
    )

    # Count how many evidence types were provided
    evidence_count = sum([
        bool(speech_ev),
        bool(visual_ev),
        bool(ocr_ev),
    ])
    verification["evidence_types_cited"] = evidence_count
    verification["multi_modal_citation"] = evidence_count >= 2

    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Benchmark JSONL (mc_questions, freetext_questions, or combined)")
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--api", choices=["ollama", "vllm"], default="ollama",
                        help="API backend: ollama or vllm (OpenAI-compatible)")
    parser.add_argument("--retrieval", choices=["keyword", "multihop"], default="keyword",
                        help="Retrieval strategy: keyword (bag-of-words) or multihop (entity expansion + cross-layer)")
    args = parser.parse_args()

    global API_BACKEND, VLLM_URL
    API_BACKEND = args.api

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Pre-load indexes
    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Answering {len(questions)} questions using index ({len(indexes)} videos loaded)...")
    print(f"Model: {args.model}")

    mc_count = sum(1 for q in questions if q.get("format") == "multiple_choice")
    ft_count = sum(1 for q in questions if q.get("format") == "free_text")
    vg_count = sum(1 for q in questions if q.get("format") == "visual_grounded")
    print(f"Format: {mc_count} MC + {ft_count} free-text + {vg_count} visual-grounded")

    stats = {
        "total": 0, "answered": 0, "no_index": 0,
        "grounding_verified": 0, "multi_modal_cited": 0,
    }

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            question_text = q["question"]
            fmt = q.get("format", "free_text")

            # Get expected answer
            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                expected_text = q.get("mc_correct_text", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                expected_text = expected
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                result = {
                    "question_id": q.get("id", f"q-{i}"),
                    "video_id": video_id,
                    "question": question_text,
                    "format": fmt,
                    "expected_answer": expected,
                    "predicted_answer": "Video index not found",
                    "segments_used": 0,
                    "grounding": {},
                    "verification": {},
                }
                fout.write(json.dumps(result) + "\n")
                continue

            # Search and answer
            if args.retrieval == "multihop":
                relevant = search_index_multihop(question_text, idx)
            else:
                relevant = search_index(question_text, idx)
            context = format_context(relevant, idx)
            prediction = answer_from_index(question_text, context, args.model, mc_options)

            # Extract answer for MC
            predicted_answer = prediction.get("answer", "")
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                import re as _re
                p = predicted_answer.strip()
                # Try multiple extraction patterns
                # 1. Starts with letter: "A", "A.", "A) ...", "A. The answer..."
                if p and p[0] in "ABCD" and (len(p) == 1 or p[1] in ".):, "):
                    predicted_answer = p[0]
                else:
                    # 2. Contains "answer is X" or "option X" pattern
                    m = _re.search(r'(?:answer|option|choice)\s*(?:is\s*)?([A-D])\b', p, _re.IGNORECASE)
                    if m:
                        predicted_answer = m.group(1).upper()
                    else:
                        # 3. Any standalone letter A-D in the text
                        m = _re.search(r'\b([A-D])\b', p)
                        if m:
                            predicted_answer = m.group(1).upper()

            # Verify grounding
            verification = verify_grounding(prediction, idx)

            stats["total"] += 1
            stats["answered"] += 1
            if verification["any_verified"]:
                stats["grounding_verified"] += 1
            if verification["multi_modal_citation"]:
                stats["multi_modal_cited"] += 1

            result = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "expected_answer_text": expected_text,
                "predicted_answer": predicted_answer,
                "segments_used": len(relevant),
                "reasoning_type": q.get("reasoning_type", ""),
                "grounding": {
                    "cited_segments": prediction.get("cited_segments", []),
                    "speech_evidence": prediction.get("speech_evidence"),
                    "visual_evidence": prediction.get("visual_evidence"),
                    "ocr_evidence": prediction.get("ocr_evidence"),
                    "reasoning": prediction.get("reasoning", ""),
                },
                "verification": verification,
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                verified_pct = stats["grounding_verified"] / max(1, stats["answered"])
                print(f"  [{i+1}/{len(questions)}] answered, "
                      f"grounding verified: {verified_pct:.0%}", flush=True)

            time.sleep(0.5)

    # Summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"No index: {stats['no_index']}")
    print(f"Grounding verified: {stats['grounding_verified']}/{stats['answered']} "
          f"({stats['grounding_verified']/max(1,stats['answered']):.0%})")
    print(f"Multi-modal citations: {stats['multi_modal_cited']}/{stats['answered']} "
          f"({stats['multi_modal_cited']/max(1,stats['answered']):.0%})")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
