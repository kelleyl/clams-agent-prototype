#!/usr/bin/env python3
"""Find all evidence paths to an answer across index layers.

For each QA pair, discovers which layers contain the answer (or supporting
evidence), enabling multi-trajectory generation with different cost/preference
tradeoffs.

Usage:
    python training_data/find_evidence_paths.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/evidence_paths.jsonl
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.video_index import overlaps


def normalize(text):
    """Normalize text for fuzzy matching."""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()


def find_answer_in_layer(answer_text, layer_items, layer_name, threshold=0.4):
    """Find items in a layer that contain the answer or key parts of it.

    Returns list of (item, match_score, matched_terms) tuples.
    """
    answer_norm = normalize(answer_text)
    answer_words = set(answer_norm.split())
    # Extract key terms (skip very common words)
    stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this", "it",
             "by", "as", "from", "be", "has", "had", "not", "but", "they",
             "which", "their", "can", "all", "would", "there", "been"}
    key_terms = answer_words - stops
    if not key_terms:
        key_terms = answer_words

    matches = []
    for item in layer_items:
        # Get all searchable text from the item
        item_texts = []
        for k, v in item.items():
            if isinstance(v, str) and k not in ("id", "origin_layer", "source_layer", "source_id"):
                item_texts.append(v)
        # Also check nested entries (credits)
        for entry in item.get("entries", []):
            for k, v in entry.items():
                if isinstance(v, str):
                    item_texts.append(v)

        item_text_norm = normalize(" ".join(item_texts))
        item_words = set(item_text_norm.split())

        # Score: what fraction of answer key terms appear in this item?
        matched = key_terms & item_words
        if not matched:
            continue
        score = len(matched) / len(key_terms)

        if score >= threshold:
            matches.append({
                "item_id": item.get("id", ""),
                "start_ms": item.get("start_ms", 0),
                "end_ms": item.get("end_ms", 0),
                "score": round(score, 3),
                "matched_terms": sorted(matched),
                "text_preview": " ".join(item_texts)[:200],
                "layer": layer_name,
            })

    return sorted(matches, key=lambda x: -x["score"])


def find_all_evidence_paths(question_data, index_data):
    """Find all ways the answer can be reached through different layers.

    Returns a dict with:
    - paths: list of evidence paths grouped by layer
    - redundancy: how many layers independently contain the answer
    - preferred_path: shortest path (fewest tool calls)
    - time_ranges: time ranges where evidence clusters
    """
    answer = question_data.get("mc_correct_text",
              question_data.get("reference_answer",
              question_data.get("answer", "")))

    if not answer or len(answer) < 3:
        return None

    layers = index_data.get("layers", {})
    grounding = question_data.get("grounding", {})

    # Also extract key entities/terms from the question itself
    question = question_data.get("question", "")

    all_paths = {}
    for layer_name in ["asr", "ocr", "visual_captions", "speakers", "entities", "credits", "chapters"]:
        layer = layers.get(layer_name, {})
        items = layer.get("items", [])
        if not items:
            continue

        matches = find_answer_in_layer(answer, items, layer_name)
        if matches:
            all_paths[layer_name] = matches[:5]  # top 5 per layer

    if not all_paths:
        return None

    # Analyze redundancy
    redundancy = len(all_paths)

    # Find time ranges where evidence clusters
    all_times = []
    for layer_name, matches in all_paths.items():
        for m in matches:
            all_times.append((m["start_ms"], m["end_ms"], layer_name))

    # Cluster overlapping evidence
    clusters = []
    for start, end, layer in sorted(all_times):
        merged = False
        for cluster in clusters:
            if overlaps(start, end, cluster["start_ms"], cluster["end_ms"]):
                cluster["start_ms"] = min(cluster["start_ms"], start)
                cluster["end_ms"] = max(cluster["end_ms"], end)
                cluster["layers"].add(layer)
                merged = True
                break
        if not merged:
            clusters.append({
                "start_ms": start,
                "end_ms": end,
                "layers": {layer},
            })

    # Convert sets to lists for JSON
    for c in clusters:
        c["layers"] = sorted(c["layers"])
        c["n_layers"] = len(c["layers"])

    # Determine preferred paths by preference mode
    modes = {}

    # Minimal: single best match from any layer
    best_match = None
    best_score = 0
    for layer_name, matches in all_paths.items():
        if matches and matches[0]["score"] > best_score:
            best_score = matches[0]["score"]
            best_match = matches[0]
    modes["minimal"] = {
        "tool_calls": 1,
        "primary_layer": best_match["layer"] if best_match else None,
        "evidence": [best_match] if best_match else [],
    }

    # Indexed: use whatever layer has the best match (assumes index exists)
    modes["indexed"] = {
        "tool_calls": min(redundancy, 2),  # search + maybe verify
        "primary_layer": best_match["layer"] if best_match else None,
        "evidence": [m[0] for m in [all_paths.get(l, []) for l in all_paths] if m][:2],
    }

    # Exhaustive: check all layers that have evidence
    all_evidence = []
    for matches in all_paths.values():
        all_evidence.extend(matches[:2])
    modes["exhaustive"] = {
        "tool_calls": redundancy,
        "primary_layer": best_match["layer"] if best_match else None,
        "evidence": all_evidence,
    }

    # From scratch: what CLAMS tools would you need to run?
    tool_mapping = {
        "asr": ["ffmpeg_extract_audio", "run_parakeet"],
        "ocr": ["ffmpeg_extract_frames", "run_swt", "run_qwen_ocr"],
        "visual_captions": ["ffmpeg_extract_frames", "run_transnet", "run_qwen_caption"],
        "speakers": ["ffmpeg_extract_audio", "run_diarization"],
        "entities": ["run_spacy_ner"],  # requires ASR/OCR first
        "credits": ["ffmpeg_extract_frames", "run_swt", "run_credits_ocr"],
        "chapters": ["ffmpeg_extract_audio", "run_diarization"],
    }
    from_scratch_tools = set()
    if best_match:
        from_scratch_tools.update(tool_mapping.get(best_match["layer"], []))
    modes["from_scratch"] = {
        "tool_calls": len(from_scratch_tools),
        "tools_needed": sorted(from_scratch_tools),
        "primary_layer": best_match["layer"] if best_match else None,
        "evidence": [best_match] if best_match else [],
    }

    return {
        "paths": all_paths,
        "redundancy": redundancy,
        "clusters": clusters,
        "modes": modes,
        "answer": answer[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, default=Path("training_data/output/evidence_paths.jsonl"))
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for idx_file in args.index_dir.glob("*.json"):
        indexes[idx_file.stem] = json.load(open(idx_file))

    print(f"Finding evidence paths for {len(questions)} questions...")

    stats = defaultdict(int)
    redundancy_dist = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            idx_data = indexes.get(video_id)
            if not idx_data:
                stats["no_index"] += 1
                continue

            result = find_all_evidence_paths(q, idx_data)
            if result is None:
                stats["no_paths"] += 1
                continue

            stats["found"] += 1
            redundancy_dist[result["redundancy"]] += 1

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": q["question"],
                "format": q.get("format", "free_text"),
                **result,
            }
            fout.write(json.dumps(output, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(questions)}] found: {stats['found']}, "
                      f"no_paths: {stats['no_paths']}", flush=True)

    print(f"\nDone: {stats['found']} with paths, {stats['no_paths']} no paths, "
          f"{stats.get('no_index', 0)} no index")
    print(f"\nRedundancy distribution (how many layers contain the answer):")
    for r in sorted(redundancy_dist):
        bar = "#" * redundancy_dist[r]
        print(f"  {r} layers: {redundancy_dist[r]:4d} {bar}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
