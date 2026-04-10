#!/usr/bin/env python3
"""Run ablation studies by removing modalities from the video index.

Tests what happens when specific information sources are removed:
- no_asr: Remove all speech transcripts
- no_visual: Remove all visual captions
- no_ocr: Remove all OCR text
- no_entities: Remove all named entities and grounding
- no_relations: Remove all relations
- asr_only: Keep only speech transcripts
- visual_only: Keep only visual captions

Usage:
    python eval/run_ablation.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --ablation no_asr \
        --output eval/results/ablation_no_asr.jsonl \
        --model qwen3:8b
"""
import argparse
import copy
import json
import os
import sys
import time
import requests
from pathlib import Path

# Import from run_index_qa
sys.path.insert(0, str(Path(__file__).parent))
from run_index_qa import search_index, format_context, answer_from_index, verify_grounding
import run_index_qa

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ABLATION_TYPES = {
    "no_index": "No index at all (blind baseline)",
    "no_asr": "Remove all speech transcripts",
    "no_visual": "Remove all visual captions",
    "no_ocr": "Remove all OCR text",
    "no_entities": "Remove named entities and grounding",
    "no_relations": "Remove relations",
    "asr_only": "Keep only speech transcripts (remove visual, OCR, entities)",
    "visual_only": "Keep only visual captions (remove ASR, OCR, entities)",
}


def ablate_index(index_data, ablation_type):
    """Create a copy of the index with specified modality removed.

    Supports both v1 (flat segments) and v2 (layered) indexes.
    """
    ablated = copy.deepcopy(index_data)

    if ablation_type == "no_index":
        # Remove everything — blind baseline
        if "layers" in ablated:
            for name in list(ablated.get("layers", {}).keys()):
                ablated["layers"][name] = {"items": []}
        if "segments" in ablated:
            for seg in ablated["segments"]:
                for key in ("asr_transcript", "ocr_text", "visual_caption"):
                    seg[key] = ""
                seg["named_entities"] = []
                seg["grounded_entities"] = {}
                seg["relations"] = []
        ablated["speaker_names"] = {}
        ablated["speaker_details"] = {}
        return ablated

    if ablated.get("schema_version", 1) >= 2:
        # v2: remove entire layers
        layers = ablated.get("layers", {})
        if ablation_type == "no_asr":
            layers.pop("asr", None)
        elif ablation_type == "no_visual":
            layers.pop("visual_captions", None)
        elif ablation_type == "no_ocr":
            layers.pop("ocr", None)
        elif ablation_type == "no_entities":
            layers.pop("entities", None)
        elif ablation_type == "no_relations":
            layers.pop("relations", None)
        elif ablation_type == "asr_only":
            for name in list(layers.keys()):
                if name not in ("asr", "shots", "scenes"):
                    layers.pop(name, None)
        elif ablation_type == "visual_only":
            for name in list(layers.keys()):
                if name not in ("visual_captions", "shots", "scenes"):
                    layers.pop(name, None)

        # Clear speaker info for asr-removed ablations
        if ablation_type in ("no_asr", "visual_only"):
            layers.pop("speakers", None)
            layers.pop("chapters", None)
    else:
        # v1: clear segment fields
        for seg in ablated.get("segments", []):
            if ablation_type == "no_asr":
                seg["asr_transcript"] = ""
            elif ablation_type == "no_visual":
                seg["visual_caption"] = ""
            elif ablation_type == "no_ocr":
                seg["ocr_text"] = ""
            elif ablation_type == "no_entities":
                seg["named_entities"] = []
                seg["grounded_entities"] = {}
            elif ablation_type == "no_relations":
                seg["relations"] = []
            elif ablation_type == "asr_only":
                seg["visual_caption"] = ""
                seg["ocr_text"] = ""
                seg["named_entities"] = []
                seg["grounded_entities"] = {}
                seg["relations"] = []
            elif ablation_type == "visual_only":
                seg["asr_transcript"] = ""
                seg["ocr_text"] = ""
                seg["named_entities"] = []
                seg["grounded_entities"] = {}
                seg["relations"] = []

        if ablation_type in ("no_asr", "visual_only"):
            ablated["speaker_names"] = {}
            ablated["speaker_details"] = {}

    return ablated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--ablation", type=str, required=True,
                        choices=list(ABLATION_TYPES.keys()))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--api", choices=["ollama", "vllm"], default="ollama",
                        help="API backend: ollama or vllm (OpenAI-compatible)")
    args = parser.parse_args()

    # Set API backend in run_index_qa module
    run_index_qa.API_BACKEND = args.api

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Pre-load and ablate indexes
    indexes = {}
    for f in args.index_dir.glob("*.json"):
        raw = json.load(open(f))
        indexes[f.stem] = ablate_index(raw, args.ablation)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"ABLATION: {args.ablation} — {ABLATION_TYPES[args.ablation]}")
    print(f"Answering {len(questions)} questions ({len(indexes)} videos loaded)...")

    stats = {"total": 0, "answered": 0, "grounding_verified": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            question_text = q["question"]
            fmt = q.get("format", "free_text")

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                continue

            relevant = search_index(question_text, idx)
            context = format_context(relevant, idx)
            prediction = answer_from_index(question_text, context, args.model, mc_options)

            predicted_answer = prediction.get("answer", "")
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                for letter in "ABCD":
                    if predicted_answer.strip().startswith(letter):
                        predicted_answer = letter
                        break

            verification = verify_grounding(prediction, idx)

            stats["total"] += 1
            stats["answered"] += 1
            if verification["any_verified"]:
                stats["grounding_verified"] += 1

            result = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted_answer,
                "ablation": args.ablation,
                "grounding": {
                    "cited_segments": prediction.get("cited_segments", []),
                    "speech_evidence": prediction.get("speech_evidence"),
                    "visual_evidence": prediction.get("visual_evidence"),
                    "ocr_evidence": prediction.get("ocr_evidence"),
                    "reasoning": prediction.get("reasoning", ""),
                },
                "verification": verification,
                "reasoning_type": q.get("reasoning_type", ""),
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(questions)}] answered", flush=True)

            time.sleep(0.5)

    print(f"\nDone. {stats['total']} answered, "
          f"{stats['grounding_verified']}/{stats['answered']} grounding verified")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
