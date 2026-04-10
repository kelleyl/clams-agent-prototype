#!/usr/bin/env python3
"""Score question quality on 4 dimensions using LLM-as-judge.

Dimensions:
1. Cognitive level (1-5): Does it require reasoning, not just lookup?
2. Naturalness (1-5): Does it sound like a human would ask this?
3. Groundedness (1-5): Is the answer clearly supported by the provided grounding?
4. Information need (1-5): Does it require combining multiple information sources?

Usage:
    python qa-data/score_question_quality.py \
        --questions qa-data/raw/llm_comprehension_qa_v2_valid.jsonl \
        --output qa-data/raw/llm_comprehension_qa_v2_scored.jsonl \
        --model qwen3:8b
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

JUDGE_PROMPT = """Rate this video comprehension question on 4 dimensions (1-5 scale).

Question: {question}
Answer: {answer}
Modalities required: {modalities}
Grounding:
  Speech excerpt: {speech}
  Visual reference: {visual}
  OCR reference: {ocr}

Rate each dimension:

1. COGNITIVE_LEVEL (1-5):
   1 = Pure lookup (who/what/when with answer directly stated)
   3 = Requires connecting two pieces of information
   5 = Requires inference, comparison, or causal reasoning

2. NATURALNESS (1-5):
   1 = Awkward, overly technical, or references annotation layers
   3 = Acceptable but stilted
   5 = Sounds like a natural human question about a video

3. GROUNDEDNESS (1-5):
   1 = Answer not supported by the provided grounding excerpts
   3 = Partially supported, some inference needed
   5 = Answer clearly and directly supported by grounding

4. CROSS_MODAL_NEED (1-5):
   1 = Answerable from a single source (just speech OR just visual)
   3 = Benefits from multiple sources but one could suffice
   5 = Genuinely requires combining information across modalities

Return JSON: {{"cognitive_level": N, "naturalness": N, "groundedness": N, "cross_modal_need": N, "brief_rationale": "..."}}"""


def call_llm(prompt, model, temperature=0.1, max_tokens=300):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def score_question(q, model):
    grounding = q.get("grounding", {})
    prompt = JUDGE_PROMPT.format(
        question=q["question"],
        answer=q.get("answer", ""),
        modalities=q.get("modalities_required", []),
        speech=grounding.get("speech_excerpt", "none")[:200] if grounding.get("speech_excerpt") else "none",
        visual=grounding.get("visual_reference", "none")[:200] if grounding.get("visual_reference") else "none",
        ocr=grounding.get("ocr_reference", "none")[:200] if grounding.get("ocr_reference") else "none",
    )

    try:
        content = call_llm(prompt, model)
        scores = json.loads(content)
        return {
            "cognitive_level": int(scores.get("cognitive_level", 0)),
            "naturalness": int(scores.get("naturalness", 0)),
            "groundedness": int(scores.get("groundedness", 0)),
            "cross_modal_need": int(scores.get("cross_modal_need", 0)),
            "brief_rationale": scores.get("brief_rationale", ""),
        }
    except (json.JSONDecodeError, requests.RequestException, ValueError) as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Scoring {len(questions)} questions with {args.model}...")

    dims = defaultdict(list)
    errors = 0

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            scores = score_question(q, args.model)
            q["quality_scores"] = scores
            fout.write(json.dumps(q, ensure_ascii=False) + "\n")
            fout.flush()

            if "error" in scores:
                errors += 1
            else:
                for dim in ["cognitive_level", "naturalness", "groundedness", "cross_modal_need"]:
                    dims[dim].append(scores[dim])

            if (i + 1) % 50 == 0:
                avgs = {d: sum(v)/len(v) for d, v in dims.items() if v}
                print(f"  [{i+1}/{len(questions)}] avgs: {', '.join(f'{d}={v:.1f}' for d,v in avgs.items())}")

            time.sleep(0.3)

    print(f"\nDone. Errors: {errors}")
    print(f"\nDimension averages ({len(dims.get('cognitive_level', []))} scored):")
    for dim in ["cognitive_level", "naturalness", "groundedness", "cross_modal_need"]:
        vals = dims[dim]
        if vals:
            print(f"  {dim}: mean={sum(vals)/len(vals):.2f}, min={min(vals)}, max={max(vals)}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
