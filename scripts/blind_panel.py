#!/usr/bin/env python3
"""Multi-model blind panel: parametric-answerability profile per question.

Instead of a single hard "is this blind-answerable" gate, this runs a PANEL of
models (sizes x families) on each question with NO video, scores each against
the gold answer, and records a profile: which models answer it blind, and an
aggregate blind_score (fraction of the panel). This turns world-knowledge
leakage into graded dataset metadata (stratification) rather than a delete.

Iterates model-by-model (outer loop) so each model loads once on the server.

Runs on aristotle. Ollama models on :11434, the gemma-4 vLLM on :8202.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import requests
from utils.answer_match import match

QA_DIR = Path("data/qa_needdown")
OLL = "http://localhost:11434/v1"
VLLM4 = "http://localhost:8202/v1"

PANEL = [
    {"name": "llama3.1:8b",       "family": "Llama",  "size": 8,  "url": OLL},
    {"name": "qwen3:8b",          "family": "Qwen",   "size": 8,  "url": OLL},
    {"name": "gemma3:12b-it-qat", "family": "Gemma",  "size": 12, "url": OLL},
    {"name": "gemma3:27b-it-qat", "family": "Gemma",  "size": 27, "url": OLL},
    {"name": "qwen3:30b",         "family": "Qwen",   "size": 30, "url": OLL},
    {"name": "gemma-4-26b",       "family": "Gemma4", "size": 26, "url": VLLM4},
    {"name": "llama3.1:70b",      "family": "Llama",  "size": 70, "url": OLL},
    {"name": "llama3.3:latest",   "family": "Llama",  "size": 70, "url": OLL},
]


def ask_blind(question, url, model, api_key, timeout=120):
    user = ("Answer this question from general knowledge. You have NOT seen any video and there "
            "are no answer options. Give a short answer only; if you cannot, reply DON'T KNOW. "
            f"/no_think\n\nQuestion: {question}\nAnswer:")
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0, "max_tokens": 512,
                                "messages": [{"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
    except Exception as ex:
        return f"__ERR__:{ex}"


def tier(size):
    return "small" if size < 15 else ("medium" if size <= 40 else "large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+", required=True)
    ap.add_argument("--models", default="", help="comma list to subset the panel")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    want = set(x.strip() for x in args.models.split(",") if x.strip())
    panel = [m for m in PANEL if not want or m["name"] in want]

    # load all videos; keep (vid, data) for write-back and a flat row list
    videos = []
    rows = []
    for vid in args.video:
        data = json.load(open(QA_DIR / f"{vid}.json"))
        videos.append((vid, data))
        for r in data.get("rows", []):
            if r["qa"].get("question") and not r["qa"].get("error"):
                rows.append(r)
    print(f"videos={len(videos)} questions={len(rows)}", flush=True)

    per_model = {}  # name -> (correct, n)
    for m in panel:                      # model OUTER loop -> each model loads once
        correct = 0
        n = 0
        for r in rows:
            # resume: skip rows this model already voted on
            existing = r["qa"].get("blind_panel", {})
            if m["name"] in existing:
                n += 1
                correct += 1 if existing[m["name"]] else 0
                continue
            q, gold = r["qa"]["question"], r["qa"]["answer"]
            ba = ask_blind(q, m["url"], m["name"], args.api_key)
            ok = (not ba.startswith("__ERR__")) and "don't know" not in ba.lower() and match(ba, gold)
            r["qa"].setdefault("blind_panel", {})[m["name"]] = bool(ok)
            n += 1
            correct += 1 if ok else 0
        per_model[m["name"]] = (correct, n)
        # persist after each model so an interrupted panel resumes cleanly
        for vid, data in videos:
            json.dump(data, open(QA_DIR / f"{vid}.json", "w"), indent=2)
        print(f"  {m['name']:20s} ({m['family']}/{m['size']}B): blind {correct}/{n} = {correct/max(n,1):.0%}",
              flush=True)

    # aggregate per-question blind_score + write each video back
    for r in rows:
        bp = r["qa"].get("blind_panel", {})
        r["qa"]["blind_score"] = round(sum(bp.values()) / len(bp), 3) if bp else None
    for vid, data in videos:
        json.dump(data, open(QA_DIR / f"{vid}.json", "w"), indent=2)

    print("\n=== by size tier ===")
    tiers = defaultdict(lambda: [0, 0])
    fams = defaultdict(lambda: [0, 0])
    for m in panel:
        c, n = per_model[m["name"]]
        tiers[tier(m["size"])][0] += c
        tiers[tier(m["size"])][1] += n
        fams[m["family"]][0] += c
        fams[m["family"]][1] += n
    for t in ("small", "medium", "large"):
        c, n = tiers[t]
        if n:
            print(f"  {t:7s}: {c/n:.0%} blind-answerable")
    print("=== by family ===")
    for f, (c, n) in fams.items():
        print(f"  {f:7s}: {c/n:.0%}")

    print("\n=== per-question strata (blind_score) ===")
    strat = defaultdict(int)
    for r in rows:
        s = r["qa"]["blind_score"]
        k = ("robust(0 models)" if s == 0 else "leaky-to-few(<=.33)" if s <= 0.33
             else "leaky-to-some(<=.66)" if s <= 0.66 else "trivial(>.66)")
        strat[k] += 1
    for k in ("robust(0 models)", "leaky-to-few(<=.33)", "leaky-to-some(<=.66)", "trivial(>.66)"):
        print(f"  {k:22s}: {strat[k]}")
    print("\noutput -> data/qa_needdown/")


if __name__ == "__main__":
    main()
