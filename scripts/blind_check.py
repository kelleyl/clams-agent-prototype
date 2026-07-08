#!/usr/bin/env python3
"""Gate 1 (necessity), free-text: can a no-video model PRODUCE the answer?

Reads a need-down QA sidecar (data/qa_needdown/<vid>.json). For each question
it asks a different-family model to answer with NO video and NO options (pure
free text), then matches against the gold answer. A question is
"blind-answerable" (leaky) if the blind answer matches. With free text there are
no distractors to eliminate, so this is a real necessity signal, unlike MC.

Runs on aristotle; use a strong model of a different family than the generator.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import requests
from utils.answer_match import decide_match

QA_DIR = Path("data/qa_needdown")


def ask_blind(question, url, model, key, timeout=90):
    user = ("Answer this question from general knowledge. You have NOT seen any video and there "
            "are no answer options. Give a short answer only; if you cannot, reply DON'T KNOW. "
            f"/no_think\n\nQuestion: {question}\nAnswer:")
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0, "max_tokens": 80,
                                "messages": [{"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
    except Exception as ex:
        return f"__ERR__:{ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--blind-url", default="http://localhost:11434/v1")
    ap.add_argument("--blind-model", default="qwen3:30b")
    ap.add_argument("--judge-model", default="llama3.3:latest",
                    help="non-thinking model used to adjudicate fuzzy answer matches")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    judge_cfg = {"url": args.blind_url, "model": args.judge_model, "api_key": args.api_key}
    path = QA_DIR / f"{args.video}.json"
    data = json.load(open(path))

    n = leaky = err = 0
    for r in data.get("rows", []):
        qa = r.get("qa", {})
        q, gold = qa.get("question"), qa.get("answer")
        if not (q and gold) or qa.get("error"):
            continue
        ba = ask_blind(q, args.blind_url, args.blind_model, args.api_key)
        if ba.startswith("__ERR__"):
            r["blind"] = {"error": ba}
            err += 1
            continue
        is_leaky = "don't know" not in ba.lower() and decide_match(q, ba, gold, judge_cfg)
        r["blind"] = {"model": args.blind_model, "blind_answer": ba, "blind_answerable": is_leaky}
        n += 1
        leaky += 1 if is_leaky else 0
        if args.verbose and is_leaky:
            print(f"  LEAKY [{r['cell']}/{r['w_role']}] {q[:70]}")
            print(f"        gold: {gold[:40]} | blind: {ba[:40]}")

    json.dump(data, open(path, "w"), indent=2)
    print(f"\nvideo: {args.video} | blind model: {args.blind_model} (free-text)")
    print(f"questions: {n} | blind-answerable (leaky): {leaky} ({leaky/n:.0%} of {n}) | errors: {err}")
    print(f"necessity-PASS (blind could not answer): {n - leaky}")


if __name__ == "__main__":
    main()
