#!/usr/bin/env python3
"""LLM question-quality judging over the v6 release candidate (idle-GPU pass).

Each judge model scores every question 1-5 on four dimensions, given the
question, gold answer, and evidence excerpt:
  clarity        - self-contained, unambiguous phrasing
  usefulness     - would an archivist/researcher plausibly ask this
  answer_support - the gold answer is correct and unambiguous given the evidence
  evidence_fit   - the evidence is sufficient to answer without guessing

Resumable (skips ids already in the output). Model-outer would need one file
per judge anyway, so run once per judge:
  python scripts/judge_question_quality_v6.py --judge qwen3.6:35b-a3b \
      --corpus /tmp/v6_review_corpus.jsonl --out data/qa_judge/v6_judge_qwen36.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import requests

PROMPT = (
    "Rate this benchmark question about an archival US broadcast. Score each dimension "
    "1 (bad) to 5 (excellent):\n"
    "- clarity: self-contained and unambiguous for a reader who has not seen the video\n"
    "- usefulness: a researcher or archivist would plausibly ask this\n"
    "- answer_support: the gold answer is correct and the ONLY defensible answer given "
    "the evidence\n"
    "- evidence_fit: the evidence suffices to answer without guessing\n\n"
    "Question: {q}\nGold answer: {a}\nEvidence:\n{e}\n\n"
    'Reply with ONLY JSON: {{"clarity": n, "usefulness": n, "answer_support": n, '
    '"evidence_fit": n}}. /no_think'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--url", default="http://localhost:11434/v1")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if Path(args.out).exists():
        for line in open(args.out):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    rows = [json.loads(l) for l in open(args.corpus)]
    n = 0
    with open(args.out, "a", buffering=1) as out:
        for r in rows:
            if r["id"] in done:
                continue
            prompt = PROMPT.format(q=r["question"], a=str(r["answer"])[:300],
                                   e=(r.get("evidence_excerpt") or "")[:2000])
            try:
                resp = requests.post(f"{args.url}/chat/completions",
                                     json={"model": args.judge, "temperature": 0,
                                           "max_tokens": 128,
                                           "messages": [{"role": "user", "content": prompt}]},
                                     timeout=180)
                txt = resp.json()["choices"][0]["message"]["content"]
                txt = txt.split("</think>")[-1]
                m = re.search(r"\{.*\}", txt, re.S)
                scores = json.loads(m.group(0)) if m else {}
            except Exception as ex:
                scores = {"error": str(ex)[:100]}
            out.write(json.dumps({"id": r["id"], "family": r.get("family"),
                                  "judge": args.judge, "scores": scores}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n} judged", flush=True)
    print(f"done: {n} new rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
