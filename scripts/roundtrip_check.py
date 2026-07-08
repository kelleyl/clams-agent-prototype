#!/usr/bin/env python3
"""Round-trip answerability gate (first v6 QC gate).

Reads a need-down QA sidecar (data/qa_needdown/<vid>.json). For each
segment-anchored question it RE-ANSWERS the question from the evidence
transcript using a DIFFERENT model family (no access to the proposed answer),
then keeps the item only if the fresh answer matches the proposed answer.

This catches answers that do not actually answer the question (e.g. a "why"
question answered with a count) and unanswerable items. It does NOT judge the
proposed answer directly (that would invite self-agreement); it re-derives it.

Runs on aristotle. Different model from the generator, by design.

Example:
  python scripts/roundtrip_check.py --video cpb-aacip-507-3x83j39n00 \
      --rt-url http://localhost:11434/v1 --rt-model qwen3:30b
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import requests
from utils.ctx_retrieval import retrieve_context
from utils.answer_match import decide_match

QA_DIR = Path("data/qa_needdown")
IDX_DIR = Path("data/video_indexes")
DESC_DIR = Path("data/video_descriptions")
STOP = set("the a an of to in on for and or but is are was were be been it this that with as at "
           "by from his her their its he she they who what why how which when where".split())


def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def asr_text(doc, start_ms, end_ms, max_chars=1500):
    parts = []
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        for it in items(doc["layers"][k]):
            if not isinstance(it, dict):
                continue
            s, e = it.get("start_ms"), it.get("end_ms")
            if s is None:
                continue
            if s < end_ms and (e or s) > start_ms:
                t = (it.get("text") or "").strip()
                if t:
                    parts.append((s, t))
    parts.sort()
    return " ".join(t for _, t in parts)[:max_chars]


def desc_window(desc, start_ms, end_ms, max_chars=2500):
    """DP prose of the description beats overlapping [start,end] (disentangle test)."""
    out = []
    for b in desc.get("beats", []):
        s, e = b.get("start_ms"), b.get("end_ms") or b.get("start_ms")
        if s is None:
            continue
        if s < end_ms and e > start_ms:
            out.append((b.get("prose") or b.get("scaffold") or "").strip())
    return "\n".join(t for t in out if t)[:max_chars]


def content_tokens(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower().replace(",", ""))
    return set(w for w in s.split() if w and w not in STOP)


def normstr(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def consistent(proposed, fresh):
    if not fresh or "unanswerable" in fresh.lower():
        return False
    np_, nf = normstr(proposed), normstr(fresh)
    if np_ and (np_ in nf or nf in np_):
        return True
    p, f = content_tokens(proposed), content_tokens(fresh)
    if not p:
        return False
    return len(p & f) / len(p) >= 0.5


def reanswer(question, ctx, url, model, api_key, timeout=90):
    user = ("Using ONLY the transcript below, answer the question in a few words. "
            "If the transcript does not answer it, reply UNANSWERABLE. /no_think\n\n"
            f"Transcript:\n{ctx}\n\nQuestion: {question}\nAnswer:")
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0, "max_tokens": 256,
                                "messages": [{"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        if "</think>" in txt:
            txt = txt.split("</think>")[-1]
        return txt.strip()
    except Exception as ex:
        return f"__ERR__:{ex}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--rt-url", default="http://localhost:11434/v1")
    ap.add_argument("--rt-model", default="qwen3:30b")
    ap.add_argument("--judge-model", default="llama3.3:latest",
                    help="non-thinking model used to adjudicate fuzzy answer matches")
    ap.add_argument("--evidence-source", choices=["index", "description"], default="index",
                    help="index=raw ASR window + lexical retrieval; description=DP prose window")
    ap.add_argument("--rt-key", default="roundtrip",
                    help="key to store the verdict under (use a distinct key to compare sources)")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    judge_cfg = {"url": args.rt_url, "model": args.judge_model, "api_key": args.api_key}

    data = json.load(open(QA_DIR / f"{args.video}.json"))
    doc = json.load(open(IDX_DIR / f"{args.video}.json"))
    desc = None
    if args.evidence_source == "description":
        dp = DESC_DIR / f"{args.video}.json"
        if not dp.exists():
            print(f"{args.video}: no description, cannot use description source", file=sys.stderr)
            sys.exit(2)
        desc = json.load(open(dp))
    rows = data.get("rows", [])

    checked = kept = rej = skipped = err = 0
    for r in rows:
        qa = r.get("qa", {})
        q, a = qa.get("question"), qa.get("answer")
        ev = r.get("evidence", {})
        if not (q and a) or qa.get("error"):
            continue
        if ev.get("start_ms") is None:          # two-hop etc.: no transcript span
            r[args.rt_key] = {"skipped": "no_span"}
            skipped += 1
            continue
        if desc is not None:
            ctx = desc_window(desc, ev["start_ms"], ev["end_ms"])
        else:
            ctx = (asr_text(doc, ev["start_ms"], ev["end_ms"]) + "\n"
                   + retrieve_context(doc, q, k=5))[:2500]
        fresh = reanswer(q, ctx, args.rt_url, args.rt_model, args.api_key)
        if fresh.startswith("__ERR__"):
            r[args.rt_key] = {"error": fresh}
            err += 1
            continue
        ok = "unanswerable" not in fresh.lower() and decide_match(q, fresh, a, judge_cfg)
        r[args.rt_key] = {"model": args.rt_model, "fresh_answer": fresh,
                          "source": args.evidence_source, "consistent": ok}
        checked += 1
        kept += 1 if ok else 0
        rej += 0 if ok else 1
        if args.verbose and not ok:
            print(f"  REJECT [{r['cell']}/{r['w_role']}] Q: {q[:70]}")
            print(f"          proposed: {str(a)[:50]} | re-answer: {fresh[:50]}")

    json.dump(data, open(QA_DIR / f"{args.video}.json", "w"), indent=2)
    print(f"\nvideo: {args.video} | round-trip model: {args.rt_model}")
    print(f"checked: {checked} | consistent(keep): {kept} | inconsistent(reject): {rej} "
          f"| skipped(no span): {skipped} | errors: {err}")
    if checked:
        print(f"pass rate: {kept/checked:.0%}")


if __name__ == "__main__":
    main()
