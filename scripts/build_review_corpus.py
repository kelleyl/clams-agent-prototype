#!/usr/bin/env python3
"""Flatten the ENTIRE preview benchmark into reviewable rows with evidence,
with ids identical to convert_v6_to_benchmark.py so verdicts can be applied.

Usage: python scripts/build_review_corpus.py --out /tmp/v6_review_corpus.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")


import hashlib


def _vid_key(vid):
    """Collision-free short video key (12-char tails collide for slug names)."""
    return hashlib.md5((vid or "").encode()).hexdigest()[:10]

QA_DIR = Path("data/qa_needdown")
VIS_DIR = Path("data/qa_visual")
EXP_DIR = Path("data/qa_exploration")

_STOP = set("the a an of to in on for and or but is are was were it this that".split())


def _toks(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return set(w for w in s.split() if len(w) > 2 and w not in _STOP)


def asr_excerpt(doc, start_ms, end_ms, query, answer="", max_chars=1800):
    parts, seen = [], set()
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        lay = doc["layers"][k]
        for it in (lay.get("items", []) if isinstance(lay, dict) else lay):
            if not isinstance(it, dict) or it.get("start_ms") is None:
                continue
            s, e = it["start_ms"], it.get("end_ms") or it["start_ms"]
            if s < end_ms and e > start_ms:
                t = (it.get("text") or "").strip()
                # dedupe near-identical turns from parallel ASR layers
                key = re.sub(r"[^a-z0-9]", "", t.lower())[:60]
                if t and key not in seen:
                    seen.add(key)
                    parts.append((s, t))
    parts.sort()
    if not parts:
        return ""
    q = _toks(query)
    ranked = sorted(range(len(parts)), key=lambda i: -len(q & _toks(parts[i][1])))
    # GUARANTEE the answer-bearing turn survives: put it FIRST with reserved
    # budget (marked with its timestamp), then fill with question-ranked
    # context in time order. Time-ordered assembly under a char cap let intro
    # turns crowd the answer turn out.
    def center_on(text, toks, cap):
        """Slice a long turn AROUND its best-matching sentence, not its head."""
        if len(text) <= cap:
            return text
        sents = re.split(r"(?<=[.!?])\s+", text)
        best = max(range(len(sents)), key=lambda k: len(toks & _toks(sents[k])))
        out, lo, hi = sents[best], best - 1, best + 1
        while lo >= 0 or hi < len(sents):
            grew = False
            if hi < len(sents) and len(out) + len(sents[hi]) < cap:
                out = out + " " + sents[hi]; hi += 1; grew = True
            if lo >= 0 and len(out) + len(sents[lo]) < cap:
                out = sents[lo] + " " + out; lo -= 1; grew = True
            if not grew:
                break
        return ("[...] " if lo >= 0 else "") + out + (" [...]" if hi < len(sents) else "")

    pieces = []
    used = set()
    a = _toks(answer)
    if a:
        a_ranked = sorted(range(len(parts)),
                          key=lambda i: -len(a & _toks(parts[i][1])))
        i = a_ranked[0]
        if a & _toks(parts[i][1]):
            pieces.append(f"[answer context @{parts[i][0] // 60000}m] "
                          + center_on(parts[i][1], a, 700))
            used.add(i)
    budget = max_chars - sum(len(p) for p in pieces)
    ctx = []
    for i in ranked[:3]:
        for j in (i, i - 1, i + 1):
            if 0 <= j < len(parts) and j not in used:
                t = parts[j][1][:350]
                if len(t) <= budget:
                    ctx.append((parts[j][0], t))
                    used.add(j)
                    budget -= len(t)
    ctx.sort()
    if ctx:
        pieces.append(" [...] ".join(t for _, t in ctx))
    return "\n".join(pieces)[:max_chars + 80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", default="qa-data/benchmark/v6_preview/benchmark_combined.jsonl")
    args = ap.parse_args()

    preview_ids = {json.loads(l)["id"] for l in open(args.preview)}
    idx_cache = {}

    def doc_of(vid):
        if vid not in idx_cache:
            p = Path(f"data/video_indexes/{vid}.json")
            idx_cache[vid] = json.load(open(p)) if p.exists() else {}
        return idx_cache[vid]

    n = 0
    with open(args.out, "w") as out:
        # speech
        for f in sorted(QA_DIR.glob("*.json")):
            if f.name == "run_stats.json":
                continue
            d = json.load(open(f))
            vid = d.get("video_id", f.stem)
            for i, r in enumerate(d.get("rows", [])):
                rid = f"v6-{_vid_key(vid)}-{i:03d}"
                if rid not in preview_ids:
                    continue
                qa = r["qa"]
                ev = r.get("evidence", {}) or {}
                excerpt = ""
                if ev.get("start_ms") is not None:
                    excerpt = asr_excerpt(doc_of(vid), ev["start_ms"],
                                          ev.get("end_ms") or ev["start_ms"],
                                          f"{qa['question']} {qa['answer']}",
                                          answer=str(qa["answer"]))
                out.write(json.dumps({
                    "id": rid, "family": "speech", "video_id": vid,
                    "cell": r.get("cell"), "w_role": r.get("w_role"),
                    "question": qa["question"], "answer": qa["answer"],
                    "rationale": qa.get("rationale", "")[:400],
                    "evidence_excerpt": excerpt,
                    "blind_score": qa.get("blind_score"),
                }) + "\n")
                n += 1
        # visual
        for f in sorted(VIS_DIR.glob("*.json")):
            d = json.load(open(f))
            vid = d.get("video_id", f.stem)
            for i, r in enumerate(d.get("rows", [])):
                rid = f"v6v-{_vid_key(vid)}-{i:03d}"
                if rid not in preview_ids:
                    continue
                qa = r["qa"]
                out.write(json.dumps({
                    "id": rid, "family": "visual", "video_id": vid,
                    "cell": r.get("cell"), "element": r.get("element"),
                    "question": qa["question"], "answer": qa["answer"],
                    "rationale": qa.get("rationale", "")[:400],
                    "evidence_excerpt": (r.get("evidence", {}) or {}).get("visual_text", "")[:1200],
                    "blind_score": qa.get("blind_score"),
                }) + "\n")
                n += 1
        # exploration
        for f in sorted(EXP_DIR.glob("*.json")):
            d = json.load(open(f))
            vid = d.get("video_id", f.stem)
            for i, r in enumerate(d.get("rows", [])):
                rid = f"v6x-{_vid_key(vid or 'corpus')}-{i:03d}"
                if rid not in preview_ids:
                    continue
                qa = r["qa"]
                gold = qa.get("answer_set") or []
                # per-item catalog evidence: identity + speaking time +
                # verification method, so a reviewer can actually judge the set
                cat = json.load(open("data/corpus_catalog.json")) \
                    if Path("data/corpus_catalog.json").exists() else {"videos": {}}

                def pinfo(gv, person):
                    for p in (cat["videos"].get(gv) or {}).get("participants", []):
                        g2 = p.get("grounded") or {}
                        if (g2.get("canonical") or p["name"]) == person or p["name"] == person:
                            return g2.get("description") or "(ungrounded)", p.get("speaking_ms", 0)
                    return "(not in catalog)", 0

                lines = []
                for g in gold:
                    if g.get("person"):
                        desc, spk = pinfo(g.get("video_id"), g["person"])
                        lines.append(f"- {g['person']}: {desc} | speaks {spk // 1000}s in "
                                     f"{g['video_id'][:34]} | verified: {g.get('verified', '?')}")
                    else:
                        lines.append(f"- [{g['video_id'][:34]} {g['start_ms'] // 60000}m] "
                                     f"{g.get('title') or ''} :: "
                                     f"{(g.get('excerpt') or g.get('snippet') or '')[:160]}")
                comp = r.get("completeness") or {}
                srt = r.get("set_roundtrip") or {}
                if comp:
                    lines.append("completeness: " + json.dumps(
                        {k: v for k, v in comp.items() if k != "near_miss_checked"}))
                if srt.get("f1") is not None:
                    lines.append(f"set round-trip F1: {srt['f1']}")
                out.write(json.dumps({
                    "id": rid, "family": "exploration", "video_id": vid,
                    "question": qa.get("question"),
                    "answer": " ;; ".join(
                        f"[{g['video_id'][:30]} {g['start_ms'] // 60000}m] "
                        f"{(g.get('person') or g.get('title') or '')}"
                        for g in gold),
                    "rationale": qa.get("rationale", ""),
                    "evidence_excerpt": "\n".join(lines),
                }) + "\n")
                n += 1
    print(f"wrote {n} review rows -> {args.out} (preview has {len(preview_ids)})")


if __name__ == "__main__":
    main()
