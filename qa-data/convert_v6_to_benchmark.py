#!/usr/bin/env python3
"""Package a v6 run into a benchmark jsonl.

Reads data/qa_needdown/ (need-down free-text questions with gate state) and
data/qa_exploration/ (retrieval_set questions), applies the keep policy, and
writes qa-data/benchmark/v6/benchmark_combined.jsonl.

Keep policy (graded-gate philosophy from V6_DATASET_PLAN.md, tightened after
the 2026-07-08 pilot smell test):
  need-down   : requires a VERIFIED evidence span - round-trip pass, not
                skipped. No-span rows (two-hop, cataloging) bypass the gate and
                the smell test showed they are exactly where hallucinated and
                unverifiable answers hide; they stay in raw data only. Blind
                stratum != trivial (robust/leaky-* are stratification metadata,
                not deletions). Near-duplicates of an earlier kept question
                (same normalized answer + >=0.45 question-token overlap, or
                >=0.7 question overlap) are dropped.
  exploration : row["keep"] as computed by generate_qa_exploration.py
                (bounds + set round-trip / hole sampling + identity gates).

Usage:
  python qa-data/convert_v6_to_benchmark.py [--out qa-data/benchmark/v6/benchmark_combined.jsonl]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

QA_DIR = Path("data/qa_needdown")
EXP_DIR = Path("data/qa_exploration")


import re


def _toks(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return set(w for w in s.split() if len(w) > 2)


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def is_near_dup(q, a, kept_by_video):
    """Same salient fact re-asked across cells (Augusta golf x3 on the pilot)."""
    qt = _toks(q)
    for pq, pa in kept_by_video:
        ov = len(qt & _toks(pq)) / max(1, len(qt))
        if (_norm(a) == _norm(pa) and ov >= 0.45) or ov >= 0.7:
            return True
    return False


def stratum(bs):
    if bs is None:
        return "unscored"
    if bs == 0:
        return "robust"
    if bs <= 0.34:
        return "leaky-to-few"
    if bs <= 0.67:
        return "leaky-to-some"
    return "trivial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qa-data/benchmark/v6/benchmark_combined.jsonl")
    ap.add_argument("--pipeline-version", default="v6.0")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    rows_out = []

    # ---- need-down free text ----
    for f in sorted(QA_DIR.glob("*.json")):
        if f.name == "run_stats.json":
            continue
        d = json.load(open(f))
        vid = d.get("video_id", f.stem)
        kept_by_video = []
        for i, r in enumerate(d.get("rows", [])):
            qa = r.get("qa", {})
            q, a = qa.get("question"), qa.get("answer")
            if not q or not a or qa.get("error"):
                stats["needdown_malformed"] += 1
                continue
            st = stratum(qa.get("blind_score"))
            rtv = r.get("roundtrip") or {}
            if rtv.get("skipped"):
                stats["needdown_drop_no_span"] += 1
                continue
            if rtv.get("consistent") is not True:
                stats["needdown_drop_roundtrip"] += 1
                continue
            if st == "trivial":
                stats["needdown_drop_trivial"] += 1
                continue
            if is_near_dup(q, str(a), kept_by_video):
                stats["needdown_drop_near_dup"] += 1
                continue
            kept_by_video.append((q, str(a)))
            ev = r.get("evidence", {}) or {}
            spans = ([{"start_ms": ev["start_ms"], "end_ms": ev.get("end_ms")}]
                     if ev.get("start_ms") is not None else [])
            rows_out.append({
                "id": f"v6-{vid.split('-')[-1][:12]}-{i:03d}",
                "video_id": vid,
                "question": q,
                "answer": a,
                "format": "freetext",
                "source_segment_times": spans,
                "reasoning_type": r.get("w_role"),
                "pipeline_version": args.pipeline_version,
                "v6": {
                    "cell": r.get("cell"),
                    "w_role": r.get("w_role"),
                    "element": r.get("element"),
                    "blind_score": qa.get("blind_score"),
                    "blind_stratum": st,
                    "blind_panel": qa.get("blind_panel"),
                    "densified": bool(qa.get("densified")),
                    "question_raw": qa.get("question_raw"),
                    "roundtrip": {k: rtv.get(k) for k in ("model", "consistent", "skipped")},
                },
            })
            stats[f"needdown_keep_{st}"] += 1

    # ---- exploration retrieval sets ----
    for f in sorted(EXP_DIR.glob("*.json")) if EXP_DIR.exists() else []:
        d = json.load(open(f))
        vid = d.get("video_id", f.stem)
        for i, r in enumerate(d.get("rows", [])):
            if not r.get("keep"):
                stats["exploration_drop_gates"] += 1
                continue
            qa = r.get("qa", {})
            gold = qa.get("answer_set") or []
            rows_out.append({
                "id": f"v6x-{(vid or 'corpus').split('-')[-1][:12]}-{i:03d}",
                "video_id": r.get("video_id"),   # None for corpus scope
                "question": qa.get("question"),
                "answer": [{k: g.get(k) for k in
                            ("video_id", "seg_id", "title", "start_ms", "end_ms", "person")}
                           for g in gold],
                "format": "retrieval_set",
                "scope": (r.get("evidence") or {}).get("scope"),
                "source_segment_times": [{"start_ms": g["start_ms"], "end_ms": g.get("end_ms")}
                                         for g in gold],
                "reasoning_type": "retrieval",
                "pipeline_version": args.pipeline_version,
                "v6": {
                    "cell": r.get("cell"),
                    "element": r.get("element"),
                    "necessity": r.get("necessity"),
                    "set_roundtrip": r.get("set_roundtrip"),
                    "completeness": r.get("completeness"),
                },
            })
            stats["exploration_keep"] += 1

    with open(out_path, "w") as out:
        for row in rows_out:
            out.write(json.dumps(row) + "\n")

    print(f"wrote {len(rows_out)} rows -> {out_path}")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
