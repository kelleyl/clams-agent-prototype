#!/usr/bin/env python3
"""Flatten v6 need-down pilot sidecars into a fast_review jsonl.

Reads the A-slice arm of the pilot (the arm the full run will use:
raw-ASR-slice evidence, gemma3:27b generator, full gate state) and emits one
row per question in the shape annotation/fast_review.py renders. Gate verdicts
(blind_score, blind_panel, roundtrip) are carried on the row for the summary
script but NOT surfaced in the review UI, so the human rating stays
independent of the automated gates.

Usage:
  python qa-data/convert_v6_pilot_to_review.py \
      --input-dir data/gen_compare/A-slice \
      --output qa-data/raw/v6/pilot_review.jsonl
  python annotation/fast_review.py --input qa-data/raw/v6/pilot_review.jsonl \
      --output annotation/fast_review_v6_pilot.json --port 8782
"""
import argparse
import json
from pathlib import Path

# Stale pre-decision run (MC-era, never gated); regenerated in the full run.
EXCLUDE_VIDEOS = {"cpb-aacip-507-154dn40c26"}


def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def asr_excerpt(doc, start_ms, end_ms, max_chars=600):
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


def exploration_rows(exp_dir, out):
    """Append kept exploration questions; the answer set renders as one
    string per line for the fast_review free-text stage 2."""
    n = 0
    for sp in sorted(Path(exp_dir).glob("*.json")):
        data = json.load(open(sp))
        vid = data.get("video_id") or sp.stem
        for i, r in enumerate(data.get("rows", [])):
            if not r.get("keep"):
                continue
            qa = r.get("qa", {})
            gold = qa.get("answer_set") or []
            lines = []
            for g in gold:
                t = f"{g['start_ms'] // 60000}m-{(g.get('end_ms') or 0) // 60000}m"
                who = f" ({g['person']})" if g.get("person") else ""
                lines.append(f"[{g['video_id'][:36]} {t}]{who} {g.get('title') or ''}")
            row = {
                "id": f"v6x-{(vid or 'corpus').split('-')[-1][:12]}-{i:03d}",
                "video_id": r.get("video_id") or "(corpus-level)",
                "question": qa.get("question"),
                "answer": " ;; ".join(lines),
                "format": "freetext",
                "task_family": f"{r.get('cell')} / retrieval",
                "verification": {"rationale": qa.get("rationale", ""), "support_spans": []},
                "v6": {"cell": r.get("cell"), "w_role": r.get("w_role"),
                       "element": r.get("element"), "blind_score": None,
                       "blind_panel": None, "densified": False,
                       "roundtrip_consistent": (r.get("set_roundtrip") or {}).get("pass"),
                       "exploration": True},
            }
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/gen_compare/A-slice")
    ap.add_argument("--exploration-dir", default="",
                    help="also append kept rows from this qa_exploration dir")
    ap.add_argument("--index-dir", default="data/video_indexes")
    ap.add_argument("--output", default="qa-data/raw/v6/pilot_review.jsonl")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    idx_dir = Path(args.index_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_skipped = 0
    with open(out_path, "w") as out:
        for sp in sorted(in_dir.glob("*.json")):
            vid = sp.stem
            if vid in EXCLUDE_VIDEOS:
                n_skipped += 1
                continue
            data = json.load(open(sp))
            idxp = idx_dir / f"{vid}.json"
            doc = json.load(open(idxp)) if idxp.exists() else {}
            for i, r in enumerate(data.get("rows", [])):
                qa = r.get("qa", {})
                q, a = qa.get("question"), qa.get("answer")
                if not q or not a or qa.get("error"):
                    continue
                ev = r.get("evidence", {}) or {}
                start, end = ev.get("start_ms"), ev.get("end_ms")
                spans = []
                if start is not None and doc:
                    txt = asr_excerpt(doc, start, end or start)
                    if txt:
                        spans.append({"modality": "asr", "text": txt})
                rt = r.get("roundtrip") or {}
                row = {
                    "id": f"v6-{vid.split('-')[-1]}-{i:03d}",
                    "video_id": vid,
                    "question": q,
                    "question_original": qa.get("question_raw"),
                    "answer": a,
                    "format": "freetext",
                    "task_family": f"{r.get('cell', '')} / {r.get('w_role', '')}",
                    "window_start_ms": start,
                    "window_end_ms": end,
                    "verification": {
                        "rationale": qa.get("rationale", ""),
                        "support_spans": spans,
                    },
                    # gate metadata for scripts/summarize_v6_pilot_review.py
                    # (not rendered by the review UI)
                    "v6": {
                        "cell": r.get("cell"),
                        "w_role": r.get("w_role"),
                        "element": r.get("element"),
                        "densified": bool(qa.get("densified")),
                        "self_contained_flag": qa.get("self_contained_flag"),
                        "blind_score": qa.get("blind_score"),
                        "blind_panel": qa.get("blind_panel"),
                        "roundtrip_consistent": rt.get("consistent"),
                        "roundtrip_fresh_answer": rt.get("fresh_answer"),
                    },
                }
                out.write(json.dumps(row) + "\n")
                n_rows += 1

    n_exp = 0
    if args.exploration_dir:
        with open(out_path, "a") as out:
            n_exp = exploration_rows(args.exploration_dir, out)

    print(f"wrote {n_rows} need-down + {n_exp} exploration rows -> {out_path} "
          f"(excluded {n_skipped} stale video(s))")


if __name__ == "__main__":
    main()
