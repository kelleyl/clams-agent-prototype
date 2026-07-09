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


_STOP = set("the a an of to in on for and or but is are was were be been it this that with as at "
             "by from who what why how which when where".split())


def _toks(s):
    import re
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return set(w for w in s.split() if len(w) > 2 and w not in _STOP)


def asr_excerpt(doc, start_ms, end_ms, query="", answer="", max_chars=900):
    """The most query-relevant ASR turns of the window, assembled in RANK order
    under a character budget (each turn capped, so a long top turn cannot push
    the answer-bearing turn past the cut), then displayed in time order with
    ellipses between non-adjacent turns. Returns (text, anchor_ms) where
    anchor_ms is the start of the best answer-bearing turn (for frame anchoring).
    """
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
    # the index carries multiple ASR layers (asr, asr_vibevoice): drop repeats
    seen_txt, deduped = set(), []
    for s, t in parts:
        key = t[:80].lower()
        if key not in seen_txt:
            seen_txt.add(key)
            deduped.append((s, t))
    parts = deduped
    if not parts:
        return "", None
    q = _toks(query)
    if not q:
        return " ".join(t for _, t in parts)[:max_chars], None

    per_turn = 300
    scores = [len(q & _toks(t)) for _, t in parts]
    ranked = sorted(range(len(parts)), key=lambda i: -scores[i])

    # frame anchor: highest-ranked turn that carries an answer token, else rank 0
    a_toks = _toks(answer)
    anchor_ms = None
    for i in ranked[:6]:
        if scores[i] == 0:
            break
        if a_toks and (a_toks & _toks(parts[i][1])):
            anchor_ms = parts[i][0]
            break
    if anchor_ms is None and scores[ranked[0]] > 0:
        anchor_ms = parts[ranked[0]][0]

    def clip_turn(text, cap):
        """Truncate a long turn AROUND its most query-relevant sentence, not its
        head - the relevant clause is often deep inside a long news-summary turn."""
        if len(text) <= cap:
            return text
        import re as _re
        sents = _re.split(r"(?<=[.!?])\s+", text)
        best = max(range(len(sents)), key=lambda i: len(q & _toks(sents[i])))
        out, lo, hi = sents[best], best - 1, best + 1
        while len(out) < cap and (lo >= 0 or hi < len(sents)):
            if hi < len(sents) and len(out) + len(sents[hi]) < cap:
                out = out + " " + sents[hi]
                hi += 1
            elif lo >= 0 and len(out) + len(sents[lo]) < cap:
                out = sents[lo] + " " + out
                lo -= 1
            else:
                break
        pre = "[...] " if lo >= 0 else ""
        post = " [...]" if hi < len(sents) else ""
        return pre + out + post

    selected, budget = set(), max_chars
    for i in ranked[:4]:
        if scores[i] == 0 or budget <= 0:
            break
        for j in (i, i - 1, i + 1):        # core first, then neighbors
            if 0 <= j < len(parts) and j not in selected:
                t = clip_turn(parts[j][1], per_turn)
                if len(t) <= budget or (j == i and not selected):
                    selected.add(j)
                    budget -= len(t)
    pieces, prev = [], None
    for j in sorted(selected):
        if prev is not None and j > prev + 1:
            pieces.append("[...]")
        pieces.append(clip_turn(parts[j][1], per_turn))
        prev = j
    return " ".join(pieces), anchor_ms


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
                # chapter titles are unreliable on ~30/108 videos; show content
                snip = (g.get("excerpt") or g.get("snippet") or "")[:90]
                lines.append(f"[{g['video_id'][:36]} {t}]{who} {g.get('title') or ''}"
                             + (f" << {snip}" if snip else ""))
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

    n_rows = n_skipped = n_policy_skipped = 0
    with open(out_path, "w") as out:
        for sp in sorted(in_dir.glob("*.json")):
            vid = sp.stem
            if vid in EXCLUDE_VIDEOS:
                n_skipped += 1
                continue
            data = json.load(open(sp))
            idxp = idx_dir / f"{vid}.json"
            doc = json.load(open(idxp)) if idxp.exists() else {}
            kept_qa = []
            for i, r in enumerate(data.get("rows", [])):
                qa = r.get("qa", {})
                q, a = qa.get("question"), qa.get("answer")
                if not q or not a or qa.get("error"):
                    continue
                # excluded from the benchmark by policy -> not worth rating time
                if (r.get("element") or "").startswith("two_hop:"):
                    n_policy_skipped += 1
                    continue
                qt = _toks(q)
                dup = any((len(qt & _toks(pq)) / max(1, len(qt)) >= 0.45
                           and str(a).lower().strip() == pa)
                          or len(qt & _toks(pq)) / max(1, len(qt)) >= 0.7
                          for pq, pa in kept_qa)
                if dup:
                    n_policy_skipped += 1
                    continue
                kept_qa.append((q, str(a).lower().strip()))
                ev = r.get("evidence", {}) or {}
                start, end = ev.get("start_ms"), ev.get("end_ms")
                spans = []
                anchor_ms = None
                if start is not None and doc:
                    txt, anchor_ms = asr_excerpt(doc, start, end or start,
                                                 query=f"{q} {a}", answer=a)
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
                    "support_anchor_ms": anchor_ms,
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
          f"(excluded {n_skipped} stale video(s), {n_policy_skipped} policy-dropped rows: "
          f"two-hop + near-dups)")


if __name__ == "__main__":
    main()
