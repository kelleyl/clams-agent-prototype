#!/usr/bin/env python3
"""Aggregate shard verdicts from the full-coverage machine review, and emit
verification shards for SUSPECT rows with FULLER evidence (reviewers saw
700-char ranked excerpts; verification re-checks against the whole window so
excerpt truncation cannot kill a good row).

Usage:
  python scripts/aggregate_machine_review.py --scratch <dir>            # aggregate
  python scripts/aggregate_machine_review.py --scratch <dir> --apply \
      --verdicts <dir>/verify_verdicts_*.json                            # later
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")


def full_asr(doc, start_ms, end_ms, max_chars=4000):
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
                if t and t[:80].lower() not in seen:
                    seen.add(t[:80].lower())
                    parts.append((s, t))
    parts.sort()
    return " ".join(t for _, t in parts)[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--n-verify-shards", type=int, default=4)
    args = ap.parse_args()
    scratch = Path(args.scratch)

    corpus = {json.loads(l)["id"]: json.loads(l)
              for l in open(scratch / "v6_review_corpus.jsonl")}
    verdicts = {}
    for f in sorted(glob.glob(str(scratch / "verdicts_*.json"))):
        try:
            for v in json.load(open(f)):
                if isinstance(v, dict) and v.get("id"):
                    verdicts[v["id"]] = v
        except Exception as ex:
            print(f"WARN: {f}: {ex}")

    missing = [i for i in corpus if i not in verdicts]
    tags = Counter()
    suspects = []
    for rid, v in verdicts.items():
        if v.get("verdict") == "SUSPECT":
            for t in v.get("tags", []):
                tags[t] += 1
            suspects.append(rid)

    print(f"corpus rows: {len(corpus)} | verdicts: {len(verdicts)} | missing: {len(missing)}")
    print(f"SUSPECT: {len(suspects)} ({100 * len(suspects) / max(len(verdicts), 1):.1f}%)")
    print("tags:", dict(tags.most_common()))
    json.dump(verdicts, open(scratch / "all_verdicts.json", "w"), indent=0)
    if missing:
        print("missing ids (first 10):", missing[:10])

    # verification shards with FULL evidence
    idx_cache = {}

    def doc_of(vid):
        if vid not in idx_cache:
            p = Path(f"data/video_indexes/{vid}.json")
            idx_cache[vid] = json.load(open(p)) if p.exists() else {}
        return idx_cache[vid]

    vrows = []
    sidecars = {"speech": Path("data/qa_needdown"), "visual": Path("data/qa_visual")}
    for rid in suspects:
        row = dict(corpus[rid])
        row["complaint"] = verdicts[rid]
        if row["family"] == "speech":
            # locate the sidecar row for the full window
            m = None
            for f in sidecars["speech"].glob("*.json"):
                if f.name == "run_stats.json":
                    continue
                d = json.load(open(f))
                if d.get("video_id") != row["video_id"]:
                    continue
                i = int(rid.rsplit("-", 1)[1])
                if i < len(d["rows"]):
                    m = d["rows"][i]
                break
            if m:
                ev = m.get("evidence", {}) or {}
                if ev.get("start_ms") is not None:
                    row["full_evidence"] = full_asr(doc_of(row["video_id"]),
                                                    ev["start_ms"],
                                                    ev.get("end_ms") or ev["start_ms"])
        vrows.append(row)

    n = max(1, args.n_verify_shards)
    for i in range(n):
        json.dump(vrows[i::n], open(scratch / f"verify_shard_{i}.json", "w"), indent=0)
    print(f"verification shards: {n} x ~{len(vrows) // max(n,1)} rows -> verify_shard_*.json")


if __name__ == "__main__":
    main()
