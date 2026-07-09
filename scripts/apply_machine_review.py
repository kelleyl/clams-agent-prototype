#!/usr/bin/env python3
"""Apply the full-coverage machine review to the preview benchmark and write
the reviewed release candidate + review report.

Disposition per row:
  OK first pass                  -> machine_review: "pass"
  SUSPECT, complaint REFUTED     -> machine_review: "pass_after_verification"
  SUSPECT, complaint CONFIRMED   -> EXCLUDED (listed in the report)
  no verdict (reviewer miss)     -> machine_review: "unreviewed" (kept, flagged)

Usage:
  python scripts/apply_machine_review.py --scratch <dir> \
      --preview qa-data/benchmark/v6_preview/benchmark_combined.jsonl \
      --out-dir qa-data/benchmark/v6
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--preview", default="qa-data/benchmark/v6_preview/benchmark_combined.jsonl")
    ap.add_argument("--out-dir", default="qa-data/benchmark/v6")
    args = ap.parse_args()
    scratch = Path(args.scratch)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts = json.load(open(scratch / "all_verdicts.json"))
    verify = {}
    for f in sorted(glob.glob(str(scratch / "verify_verdicts_*.json"))):
        for v in json.load(open(f)):
            if isinstance(v, dict) and v.get("id"):
                verify[v["id"]] = v

    rows = [json.loads(l) for l in open(args.preview)]
    kept, excluded = [], []
    disp = Counter()
    for r in rows:
        rid = r["id"]
        v = verdicts.get(rid)
        if v is None:
            r["machine_review"] = {"status": "unreviewed"}
            disp["unreviewed"] += 1
            kept.append(r)
            continue
        if v.get("verdict") != "SUSPECT":
            r["machine_review"] = {"status": "pass"}
            disp["pass"] += 1
            kept.append(r)
            continue
        vv = verify.get(rid)
        if vv and vv.get("verdict") == "REFUTED":
            r["machine_review"] = {"status": "pass_after_verification",
                                   "complaint": v.get("tags"),
                                   "refutation": vv.get("reason")}
            disp["pass_after_verification"] += 1
            kept.append(r)
        else:
            disp["excluded"] += 1
            excluded.append({"id": rid, "video_id": r.get("video_id"),
                             "question": r.get("question"),
                             "answer": r.get("answer") if isinstance(r.get("answer"), str) else None,
                             "tags": v.get("tags"), "reason": v.get("reason"),
                             "confirmation": (vv or {}).get("reason")})

    with open(out_dir / "benchmark_combined.jsonl", "w") as out:
        for r in kept:
            out.write(json.dumps(r) + "\n")
    json.dump(excluded, open(out_dir / "review_excluded.json", "w"), indent=1)

    fam = Counter()
    strata = Counter()
    for r in kept:
        fam["visual" if r["id"].startswith("v6v") else
            "exploration" if r["id"].startswith("v6x") else "speech"] += 1
        strata[(r.get("v6") or {}).get("blind_stratum") or "n/a"] += 1

    report = out_dir / "REVIEW_REPORT.md"
    with open(report, "w") as f:
        f.write("# v6.0 release candidate - machine review report\n\n")
        f.write("Full-coverage two-stage review: 13 independent reviewer agents over all\n")
        f.write("preview rows (evidence-grounded rubric), then adversarial verification of\n")
        f.write("every complaint against the full evidence window. Human sample review\n")
        f.write("(fast_review, port 8782) ratifies before release.\n\n")
        f.write(f"- kept: {len(kept)} | excluded: {len(excluded)}\n")
        f.write(f"- disposition: {dict(disp)}\n")
        f.write(f"- families: {dict(fam)}\n")
        f.write(f"- blind strata (kept): {dict(strata)}\n\n")
        f.write("## Excluded rows\n\n")
        for e in excluded:
            f.write(f"- `{e['id']}` [{','.join(e.get('tags') or [])}] {e.get('reason')}\n")
    print(f"kept {len(kept)} | excluded {len(excluded)} -> {out_dir}/")
    print("disposition:", dict(disp))
    print("families:", dict(fam))


if __name__ == "__main__":
    main()
