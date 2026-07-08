#!/usr/bin/env python3
"""Compare generator models from data/gen_compare/<slug>/ archives."""
import json
import glob
import os
import statistics
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = "data/gen_compare"


def load(slug):
    rows = []
    for f in glob.glob(f"{BASE}/{slug}/*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for r in d.get("rows", []):
            if r["qa"].get("question") and not r["qa"].get("error"):
                rows.append(r)
    return rows


def main():
    slugs = sorted(d for d in os.listdir(BASE) if os.path.isdir(f"{BASE}/{d}")) if os.path.isdir(BASE) else []
    print(f"{'generator':14}{'n':>4}{'robust%':>9}{'rt_pass%':>9}{'ans_words':>10}{'why/how%':>9}{'vague%':>8}")
    for slug in slugs:
        rows = load(slug)
        n = len(rows) or 1
        robust = sum(1 for r in rows if r["qa"].get("blind_score") == 0)
        rt = [r for r in rows if isinstance(r.get("roundtrip"), dict) and "consistent" in r["roundtrip"]]
        rt_pass = sum(1 for r in rt if r["roundtrip"]["consistent"])
        alen = [len(str(r["qa"].get("answer", "")).split()) for r in rows]
        wh = sum(1 for r in rows if r.get("w_role") in ("why", "how"))
        import re
        vague = sum(1 for r in rows if re.search(r"\b(this segment|the report|the speaker|referenced)\b",
                                                 r["qa"].get("question", ""), re.I))
        print(f"{slug:14}{len(rows):>4}{100*robust/n:>8.0f}%"
              f"{(100*rt_pass/len(rt) if rt else 0):>8.0f}%"
              f"{statistics.median(alen) if alen else 0:>10.0f}"
              f"{100*wh/n:>8.0f}%{100*vague/n:>7.0f}%")


if __name__ == "__main__":
    main()
