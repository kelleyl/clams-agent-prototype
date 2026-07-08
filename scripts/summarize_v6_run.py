#!/usr/bin/env python3
"""Aggregate stats for a v6 run: cells, 5W roles, gate rates, blind strata,
per-video counts. Writes data/qa_needdown/run_stats.json and prints a summary.

Usage: python scripts/summarize_v6_run.py
"""
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

QA_DIR = Path("data/qa_needdown")
EXP_DIR = Path("data/qa_exploration")
EXCL = Path("data/v6_excluded_videos.json")


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
    cells = Counter()
    roles = Counter()
    strata = Counter()
    rt = Counter()
    per_video = {}
    survivors = 0
    n_q = 0

    for f in sorted(QA_DIR.glob("*.json")):
        if f.name == "run_stats.json":
            continue
        d = json.load(open(f))
        vid_n = vid_keep = 0
        for r in d.get("rows", []):
            qa = r.get("qa", {})
            if not qa.get("question") or qa.get("error"):
                continue
            n_q += 1
            vid_n += 1
            cells[r.get("cell")] += 1
            roles[r.get("w_role")] += 1
            strata[stratum(qa.get("blind_score"))] += 1
            rtv = (r.get("roundtrip") or {})
            key = ("pass" if rtv.get("consistent") else
                   "fail" if rtv.get("consistent") is False else
                   "skipped" if rtv.get("skipped") else "missing")
            rt[key] += 1
            keep = (key in ("pass", "skipped")
                    and stratum(qa.get("blind_score")) != "trivial")
            survivors += 1 if keep else 0
            vid_keep += 1 if keep else 0
        per_video[d.get("video_id", f.stem)] = {"n": vid_n, "keep": vid_keep}

    exp_rows = exp_keep = 0
    for f in sorted(EXP_DIR.glob("*.json")) if EXP_DIR.exists() else []:
        d = json.load(open(f))
        for r in d.get("rows", []):
            exp_rows += 1
            exp_keep += 1 if r.get("keep") else 0

    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True).strip()
    except Exception:
        sha = None

    why_how = roles.get("why", 0) + roles.get("how", 0)
    stats = {
        "git_sha": sha,
        "n_videos": len(per_video),
        "n_questions": n_q,
        "survivors_est": survivors,
        "why_how_share": round(why_how / max(n_q, 1), 3),
        "cells": dict(cells),
        "w_roles": dict(roles),
        "blind_strata": dict(strata),
        "roundtrip": dict(rt),
        "exploration": {"rows": exp_rows, "keep": exp_keep},
        "videos_with_zero_keep": [v for v, c in per_video.items() if c["keep"] == 0],
        "excluded": (json.load(open(EXCL)).get("excluded", []) if EXCL.exists() else []),
        "per_video": per_video,
    }
    json.dump(stats, open(QA_DIR / "run_stats.json", "w"), indent=1)

    print(f"videos: {stats['n_videos']} | questions: {n_q} | est. survivors "
          f"(rt-pass + non-trivial): {survivors}")
    print(f"why/how share: {100 * stats['why_how_share']:.1f}%  (v5.2 baseline 6.5%)")
    print("blind strata:", dict(strata))
    print("round-trip:", dict(rt))
    print(f"exploration: {exp_keep}/{exp_rows} kept")
    zk = stats["videos_with_zero_keep"]
    print(f"videos with zero surviving questions: {len(zk)}"
          + (f" -> {[v[:40] for v in zk[:8]]}" if zk else ""))
    print(f"stats -> {QA_DIR / 'run_stats.json'}")


if __name__ == "__main__":
    main()
