#!/usr/bin/env python3
"""Summarize the v6 pilot human rating against the automated gates.

Joins fast_review votes with the review jsonl's carried gate metadata and
prints pass rates overall and by cell / 5W role / blind stratum / round-trip
verdict, plus the go/no-go acceptance view from V6_DATASET_PLAN.md section D.

Usage:
  python scripts/summarize_v6_pilot_review.py \
      --review qa-data/raw/v6/pilot_review.jsonl \
      --votes annotation/fast_review_v6_pilot.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def stratum(blind_score):
    if blind_score is None:
        return "unscored"
    if blind_score == 0:
        return "robust"
    if blind_score <= 0.34:
        return "leaky-to-few"
    if blind_score <= 0.67:
        return "leaky-to-some"
    return "trivial"


def rate(pairs):
    """pairs: list of (q_vote, a_vote). Returns display string."""
    n = len(pairs)
    if not n:
        return "  (none)"
    q_up = sum(1 for q, _ in pairs if q == "up")
    a_up = sum(1 for q, a in pairs if q == "up" and a == "up")
    return (f"n={n:3d}  Q-up {q_up:3d} ({100*q_up/n:3.0f}%)  "
            f"both-up {a_up:3d} ({100*a_up/n:3.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="qa-data/raw/v6/pilot_review.jsonl")
    ap.add_argument("--votes", default="annotation/fast_review_v6_pilot.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.review) if l.strip()]
    votes = json.loads(Path(args.votes).read_text()) if Path(args.votes).exists() else {}

    joined = []
    for r in rows:
        v = votes.get(r["id"]) or {}
        joined.append((r, v.get("q_vote"), v.get("a_vote")))

    rated = [(r, q, a) for r, q, a in joined if q]
    print(f"questions: {len(rows)} | rated: {len(rated)} | unrated: {len(rows) - len(rated)}\n")
    if not rated:
        return

    pairs_all = [(q, a) for _, q, a in rated]
    print("OVERALL           ", rate(pairs_all))

    for key, label in [(lambda r: (r['v6'].get('cell') or '?'), "by cell"),
                       (lambda r: (r['v6'].get('w_role') or '?'), "by 5W role"),
                       (lambda r: stratum(r['v6'].get('blind_score')), "by blind stratum"),
                       (lambda r: {True: "rt-pass", False: "rt-fail", None: "rt-skipped"}
                        [r['v6'].get('roundtrip_consistent')], "by round-trip verdict"),
                       (lambda r: r['video_id'], "by video")]:
        print(f"\n=== {label} ===")
        groups = defaultdict(list)
        for r, q, a in rated:
            groups[key(r)].append((q, a))
        for g in sorted(groups):
            print(f"  {g[:44]:46s}", rate(groups[g]))

    # Acceptance view (plan section D): human interest above threshold on the
    # material the benchmark would actually keep (rt-pass, non-trivial).
    keep = [(q, a) for r, q, a in rated
            if r['v6'].get('roundtrip_consistent')
            and stratum(r['v6'].get('blind_score')) != "trivial"]
    print("\n=== ACCEPTANCE (rt-pass + non-trivial stratum: what v6 would keep) ===")
    print("  kept slice        ", rate(keep))
    print("\nGo/no-go guidance: Q-up on the kept slice is the interest metric;")
    print("both-up is the end-to-end quality metric. The plan's recurring-failure")
    print("bar: do not scale if a systemic failure mode explains the down-votes.")


if __name__ == "__main__":
    main()
