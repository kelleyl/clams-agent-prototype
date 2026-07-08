#!/usr/bin/env python3
"""Build a deterministic salience map per video from existing index layers.

The salience map answers "what matters in this video" so that question
generation can work need-down (ask about salient elements) rather than
evidence-up (extract whatever is in a window). NO LLM, NO network.

Signals used (all already in data/video_indexes/*.json):
  - chapters            -> salient segments (titled time spans)
  - speakers            -> main participants by speaking time
  - entities (asr/ocr)  -> recurring named PERSON/ORG
  - scenes              -> production structure (slate/chyron/credits labels)

Output: data/salience_maps/<video_id>.json + a printed summary with --verbose.
"""
import argparse
import difflib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
try:
    from utils.video_index import _is_groundable_entity
except Exception:
    def _is_groundable_entity(t, typ):
        return len(t) > 2 and not t.islower()

IDX_DIR = Path("data/video_indexes")
OUT_DIR = Path("data/salience_maps")


def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _norm_speaker(n):
    # drop diarization segment suffixes like "_seg1", "_seg3"
    return re.sub(r"_seg\d+$", "", n).strip()


def _merge_speaker_variants(spk, thresh=0.88):
    """Merge near-duplicate speaker names (e.g. 'Dr. Krim' / 'Dr. Krimm')."""
    merged = {}
    for nm in sorted(spk, key=lambda n: -spk[n]["speaking_ms"]):
        hit = None
        for c in merged:
            if difflib.SequenceMatcher(None, nm.lower(), c.lower()).ratio() >= thresh:
                hit = c
                break
        if hit:
            for k in ("speaking_ms", "turns", "words"):
                merged[hit][k] += spk[nm][k]
        else:
            merged[nm] = dict(spk[nm])
    return merged


def build(doc):
    L = doc.get("layers", {})
    dur = doc.get("duration_ms")
    chapters = items(L.get("chapters"))
    speakers = items(L.get("speakers"))
    scenes = items(L.get("scenes"))
    ents = items(L.get("entities"))

    # --- participants: aggregate speaker turns by name ---
    spk = defaultdict(lambda: {"speaking_ms": 0, "turns": 0, "words": 0})
    for t in speakers:
        nm = _norm_speaker((t.get("speaker_name") or t.get("speaker_id") or "").strip())
        if not nm:
            continue
        s, e = t.get("start_ms", 0), t.get("end_ms", 0)
        spk[nm]["speaking_ms"] += max(0, (e or 0) - (s or 0))
        spk[nm]["turns"] += 1
        spk[nm]["words"] += len((t.get("text") or "").split())
    spk = _merge_speaker_variants(spk)

    # --- recurring named entities from asr/ocr (exclude caption noise) ---
    ent_count = defaultdict(int)
    ent_type = {}
    for e in ents:
        sl = (e.get("source_layer") or "").lower()
        if not (sl.startswith("asr") or "ocr" in sl):
            continue
        typ = e.get("type")
        if typ not in ("PERSON", "ORG"):
            continue
        txt = (e.get("text") or "").strip()
        if not txt or not _is_groundable_entity(txt, typ) or len(txt.split()) < 2:
            continue
        ent_count[txt] += 1
        ent_type[txt] = typ

    # resolve a (possibly partial) speaker name to the fullest matching named
    # entity, e.g. "Jim" -> "Jim Lehrer", "Haynes" -> "Haynes Johnson". Prefers
    # PERSON entities and the most-mentioned candidate.
    def resolve_name(name):
        nl = name.lower().strip()
        ntok = set(nl.split())
        best = None  # (priority, count, fullname)
        for et, c in ent_count.items():
            el = et.lower()
            etok = set(el.split())
            if el == nl:
                pri = 3.0
            elif ntok and ntok <= etok and len(etok) > len(ntok):
                pri = 2.0          # short form of a fuller entity (first name -> full)
            elif etok and etok <= ntok:
                pri = 1.0          # name already fuller than the entity
            else:
                continue
            if ent_type.get(et) == "PERSON":
                pri += 0.5
            if best is None or (pri, c) > (best[0], best[1]):
                best = (pri, c, et)
        return best[2] if best else None

    participants = []
    used_ents = set()
    for nm, st in sorted(spk.items(), key=lambda kv: -kv[1]["speaking_ms"]):
        full = resolve_name(nm)
        if full:
            used_ents.add(full)
        display = full if (full and len(full) >= len(nm)) else nm
        participants.append({"name": display, "kind": "both" if full else "speaker",
                             "speaking_ms": st["speaking_ms"], "turns": st["turns"],
                             "mention_count": ent_count.get(full, 0)})
    for et, c in sorted(ent_count.items(), key=lambda kv: -kv[1]):
        if et in used_ents:
            continue
        participants.append({"name": et, "kind": "entity", "type": ent_type[et],
                             "speaking_ms": 0, "turns": 0, "mention_count": c})

    # --- salient segments from chapters (with dominant speaker + snippet) ---
    segs = []
    for ch in chapters:
        s, e = ch.get("start_ms", 0), ch.get("end_ms", 0)
        # dominant speaker by overlap
        best, best_ov, snippet = None, 0, ""
        for t in speakers:
            ov = overlap(s, e, t.get("start_ms", 0), t.get("end_ms", 0))
            if ov > best_ov:
                best_ov = ov
                best = (t.get("speaker_name") or t.get("speaker_id") or "").strip()
                snippet = (t.get("text") or "")[:160]
        segs.append({"id": ch.get("id"), "title": ch.get("title"),
                     "start_ms": s, "end_ms": e, "dur_ms": max(0, (e or 0) - (s or 0)),
                     "dominant_speaker": best, "snippet": snippet})
    segs.sort(key=lambda x: -x["dur_ms"])

    scene_labels = defaultdict(int)
    for sc in scenes:
        lab = sc.get("label")
        if lab:
            scene_labels[lab] += 1

    return {
        "video_id": doc.get("video_id"),
        "duration_ms": dur,
        "broadcast_date": doc.get("broadcast_date"),
        "n_chapters": len(chapters),
        "n_named_speakers": sum(1 for p in participants if p["kind"] in ("speaker", "both")),
        "salient_segments": segs,
        "main_participants": participants[:15],
        "scene_labels": dict(sorted(scene_labels.items(), key=lambda kv: -kv[1])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-video", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(IDX_DIR.glob("*.json"))
    if args.only_video:
        paths = [p for p in paths if args.only_video in p.stem]
    if args.limit:
        paths = paths[:args.limit]

    n_ok = n_chap = n_part = 0
    for p in paths:
        doc = json.load(open(p))
        sm = build(doc)
        json.dump(sm, open(OUT_DIR / f"{p.stem}.json", "w"), indent=2)
        n_ok += 1
        n_chap += sm["n_chapters"]
        n_part += sm["n_named_speakers"]
        if args.verbose:
            print(f"\n=== {sm['video_id']}  (chapters={sm['n_chapters']}, "
                  f"named_speakers={sm['n_named_speakers']}) ===")
            print("  scene labels:", sm["scene_labels"])
            print("  top participants:")
            for pt in sm["main_participants"][:8]:
                print("    %-22s %-7s speak=%5ds turns=%-3d mentions=%d"
                      % (pt["name"][:22], pt["kind"], pt["speaking_ms"] // 1000,
                         pt["turns"], pt["mention_count"]))
            print("  salient segments (by duration):")
            for sg in sm["salient_segments"][:6]:
                print("    [%4ds] %-28s spk=%s"
                      % (sg["dur_ms"] // 1000, str(sg["title"])[:28], sg["dominant_speaker"]))

    print(f"\nbuilt {n_ok} salience maps | avg chapters={n_chap/max(n_ok,1):.1f} "
          f"avg named_speakers={n_part/max(n_ok,1):.1f} -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
