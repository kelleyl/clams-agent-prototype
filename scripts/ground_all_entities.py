#!/usr/bin/env python3
"""Run entity grounding (Wikidata + Wikipedia) on all video indexes.

Skips entities already grounded. Uses 2s delay between API calls
to avoid rate limiting.
"""
import json
import sys
import os
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from utils.video_index import wikidata_lookup, wikipedia_lookup, _is_groundable_entity
from pathlib import Path

DELAY = 2.0  # seconds between API calls

def ground_video(index_path):
    doc = json.load(open(index_path))
    segments = doc.get("segments", [])

    # Collect unique entities not yet grounded
    already_grounded = set()
    for seg in segments:
        already_grounded.update(seg.get("grounded_entities", {}).keys())

    unique = {}
    for seg in segments:
        for ent in seg.get("named_entities", []):
            text = ent.get("text", "").strip()
            etype = ent.get("type", "")
            if (text and text not in unique and text not in already_grounded
                    and _is_groundable_entity(text, etype)):
                unique[text] = etype

    if not unique:
        return 0, len(already_grounded)

    # Ground each entity
    new_grounded = {}
    for entity_text, entity_type in unique.items():
        parts = []
        try:
            wd = wikidata_lookup(entity_text)
            if wd:
                parts.append(wd)
            wiki = wikipedia_lookup(entity_text)
            if wiki and (not wd or len(wiki) > len(wd)):
                parts.append(wiki)
        except Exception as e:
            pass  # Skip on error, don't crash
        new_grounded[entity_text] = " ".join(parts).strip()
        time.sleep(DELAY)

    # Write back
    n_new = sum(1 for v in new_grounded.values() if v)
    for seg in segments:
        for ent in seg.get("named_entities", []):
            text = ent.get("text", "").strip()
            if text in new_grounded and new_grounded[text]:
                seg.setdefault("grounded_entities", {})[text] = new_grounded[text]

    doc["segments"] = segments
    with open(index_path, "w") as f:
        json.dump(doc, f, indent=2)

    return n_new, len(already_grounded) + n_new


def main():
    idx_dir = Path("data/video_indexes")
    videos = sorted(idx_dir.glob("*.json"))

    print(f"Grounding entities for {len(videos)} videos (delay={DELAY}s)...", flush=True)
    total_new = 0
    total_existing = 0

    for i, f in enumerate(videos):
        vid = f.stem
        doc = json.load(open(f))
        n_segs = len(doc.get("segments", []))
        n_ent = sum(len(s.get("named_entities", [])) for s in doc.get("segments", []))
        n_existing = sum(len(s.get("grounded_entities", {})) for s in doc.get("segments", []))

        if n_ent == 0:
            continue

        n_new, n_total = ground_video(str(f))
        total_new += n_new
        total_existing += n_existing
        print(f"  [{i+1}/{len(videos)}] {vid[-30:]:30s} new:{n_new:3d} total:{n_total:4d}/{n_ent}", flush=True)

    print(f"\nDone. New: {total_new}, Previously grounded: {total_existing}")


if __name__ == "__main__":
    main()
