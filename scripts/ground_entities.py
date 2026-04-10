#!/usr/bin/env python3
"""Run entity grounding (Wikidata + Wikipedia, no web search) on video indexes."""
import json
import sys
import os
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from utils.video_index import wikidata_lookup, wikipedia_lookup, _is_groundable_entity
from pathlib import Path

# Process ALL videos
import glob
guids = [Path(f).stem for f in sorted(glob.glob(str(Path(path).parent / "*.json")))]
# Actually just get all from index dir
guids = [f.stem for f in sorted(Path("data/video_indexes").glob("*.json"))]

for vid in guids:
    path = Path(f"data/video_indexes/{vid}.json")
    doc = json.load(open(path))

    unique = {}
    for seg in doc["segments"]:
        for ent in seg.get("named_entities", []):
            text = ent.get("text", "").strip()
            etype = ent.get("type", "")
            if text and text not in unique and _is_groundable_entity(text, etype):
                unique[text] = etype

    grounded = {}
    for i, (entity_text, entity_type) in enumerate(unique.items()):
        parts = []
        try:
            wd = wikidata_lookup(entity_text)
            if wd:
                parts.append(wd)
            wiki = wikipedia_lookup(entity_text)
            if wiki and (not wd or len(wiki) > len(wd)):
                parts.append(wiki)
        except Exception as e:
            print(f"  ERROR on {entity_text}: {e}", file=sys.stderr)
        grounded[entity_text] = " ".join(parts).strip()
        time.sleep(2)  # Rate limit: Wikidata enforces ~30 req/min

    n_grounded = sum(1 for v in grounded.values() if v)
    for seg in doc["segments"]:
        for ent in seg.get("named_entities", []):
            text = ent.get("text", "").strip()
            if text in grounded and grounded[text]:
                seg.setdefault("grounded_entities", {})[text] = grounded[text]

    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"{vid[-25:]:25s} grounded: {n_grounded}/{len(unique)}", flush=True)
