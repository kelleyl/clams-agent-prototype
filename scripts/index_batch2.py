#!/usr/bin/env python3
"""Build indexes + NER + relations + speaker assignment for batch 2 NewsHour videos."""
import json
import sys
import os
import spacy
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from utils.video_index import VideoIndex, _extract_broadcast_date, _extract_svo_triples, _is_groundable_entity

nlp = spacy.load("en_core_web_sm")
idx = VideoIndex(index_dir=Path("data/video_indexes"), chroma_dir=Path("data/chroma_db"))

BLOCKLIST = frozenset([
    "night","tonight","today","tomorrow","yesterday","morning","afternoon","evening",
    "dawn","dusk","midnight","noon","days","day","weeks","week","months","month",
    "years","year","hours","hour","minutes","minute","seconds","second",
    "tuesdays","mondays","wednesdays","thursdays","fridays","saturdays","sundays",
    "greenery","curls","mug","stolen","king","queen",
    "good morning","good evening","good night",
    "the early 20th century","the mid-20th century",
    "the early to mid-20th century","the late 20th century",
    "about 10 minutes","his day","all these years",
])

GUIDS = [
    "cpb-aacip-507-5q4rj49c6q","cpb-aacip-507-z892805z01","cpb-aacip-507-4b2x34n67h",
    "cpb-aacip-507-7659c6sk9k","cpb-aacip-507-ns0ks6jx8k","cpb-aacip-507-3b5w669q04",
    "cpb-aacip-507-h707w67w4s","cpb-aacip-507-dj58c9rs7q","cpb-aacip-507-fj29883b24",
    "cpb-aacip-507-f47gq6rq94","cpb-aacip-507-6h4cn6zj6q","cpb-aacip-507-ws8hd7pn8f",
    "cpb-aacip-507-4t6f18sz37","cpb-aacip-507-4t6f18t00t","cpb-aacip-507-3f4kk94w29",
    "cpb-aacip-507-kk94747k0n","cpb-aacip-507-4f1mg7gc44","cpb-aacip-507-5x2599zp0m",
    "cpb-aacip-507-251fj2b03t","cpb-aacip-507-jw86h4dh3c","cpb-aacip-507-4t6f18t15n",
    "cpb-aacip-507-bn9x05xz7r","cpb-aacip-507-jm23b5x19r","cpb-aacip-507-3x83j39n00",
    "cpb-aacip-507-dv1cj8881m",
]

for vid in GUIDS:
    mmif_path = f"data/combined_outputs/{vid}.mmif"
    if not os.path.exists(mmif_path):
        print(f"{vid}: MMIF not found, skipping")
        continue

    m = json.load(open(mmif_path))
    vp = ""
    for doc in m.get("documents", []):
        if "VideoDocument" in doc.get("@type", ""):
            vp = doc["properties"].get("location", "").replace("file://", "")

    segments = idx._extract_segments(m, vid)
    dur = max(s.end_ms for s in segments) if segments else 0
    d = {
        "video_id": vid, "video_path": vp,
        "broadcast_date": _extract_broadcast_date(vid),
        "duration_ms": dur, "indexed_at": "2026-03-19",
        "source_mmif": mmif_path,
        "segments": [s.to_dict() for s in segments],
        "persons": [], "speaker_names": {}, "speaker_details": {},
    }
    with open(f"data/video_indexes/{vid}.json", "w") as f:
        json.dump(d, f, indent=2)

    # NER
    d = json.load(open(f"data/video_indexes/{vid}.json"))
    te = 0
    for seg in d["segments"]:
        tp = [seg.get(k, "") for k in ("asr_transcript", "ocr_text", "visual_caption") if seg.get(k)]
        text = " ".join(tp)
        if len(text) < 10:
            continue
        sd = nlp(text)
        ex = {(e["text"], e["type"]) for e in seg.get("named_entities", [])}
        for ent in sd.ents:
            if ent.label_ in ("CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"):
                continue
            t = ent.text.strip()
            if len(t) <= 2 or t.lower() in BLOCKLIST:
                continue
            if t.replace(",", "").replace(".", "").replace("-", "").isdigit():
                continue
            if ent.label_ in ("DATE", "TIME") and t == t.lower() and not any(c.isdigit() for c in t):
                continue
            if (t, ent.label_) not in ex:
                seg.setdefault("named_entities", []).append({"text": t, "type": ent.label_})
                ex.add((t, ent.label_))
                te += 1

    # Relations
    tr = 0
    for i, seg in enumerate(d["segments"]):
        ts = [seg.get(k, "") for k in ("asr_transcript", "visual_caption") if seg.get(k)]
        if i + 1 < len(d["segments"]) and d["segments"][i + 1].get("asr_transcript"):
            ts.append(" ".join(d["segments"][i + 1]["asr_transcript"].split()[:50]))
        text = " ".join(ts)
        if len(text) < 15:
            continue
        ol = len(" ".join(ts[:-1])) if len(ts) > 1 else len(text)
        sd = nlp(text)
        ex = {tuple(r) for r in seg.get("relations", [])}
        for sent in sd.sents:
            if sent.start_char >= ol:
                break
            for s, p, o in _extract_svo_triples(sent):
                if (s, p, o) not in ex:
                    seg.setdefault("relations", []).append([s, p, o])
                    ex.add((s, p, o))
                    tr += 1

    # Speaker assignment
    diar_path = Path(f"data/diarization/{vid}.json")
    n_spk = 0
    if diar_path.exists():
        diar = json.load(open(diar_path))
        diar_segs = diar.get("segments", [])
        for seg in d["segments"]:
            seg_start_s = seg["start_ms"] / 1000
            seg_end_s = seg["end_ms"] / 1000
            speaker_time = defaultdict(float)
            for ds in diar_segs:
                ov_start = max(seg_start_s, ds["start"])
                ov_end = min(seg_end_s, ds["end"])
                if ov_end > ov_start:
                    speaker_time[ds["speaker"]] += ov_end - ov_start
            if speaker_time:
                ranked = sorted(speaker_time.items(), key=lambda x: -x[1])
                seg["speakers"] = [{"id": spk, "duration_s": round(dur, 1)} for spk, dur in ranked if dur > 0.3]
                seg["primary_speaker"] = ranked[0][0]
                n_spk += 1

    with open(f"data/video_indexes/{vid}.json", "w") as f:
        json.dump(d, f, indent=2)

    segs = d["segments"]
    caps = sum(1 for s in segs if s.get("visual_caption"))
    asr = sum(1 for s in segs if s.get("asr_transcript"))
    print(f"{vid[-25:]:25s} segs:{len(segs):4d} asr:{asr:4d} caps:{caps:4d} ent:{te:5d} rels:{tr:4d} spk:{n_spk:4d}")
