#!/usr/bin/env python3
"""Build the corpus catalog for v6 exploration/retrieval questions.

Aggregates every salience map + entity grounding into one searchable table:
one record per video with its salient segments and its named participants
(grounded to Wikipedia/Wikidata where the LLM disambiguator matched, with
time-invariant fields like Occupation parsed out). Pure aggregation, no LLM.

The exploration generator (scripts/generate_qa_exploration.py) derives both
program-level ("which segments of this broadcast discuss X") and corpus-level
("find interviews with scientists on climate") candidate sets from this file.

Usage:
  python scripts/build_corpus_catalog.py            # -> data/corpus_catalog.json
  python scripts/build_corpus_catalog.py --stats    # print coverage stats only
"""
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

SAL_DIR = Path("data/salience_maps")
GRD_DIR = Path("data/entity_grounding_llm")
IDX_DIR = Path("data/video_indexes")
PROV = Path("data/v5_1_video_provenance.json")
OUT = Path("data/corpus_catalog.json")

NON_CONTENT_TITLES = {"introduction", "closing", "credits", "intro", "opening",
                      "end credits", "bumper", "promo"}

# time-invariant Wikidata fields, safe over old broadcasts
# (same policy as generate_qa_needdown.py)
KEEP_FIELDS = ["Occupation", "Field of work", "Educated at", "Industry"]


def parse_wd_fields(wd):
    """Parse the flattened 'Field: v1, v2. Field2: v3.' wikidata string
    (duplicated from scripts/generate_qa_needdown.py, which chdirs on import)."""
    fields = {}
    for part in re.split(r"\.\s+", wd or ""):
        m = re.match(r"\s*([A-Za-z ]+):\s*(.+)", part)
        if m:
            fields[m.group(1).strip()] = [v.strip() for v in m.group(2).split(",") if v.strip()]
    return fields


_STOP = set("the a an of and in on to for with his her its this that".split())


def _content_toks(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return [w for w in s.split() if len(w) > 3 and w not in _STOP]


def title_coherence(doc, start_ms, end_ms, title):
    """Fraction of the chapter title's content tokens that occur in the ASR of
    its own time window. Low values indicate misaligned upstream chaptering
    (observed: titles shifted relative to spans on some videos), which poisons
    any prompt or answer that trusts the title."""
    q = _content_toks(title)
    if not q:
        return None
    words = set()
    for k, lay in doc.get("layers", {}).items():
        if not k.lower().startswith("asr"):
            continue
        for it in (lay.get("items", []) if isinstance(lay, dict) else lay):
            if not isinstance(it, dict) or it.get("start_ms") is None:
                continue
            s, e = it["start_ms"], it.get("end_ms") or it["start_ms"]
            if s < end_ms and e > start_ms:
                words.update(_content_toks(it.get("text")))
    return round(sum(1 for w in q if w in words) / len(q), 3)


def norm_name(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).strip()


def name_match(a, b):
    """Token containment either way: 'Jim Lehrer' ~ 'JIM LEHRER', 'Lehrer'."""
    ta, tb = set(norm_name(a).split()), set(norm_name(b).split())
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    prov = json.load(open(PROV)) if PROV.exists() else {}
    videos = {}
    occ_vocab = Counter()
    n_segments = n_parts = n_grounded_parts = 0

    for sp in sorted(SAL_DIR.glob("*.json")):
        vid = sp.stem
        sal = json.load(open(sp))
        grdp = GRD_DIR / f"{vid}.json"
        grounded = [e for e in (json.load(open(grdp)).get("entities", []) if grdp.exists() else [])
                    if e.get("grounded")]

        idxp = IDX_DIR / f"{vid}.json"
        doc = json.load(open(idxp)) if idxp.exists() else {}

        segments = []
        for s in sal.get("salient_segments", []):
            title = (s.get("title") or "").strip()
            segments.append({
                "seg_id": s.get("id"),
                "title": title,
                "content": title.lower() not in NON_CONTENT_TITLES,
                "start_ms": s.get("start_ms"),
                "end_ms": s.get("end_ms"),
                "dur_ms": s.get("dur_ms"),
                "dominant_speaker": s.get("dominant_speaker"),
                "snippet": (s.get("snippet") or "")[:200],
                "title_coherence": title_coherence(doc, s.get("start_ms") or 0,
                                                   s.get("end_ms") or 0, title),
            })
        n_segments += len(segments)
        coh = [s["title_coherence"] for s in segments
               if s["title_coherence"] is not None and s["content"]]
        video_coherence = round(sum(coh) / len(coh), 3) if coh else None

        participants = []
        for p in sal.get("main_participants", []):
            name = p.get("name", "")
            if name.upper().startswith("SPEAKER_"):
                continue
            rec = {
                "name": name,
                "kind": p.get("kind"),
                "speaking_ms": p.get("speaking_ms", 0),
                "mention_count": p.get("mention_count", 0),
            }
            for e in grounded:
                if e.get("type") == "PERSON" and name_match(name, e.get("text", "")):
                    wd = e.get("wikidata") or ""
                    fields = parse_wd_fields(wd)
                    rec["grounded"] = {
                        "canonical": e.get("title"),
                        "wikipedia": e.get("wikipedia"),
                        # description head ("American poet, author, and civil
                        # rights activist (1928-2014)") corroborates noisy
                        # occupation lists downstream
                        "description": re.split(r"\.\s+[A-Z][a-z]+ ?[a-z]*:", wd)[0][:200],
                        "fields": {k: v for k, v in fields.items() if k in KEEP_FIELDS},
                    }
                    occ_vocab.update(o.lower() for o in fields.get("Occupation", []))
                    break
            participants.append(rec)
        n_parts += len(participants)
        n_grounded_parts += sum(1 for p in participants if p.get("grounded"))

        # grounded ORG/PERSON mentions with a first-mention timestamp, for
        # entity-scoped retrieval questions
        entities = [{
            "text": e.get("text"),
            "type": e.get("type"),
            "canonical": e.get("title"),
            "count": e.get("count", 0),
            "start_ms": e.get("start_ms"),
        } for e in grounded]

        pv = prov.get(vid, {}) if isinstance(prov, dict) else {}
        videos[vid] = {
            "video_id": vid,
            "broadcast_date": sal.get("broadcast_date"),
            "duration_ms": sal.get("duration_ms"),
            "provenance": {
                "source": pv.get("source") or pv.get("collection"),
                "genre": pv.get("primary_genre"),
                "era": pv.get("era_estimate"),
            },
            "scene_labels": sal.get("scene_labels", {}),
            "title_coherence": video_coherence,
            "segments": segments,
            "participants": participants,
            "entities": entities,
        }

    catalog = {"n_videos": len(videos), "videos": videos}
    if not args.stats:
        json.dump(catalog, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")

    print(f"videos: {len(videos)} | segments: {n_segments} "
          f"(content: {sum(1 for v in videos.values() for s in v['segments'] if s['content'])})")
    print(f"named participants: {n_parts} | grounded: {n_grounded_parts}")
    print("top occupations:", occ_vocab.most_common(15))
    low = sorted(((v["title_coherence"], vid) for vid, v in videos.items()
                  if v["title_coherence"] is not None and v["title_coherence"] < 0.5))
    print(f"\nlow title-coherence videos (<0.5, chapter titles unreliable): {len(low)}")
    for c, vid in low[:15]:
        print(f"  {c:.2f}  {vid[:70]}")


if __name__ == "__main__":
    main()
