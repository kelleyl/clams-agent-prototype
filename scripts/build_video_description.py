#!/usr/bin/env python3
"""Build a dense, temporally-ordered natural-language DESCRIPTION of a video
from its multi-layer index (the "serialized index as prose" experiment).

Two stages, both written to data/video_descriptions/<vid>.json:

  1. SCAFFOLD (deterministic, always faithful by construction): for each chapter
     beat, interleave who is speaking (speakers layer, resolved names + text),
     what is visually on screen (VLM general-scene caption, kept as a HEDGED
     observation), and on-screen text (text-focus caption / OCR). No LLM, so it
     cannot hallucinate -- it is a strict rendering of index contents.

  2. PROSE (optional, --llm): dense-paraphrase each scaffold beat into fluent
     "now person A is on screen, they said Y" narration under a strict
     rewrite-only / add-nothing prompt. Runs on the aristotle LLM.

The scaffold is the ground truth for faithfulness auditing; the prose is what a
downstream generator would read. Round-trip QA must still verify answers against
the RAW index, not this description.

Inputs (on disk):
  data/video_indexes/<vid>.json         (required)
  data/salience_maps/<vid>.json         (optional -- resolved participant names)
  data/entity_grounding_llm/<vid>.json  (optional -- grounded entities cast list)

Examples:
  # deterministic scaffold only, no LLM (safe to run anywhere)
  python scripts/build_video_description.py --only-video cpb-aacip-507-3x83j39n00
  # + LLM prose pass on aristotle
  python scripts/build_video_description.py --only-video cpb-aacip-507-3x83j39n00 \
      --llm --url http://localhost:11434/v1 --model gemma3:27b-it-qat
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import requests

IDX_DIR = Path("data/video_indexes")
SAL_DIR = Path("data/salience_maps")
GRD_DIR = Path("data/entity_grounding_llm")
OUT_DIR = Path("data/video_descriptions")

GENERIC_SPK = re.compile(r"^(SPEAKER[_\s]?\d+|spk[\w]*\d+|unknown)", re.I)


def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def ms_to_ts(ms):
    s = int((ms or 0) / 1000)
    return f"{s // 60:02d}:{s % 60:02d}"


def overlaps(it, a, b):
    s = it.get("start_ms")
    if s is None:
        return False
    e = it.get("end_ms") or s
    return s < b and e > a


def named(spk_name):
    """Return a display name, or None if the speaker id is a generic placeholder."""
    if not spk_name or GENERIC_SPK.match(str(spk_name).strip()):
        return None
    return str(spk_name).strip()


def clean_caption(txt):
    """VLM general-scene captions carry chain-of-thought ('2. **Identify...**').
    Strip enumeration/markdown scaffolding and keep the descriptive prose."""
    if not txt:
        return ""
    txt = re.sub(r"\*\*[^*]+\*\*", "", txt)            # bold headers
    txt = re.sub(r"^\s*\d+\.\s*", " ", txt, flags=re.M)  # "2. " list markers
    txt = txt.replace("\n", " ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:400]


def merge_turns(spk_items):
    """Merge consecutive same-speaker segments into single turns (start-sorted)."""
    turns = []
    for s in sorted(spk_items, key=lambda x: x.get("start_ms", 0)):
        who = named(s.get("speaker_name") or s.get("speaker_id"))
        text = (s.get("text") or "").strip()
        if not text:
            continue
        if turns and turns[-1]["who"] == who:
            turns[-1]["text"] += " " + text
            turns[-1]["end_ms"] = s.get("end_ms", turns[-1]["end_ms"])
        else:
            turns.append({"who": who, "text": text,
                          "start_ms": s.get("start_ms", 0), "end_ms": s.get("end_ms", 0)})
    return turns


def build_scaffold_beat(ch, L, cast):
    """Deterministic, faithful scaffold lines for one chapter beat."""
    a, b = ch["start_ms"], ch["end_ms"]
    lines = [f"[{ms_to_ts(a)}-{ms_to_ts(b)}] CHAPTER: {ch.get('title', '(untitled)')}"]

    # who speaks + what they say (merged turns)
    spk = [s for s in items(L.get("speakers")) if overlaps(s, a, b)]
    for t in merge_turns(spk):
        who = t["who"] or "An unidentified speaker"
        say = t["text"][:600]
        lines.append(f"  SPEECH -- {who}: \"{say}\"")

    # on-screen text (OCR / text-focus captions); dedup consecutive repeats
    last_ocr = None
    for c in items(L.get("caption_qwen3vl-8b_text_focus")):
        if overlaps(c, a, b):
            txt = (c.get("text") or "").strip()
            if txt and txt != last_ocr:
                lines.append(f"  ON-SCREEN TEXT: \"{txt[:200]}\"")
                last_ocr = txt

    # visual observation (VLM general-scene caption) -- HEDGED, one per beat max
    vis = [c for c in items(L.get("caption_qwen3vl-8b_general_scene")) if overlaps(c, a, b)]
    if vis:
        vc = clean_caption(vis[len(vis) // 2].get("text"))   # a mid-beat frame
        if vc:
            lines.append(f"  VISUAL (model estimate, may be uncertain): {vc}")

    return {"start_ms": a, "end_ms": b, "title": ch.get("title"),
            "scaffold": "\n".join(lines)}


DP_SYS = (
    "You convert a structured multimodal index of one segment of an archival TV broadcast "
    "into a faithful, fluent prose description of what happens. Write 2 to 5 natural sentences "
    "in present-tense narration ('Now X is on screen and says ...'; 'A chyron reads ...'). "
    "STRICT RULES: use ONLY facts present in the structured input. Do NOT add any visual detail, "
    "identity, event, date, or fact that is not in the input. Do NOT guess or infer beyond it. "
    "Preserve every name, quotation, number, and on-screen text string exactly as given. Items "
    "marked VISUAL are an uncertain automatic estimate -- if you use one, keep it hedged "
    "('appears to show'). Output only the prose, no preamble."
)


def chunk_scaffold(scaffold, max_words=300):
    """Split a beat scaffold into <=max_words chunks on line boundaries (never
    mid-turn), so each DP rewrite is near 1:1 rather than a lossy summary. The
    CHAPTER header rides on the first chunk; later chunks get a (cont.) marker."""
    lines = scaffold.split("\n")
    header = lines[0] if lines and lines[0].startswith("[") else ""
    body = lines[1:] if header else lines
    chunks, cur, cw = [], [], 0
    for ln in body:
        w = len(ln.split())
        if cur and cw + w > max_words:
            chunks.append(cur)
            cur, cw = [], 0
        cur.append(ln)
        cw += w
    if cur:
        chunks.append(cur)
    out = []
    for i, c in enumerate(chunks):
        head = header if i == 0 else (header.split("]")[0] + "] (cont.)" if header else "")
        out.append("\n".join(([head] if head else []) + c))
    return out or [scaffold]


def dp_beat(scaffold, url, model, api_key, timeout=120):
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0.2, "max_tokens": 400,
                                "messages": [{"role": "system", "content": DP_SYS},
                                             {"role": "user", "content": scaffold}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
    except Exception as ex:
        return f"__ERR__:{ex}"


def build_cast(vid, L):
    """A resolved cast list (named speakers + grounded entities) for context/audit."""
    cast = {}
    for s in items(L.get("speakers")):
        nm = named(s.get("speaker_name") or s.get("speaker_id"))
        if nm:
            cast[nm] = cast.get(nm, 0) + 1
    grd = GRD_DIR / f"{vid}.json"
    ents = []
    if grd.exists():
        g = json.load(open(grd))
        ents = [e.get("text") for e in g.get("entities", []) if e.get("grounded") or e.get("candidates")]
    return {"speakers": sorted(cast, key=lambda k: -cast[k]), "entities": ents[:20]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-video", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--llm", action="store_true", help="run the DP prose pass (needs a server)")
    ap.add_argument("--url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="gemma3:27b-it-qat")
    ap.add_argument("--chunk-words", type=int, default=300,
                    help="max scaffold words per DP call (keeps rewrites ~1:1)")
    ap.add_argument("--api-key", default="EMPTY")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    idx_files = sorted(IDX_DIR.glob("*.json"))
    if args.only_video:
        idx_files = [p for p in idx_files if args.only_video in p.stem]
    elif not args.all:
        print("specify --only-video <id> or --all", file=sys.stderr)
        sys.exit(2)

    for ip in idx_files:
        vid = ip.stem
        doc = json.load(open(ip))
        L = doc.get("layers", {})
        chapters = sorted(items(L.get("chapters")), key=lambda c: c.get("start_ms", 0))
        if not chapters:
            print(f"{vid}: no chapters, skipping", flush=True)
            continue
        cast = build_cast(vid, L)
        beats = [build_scaffold_beat(ch, L, cast) for ch in chapters]

        if args.llm:
            for i, bt in enumerate(beats):
                subs = chunk_scaffold(bt["scaffold"], args.chunk_words)
                proses = [dp_beat(s, args.url, args.model, args.api_key) for s in subs]
                bt["prose"] = " ".join(proses)
                bt["n_chunks"] = len(subs)
                print(f"  {vid} beat {i + 1}/{len(beats)} dp done ({len(subs)} chunk(s))", flush=True)

        out = {"video_id": vid, "broadcast_date": doc.get("broadcast_date"),
               "cast": cast, "n_beats": len(beats), "beats": beats,
               "model": args.model if args.llm else None}
        json.dump(out, open(OUT_DIR / f"{vid}.json", "w"), indent=2)
        wc = sum(len(b["scaffold"].split()) for b in beats)
        print(f"{vid}: {len(beats)} beats, ~{wc} scaffold words, "
              f"{'PROSE' if args.llm else 'scaffold only'} -> {OUT_DIR}/{vid}.json", flush=True)


if __name__ == "__main__":
    main()
