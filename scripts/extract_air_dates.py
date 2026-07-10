#!/usr/bin/env python3
"""Extract per-video air dates from filename slugs and slate OCR captions.

Sources, in confidence order:
  slug   - dates embedded in slug filenames (ABC_..._3_9_1980)
  slate  - text_focus captions on SWT Slate frames (AIR: 3/11/87, DATE: 7/08/04,
           Dec 29, 2000 forms)

Writes data/video_air_dates.json: {video_id: {date: 'YYYY-MM-DD'|'YYYY', source,
raw}}. Deterministic; no LLM.
"""
import glob
import json
import re

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}


def norm_year(y):
    y = int(y)
    if y < 100:
        y += 1900 if y > 30 else 2000
    return y


def from_slug(vid):
    m = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})", vid)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12:
            return f"{y:04d}-{mo:02d}-{d:02d}", vid
    m = re.search(r"\b(19|20)(\d{2})\b", vid)
    if m:
        return f"{m.group(1)}{m.group(2)}", vid
    return None, None


def from_text(t):
    t = t or ""
    m = re.search(r"(?:AIR|DATE|REC)[:\s]+(\d{1,2})/(\d{1,2})/(\d{2,4})", t, re.I)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), norm_year(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    m = re.search(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})", t)
    if m and m.group(1)[:3].lower() in MONTHS:
        return f"{int(m.group(3)):04d}-{MONTHS[m.group(1)[:3].lower()]:02d}-{int(m.group(2)):02d}"
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", t)
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), norm_year(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def main():
    out = {}
    for f in sorted(glob.glob("data/video_indexes/*.json")):
        vid = f.split("/")[-1][:-5]
        date, raw = from_slug(vid)
        if date:
            out[vid] = {"date": date, "source": "slug", "raw": raw[:60]}
            continue
        doc = json.load(open(f))
        lay = doc.get("layers", {}).get("caption_qwen3vl-8b_text_focus")
        items = lay.get("items", []) if isinstance(lay, dict) else (lay or [])
        for it in items:
            if (it.get("scene_label") or "").lower() not in ("slate", "credits"):
                continue
            d = from_text(it.get("text"))
            if d:
                out[vid] = {"date": d, "source": "slate",
                            "raw": (it.get("text") or "")[:80]}
                break
    json.dump(out, open("data/video_air_dates.json", "w"), indent=1)
    full = sum(1 for v in out.values() if len(v["date"]) == 10)
    print(f"air dates: {len(out)}/108 videos ({full} full dates, "
          f"{len(out) - full} year-only)")


if __name__ == "__main__":
    main()
