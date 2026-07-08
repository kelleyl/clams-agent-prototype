#!/usr/bin/env python3
"""Context-aware entity grounding with LLM disambiguation.

Pipeline per entity:
  1. extract groundable proper-noun candidates from layers.entities.items
     (PERSON/ORG from ASR/OCR; drops visual_captions noise)
  2. fetch top-k Wikipedia candidate articles (title + extract)
  3. assemble the broadcast transcript context around the mention
  4. ask an LLM (on aristotle vLLM, OpenAI-compatible endpoint) which
     candidate the broadcast refers to, or NONE
  5. keep the grounding only if the LLM picks a candidate

The naive top-hit grounder mis-grounds (e.g. "Thomas Vernon" -> 1822
entomologist). The LLM sees the 1980s farm-credit context and rejects it.

LLM runs on ARISTOTLE, never locally. Endpoint is configurable; default is a
localhost vLLM tunnel. Use --packages-only to build/inspect the evidence
packages with NO LLM call (safe to run anywhere with network).

Examples:
  # local, no LLM: inspect what would be sent for one NewsHour video
  python scripts/ground_entities_llm.py --packages-only \
      --only-video cpb-aacip-507-154dn40c26 --max-per-video 8 --verbose

  # on aristotle (vLLM up on :8001): real disambiguated grounding
  python scripts/ground_entities_llm.py --disambiguate \
      --vllm-url http://localhost:8001/v1 --model Qwen/Qwen3.5-27B
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

import requests
from utils.video_index import wikidata_lookup, _is_groundable_entity

IDX_DIR = Path("data/video_indexes")
OUT_DIR = Path("data/entity_grounding_llm")
UA = {"User-Agent": "CLAMSAgent/1.0 (research prototype; kelleylynch)"}


def norm(t):
    return " ".join((t or "").split()).lower()


def benchmark_video_ids(path):
    vids = set()
    for line in open(path):
        line = line.strip()
        if line:
            try:
                vids.add(json.loads(line)["video_id"])
            except Exception:
                pass
    return {v for v in vids if v}


def candidates_for_index(doc, types, src_layers, max_per_video):
    items = doc.get("layers", {}).get("entities", {}).get("items", []) or []
    agg = {}
    for e in items:
        typ = e.get("type")
        sl = (e.get("source_layer") or "").lower()
        txt = (e.get("text") or "").strip()
        if typ not in types:
            continue
        if not any(sl.startswith(s) or s in sl for s in src_layers):
            continue
        if not _is_groundable_entity(txt, typ):
            continue
        if len(txt.split()) < 2:          # require multi-token; first names mis-ground
            continue
        k = norm(txt)
        if not k:
            continue
        rec = agg.get(k)
        if rec is None:
            agg[k] = {"text": txt, "type": typ, "count": 1,
                      "start_ms": e.get("start_ms"), "end_ms": e.get("end_ms")}
        else:
            rec["count"] += 1
    ranked = sorted(agg.values(), key=lambda r: (-r["count"], r["text"].lower()))
    return ranked[:max_per_video] if max_per_video else ranked


def asr_layers(doc):
    return [k for k in doc.get("layers", {}) if k.lower().startswith("asr")]


def gather_context(doc, start_ms, end_ms, window_ms, max_chars=1200):
    if start_ms is None:
        return ""
    lo, hi = start_ms - window_ms, (end_ms or start_ms) + window_ms
    parts = []
    for lk in asr_layers(doc):
        layer = doc["layers"][lk]
        layer_items = layer.get("items", []) if isinstance(layer, dict) else layer
        for it in layer_items:
            if not isinstance(it, dict):
                continue
            s = it.get("start_ms"); e = it.get("end_ms", s)
            if s is None:
                continue
            if s < hi and (e or s) > lo:
                t = (it.get("text") or "").strip()
                if t:
                    parts.append((s, t))
    parts.sort()
    ctx = " ".join(t for _, t in parts)
    return ctx[:max_chars]


_WIKI_CACHE = {}
WIKI_CACHE_PATH = OUT_DIR / "_wiki_cache.json"


def load_wiki_cache():
    global _WIKI_CACHE
    if WIKI_CACHE_PATH.exists():
        try:
            _WIKI_CACHE = json.load(open(WIKI_CACHE_PATH))
        except Exception:
            _WIKI_CACHE = {}


def save_wiki_cache():
    try:
        json.dump(_WIKI_CACHE, open(WIKI_CACHE_PATH, "w"))
    except Exception:
        pass


def wikipedia_candidates(entity, k=3, retries=6):
    """Return up to k {title, extract} candidates via ONE MediaWiki call.

    Uses generator=search + prop=extracts (halves request volume vs separate
    search+extract). Retries with backoff on throttling. Cached on disk so the
    full run is resumable and gentle on the shared-IP rate limit.
    """
    key = norm(entity)
    if key in _WIKI_CACHE:
        return _WIKI_CACHE[key]
    out = None
    for attempt in range(retries):
        try:
            r = requests.get("https://en.wikipedia.org/w/api.php",
                             params={"action": "query", "generator": "search",
                                     "gsrsearch": entity, "gsrlimit": k,
                                     "prop": "extracts", "exintro": 1,
                                     "explaintext": 1, "exsentences": 2,
                                     "format": "json", "redirects": 1},
                             headers=UA, timeout=15)
            if r.status_code in (429, 500, 502, 503):
                raise requests.HTTPError(f"status {r.status_code}")
            r.raise_for_status()
            pages = list(r.json().get("query", {}).get("pages", {}).values())
            pages.sort(key=lambda p: p.get("index", 1e9))
            out = []
            for p in pages:
                title = p["title"]
                extract = (p.get("extract") or "").strip()
                low = extract.lower()
                # drop disambiguation / list pages; they are not real groundings
                if ("(disambiguation)" in title or " may refer to" in low
                        or " may also refer to" in low):
                    continue
                out.append({"title": title, "extract": extract})
            break
        except Exception as ex:
            if attempt == retries - 1:
                print(f"      ! wiki fetch failed for {entity!r}: {ex}", flush=True)
                out = None
            else:
                time.sleep(min(60, 5 * (2 ** attempt)))
    if out is None:
        return []   # do NOT cache failures, so a later run can retry
    _WIKI_CACHE[key] = out
    save_wiki_cache()
    return out


def llm_disambiguate(entity, etype, context, candidates, vllm_url, model, api_key,
                     broadcast="", timeout=60):
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"[{i}] {c['title']}: {c['extract'][:300]}")
    bline = f"Broadcast: {broadcast}\n" if broadcast else ""
    user = (f'Entity mentioned in an archival broadcast: "{entity}" (type {etype}).\n'
            f"{bline}"
            f"\nTranscript context around the mention:\n{context or '(no transcript context available)'}\n\n"
            f"Candidate Wikipedia articles:\n" + "\n".join(lines) + "\n\n"
            "Which candidate is the entity the broadcast refers to? "
            "Use the broadcast date and topic: reject any candidate that is "
            "anachronistic (e.g. created after the broadcast aired) or off-topic. "
            "Answer with only the candidate number, or NONE if none match.")
    sys_msg = ("You disambiguate named entities from archival broadcast transcripts "
               "against candidate Wikipedia articles. Be strict: if the broadcast "
               "context and date do not clearly match a candidate, answer NONE. "
               "Never choose a candidate that did not exist when the broadcast aired. "
               "Reply with only a number or the word NONE.")
    try:
        r = requests.post(f"{vllm_url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0,
                                "max_tokens": 8,
                                "messages": [{"role": "system", "content": sys_msg},
                                             {"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"].strip().upper()
    except Exception as ex:
        return None, f"llm_error:{ex}"
    if "NONE" in ans:
        return None, ans
    digits = "".join(ch for ch in ans if ch.isdigit())
    if digits:
        idx = int(digits) - 1
        if 0 <= idx < len(candidates):
            return idx, ans
    return None, ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-from", default="qa-data/benchmark/v5_2/benchmark_combined.jsonl")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only-video", default="", help="only process indexes whose stem contains this")
    ap.add_argument("--types", default="PERSON,ORG")
    ap.add_argument("--source-layers", default="asr,ocr")
    ap.add_argument("--max-per-video", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--context-window-ms", type=int, default=30000)
    ap.add_argument("--delay", type=float, default=0.3)
    # modes
    ap.add_argument("--packages-only", action="store_true",
                    help="build+save evidence packages (wiki candidates + context); NO LLM")
    ap.add_argument("--disambiguate", action="store_true", help="run LLM disambiguation (aristotle)")
    # llm (aristotle vLLM, OpenAI-compatible)
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8001/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-27B"))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip videos that already have a sidecar (resume / fill-in)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    types = {t.strip().upper() for t in args.types.split(",") if t.strip()}
    src = {s.strip().lower() for s in args.source_layers.split(",") if s.strip()}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    load_wiki_cache()

    wanted = None if args.all else benchmark_video_ids(args.videos_from)
    indexes = sorted(IDX_DIR.glob("*.json"))
    indexes = [p for p in indexes if wanted is None or p.stem in wanted]
    if args.only_video:
        indexes = [p for p in indexes if args.only_video in p.stem]
    if args.limit:
        indexes = indexes[:args.limit]

    mode = "PACKAGES-ONLY (no LLM)" if args.packages_only else \
           ("DISAMBIGUATE (LLM)" if args.disambiguate else "CANDIDATES-ONLY (no network)")
    print(f"{mode}: {len(indexes)} videos | types={sorted(types)} | "
          f"topk={args.topk}" + (f" | model={args.model} @ {args.vllm_url}" if args.disambiguate else ""),
          flush=True)

    stats = Counter()
    for vi, idx_path in enumerate(indexes):
        if args.skip_existing and (OUT_DIR / f"{idx_path.stem}.json").exists():
            stats["skipped_existing"] += 1
            continue
        doc = json.load(open(idx_path))
        bdate = doc.get("broadcast_date") or "unknown date"
        broadcast = f"program {doc.get('video_id', idx_path.stem)}, aired {bdate}"
        cands = candidates_for_index(doc, types, src, args.max_per_video)
        results = []
        for c in cands:
            pkg = {"text": c["text"], "type": c["type"], "count": c["count"],
                   "start_ms": c["start_ms"]}
            if args.packages_only or args.disambiguate:
                pkg["context"] = gather_context(doc, c["start_ms"], c["end_ms"],
                                                args.context_window_ms)
                pkg["candidates"] = wikipedia_candidates(c["text"], args.topk)
                time.sleep(args.delay)
                stats["packages"] += 1
            if args.disambiguate:
                if pkg["candidates"]:
                    idx_sel, raw = llm_disambiguate(c["text"], c["type"], pkg["context"],
                                                    pkg["candidates"], args.vllm_url,
                                                    args.model, args.api_key,
                                                    broadcast=broadcast)
                    pkg["llm_raw"] = raw
                    if idx_sel is not None:
                        chosen = pkg["candidates"][idx_sel]
                        pkg["grounded"] = True
                        pkg["title"] = chosen["title"]
                        pkg["wikipedia"] = chosen["extract"]
                        pkg["wikidata"] = wikidata_lookup(chosen["title"])
                        stats["grounded"] += 1
                    else:
                        pkg["grounded"] = False
                        stats["rejected"] += 1
                    time.sleep(args.delay)
                else:
                    pkg["grounded"] = False
                    stats["no_candidates"] += 1
            results.append(pkg)

        if args.packages_only or args.disambiguate:
            json.dump({"video_id": idx_path.stem, "mode": mode, "entities": results},
                      open(OUT_DIR / f"{idx_path.stem}.json", "w"), indent=2)

        line = f"  [{vi+1}/{len(indexes)}] {idx_path.stem[-32:]:32s} cand:{len(cands):4d}"
        if args.disambiguate:
            line += f" grounded:{stats['grounded']} rejected:{stats['rejected']}"
        print(line, flush=True)
        if args.verbose:
            for pkg in results[:6]:
                print(f"    · {pkg['text']!r} ({pkg['type']}, x{pkg['count']})")
                if "candidates" in pkg:
                    print(f"      ctx: {pkg.get('context','')[:160]!r}")
                    for j, c in enumerate(pkg["candidates"], 1):
                        print(f"      [{j}] {c['title']}: {c['extract'][:120]}")
                if "llm_raw" in pkg:
                    print(f"      LLM -> {pkg['llm_raw']}  grounded={pkg.get('grounded')}")

    print(f"\n=== {dict(stats)} ===")
    if args.packages_only or args.disambiguate:
        print(f"output -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
