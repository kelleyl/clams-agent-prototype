#!/usr/bin/env python3
"""Exploration/retrieval question generation (v6, the missing highest-value type).

Two scopes, both with DETERMINISTIC candidate sets from the corpus catalog and
per-item LLM verification (the inverse of asking an LLM to invent an answer):

  program-level  "In which segments of this broadcast is X discussed?"
                 X = a recurring grounded entity or an LLM-proposed cross-segment
                 theme; candidates = segments whose ASR/title matches X.
  corpus-level   "Find broadcasts where a <occupation> speaks [about <topic>]."
                 candidates = grounded participants with that occupation.

Answers are retrieval SETS ({video_id, seg_id, title, start_ms, end_ms}),
format "retrieval_set", scored downstream by set F1.

Gates (empirical, decoupled families; generator gemma3:27b, verifier llama3.3):
  1. per-item verification  - verifier confirms each candidate against its ASR
                              window ("discussed, not a passing mention").
  2. size bounds            - keep 2-10 verified items.
  3. set round-trip         - verifier re-derives the set from a shuffled pool
                              of gold + distractor segments; keep if F1 >= 0.8.
  4. completeness sampling  - verify a sample of near-miss EXCLUDED segments;
                              any hit -> completeness_risk (excluded from the
                              benchmark keep-set, kept as data).
  Blind panel is inapplicable by construction (set answers over an obscure
  corpus); rows carry necessity: "robust_by_construction".

Usage:
  python scripts/generate_qa_exploration.py --dry-run --only-video cpb-aacip-507-1v5bc3tf81 --verbose
  python scripts/generate_qa_exploration.py --scope program --only-video <vid> \
      --gen-url http://localhost:11434/v1 --gen-model gemma3:27b-it-qat \
      --verify-url http://localhost:11434/v1 --verify-model llama3.3:latest
  python scripts/generate_qa_exploration.py --scope corpus --limit-questions 30
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")
import requests

from utils.ctx_retrieval import _toks  # shared tokenizer/stoplist

CATALOG = Path("data/corpus_catalog.json")
IDX_DIR = Path("data/video_indexes")
OUT_DIR = Path("data/qa_exploration")

RNG = random.Random(6)  # deterministic sampling


# ---------- shared LLM helpers ----------
def chat(url, model, api_key, sysm, user, temperature=0.2, max_tokens=400, timeout=120):
    try:
        msgs = ([{"role": "system", "content": sysm}] if sysm else []) + \
               [{"role": "user", "content": user}]
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": temperature,
                                "max_tokens": max_tokens, "messages": msgs},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return txt.split("</think>")[-1].strip() if "</think>" in txt else txt.strip()
    except Exception as ex:
        return f"__ERR__:{ex}"


def json_out(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


# ---------- index text access ----------
def _items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def asr_window(doc, start_ms, end_ms, max_chars=1800):
    parts = []
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        for it in _items(doc["layers"][k]):
            if not isinstance(it, dict):
                continue
            s, e = it.get("start_ms"), it.get("end_ms")
            if s is None:
                continue
            if s < end_ms and (e or s) > start_ms:
                t = (it.get("text") or "").strip()
                if t:
                    parts.append((s, t))
    parts.sort()
    return " ".join(t for _, t in parts)[:max_chars]


def seg_excerpt(doc, seg, topic, max_chars=320):
    """Topic-focused excerpt from within one segment's ASR window, so the
    set round-trip judges from the same information basis as item verification."""
    text = asr_window(doc, seg["start_ms"], seg["end_ms"], 100000)
    q = set(_toks(topic))
    best, best_score = "", -1
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        score = len(q & set(_toks(chunk)))
        if score > best_score and len(chunk) > 40:
            best, best_score = chunk, score
    return best[:max_chars]


def topic_ctx(doc, seg, topic, max_chars=1800):
    """Topic-focused verification context: the most topic-relevant ASR turns of
    the WHOLE segment window, time-ordered. Segments are chapter-sized (minutes),
    so a head-truncated window routinely misses the topic entirely."""
    turns = []
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        for it in _items(doc["layers"][k]):
            if not isinstance(it, dict) or it.get("start_ms") is None:
                continue
            s, e = it["start_ms"], it.get("end_ms") or it["start_ms"]
            if s < seg["end_ms"] and e > seg["start_ms"]:
                t = (it.get("text") or "").strip()
                if t:
                    turns.append((s, t))
    turns.sort()
    q = set(_toks(topic))
    scored = sorted(range(len(turns)),
                    key=lambda i: -len(q & set(_toks(turns[i][1]))))
    picked = set()
    for i in scored[:8]:
        picked.update({i - 1, i, i + 1})  # neighbors for continuity
    ordered = [turns[i][1] for i in sorted(p for p in picked if 0 <= p < len(turns))]
    out = " ".join(ordered)[:max_chars]
    return out if out else " ".join(t for _, t in turns)[:max_chars]


def seg_matches(doc, seg, query_toks):
    """Lexical hit rate of query tokens inside a segment's FULL ASR window.
    Titles are deliberately excluded: chapter titles are unreliable on some
    videos (misaligned upstream chaptering)."""
    text = asr_window(doc, seg["start_ms"], seg["end_ms"], 100000)
    toks = set(_toks(text))
    if not query_toks:
        return 0.0
    return len(set(query_toks) & toks) / len(set(query_toks))


# ---------- target construction ----------
def program_targets(vid, cat_video, doc, max_targets):
    """Entity-recurrence and theme targets for one video."""
    targets = []
    segs = [s for s in cat_video["segments"] if s["content"] and (s.get("dur_ms") or 0) > 45000]
    if len(segs) < 3:
        return targets  # retrieval over <3 segments is trivial

    # entity-based: recurring grounded entities (count >= 3), deduped by canonical
    seen_canonical = set()
    for e in cat_video.get("entities", []):
        if (e.get("count") or 0) < 3 or not e.get("canonical"):
            continue
        if e["canonical"] in seen_canonical:
            continue
        seen_canonical.add(e["canonical"])
        q_toks = _toks(e["text"])
        cands = [s for s in segs if seg_matches(doc, s, q_toks) >= 0.99]  # all tokens present
        # multi-segment but not saturated: a topic present in most segments (the
        # broadcast's central figure) has fuzzy boundaries and makes a bad
        # retrieval question
        if 2 <= len(cands) <= max(2, len(segs) // 2):
            targets.append({"kind": "entity", "topic": e["canonical"],
                            "surface": e["text"], "candidates": cands,
                            "pool": segs})
        if len(targets) >= max_targets:
            return targets

    # theme-based placeholder: theme proposal happens at generation time (LLM);
    # here we record one theme slot per video with the title list as input.
    # Titles feed the theme proposal, so gate on title coherence (30/108 videos
    # have chapter titles misaligned with their spans).
    if (cat_video.get("title_coherence") or 0) >= 0.5:
        targets.append({"kind": "theme", "topic": None,
                        "titles": [s["title"] for s in segs], "pool": segs})
    return targets[:max_targets]


def corpus_targets(catalog, max_questions):
    """Occupation targets across the whole corpus. Wikidata occupation lists are
    noisy (Maya Angelou lists 'politician'), so an occupation counts only if it
    is CORROBORATED by the entity's description head ('American poet, author,
    and civil rights activist ...')."""
    by_occ = defaultdict(list)
    occ_any = defaultdict(set)   # occupation -> videos where ANY listed occ matches
    for vid, v in catalog["videos"].items():
        for p in v["participants"]:
            g = p.get("grounded")
            if not g:
                continue
            occs = [o.lower() for o in (g.get("fields", {}).get("Occupation") or [])]
            desc = set(_toks(g.get("description") or ""))
            plausible = [o for o in occs if set(_toks(o)) & desc]
            for occ in occs:
                occ_any[occ].add(vid)   # exclusion stays conservative: ANY listed occ
            for occ in plausible:
                by_occ[occ].append({
                    "video_id": vid, "person": g.get("canonical") or p["name"],
                    "name": p["name"], "speaking_ms": p.get("speaking_ms", 0),
                    "kind": p.get("kind"), "description": g.get("description") or "",
                })
    # videos with at least one grounded participant: the catalog-covered corpus
    covered = {vid for vid, v in catalog["videos"].items()
               if any(p.get("grounded") for p in v["participants"])}
    targets = []
    # skip presenter-ish occupations: every broadcast has one, sets get huge/trivial
    boring = {"news presenter", "television presenter", "journalist"}
    for occ, hits in sorted(by_occ.items(), key=lambda kv: -len(kv[1])):
        if occ in boring:
            continue
        speak = [h for h in hits if h["speaking_ms"] > 30000]
        vids = {h["video_id"] for h in speak}
        if 2 <= len(vids) <= 10:
            targets.append({"kind": "occupation", "occupation": occ, "hits": speak,
                            # distractors must be catalog-covered and free of the
                            # occupation under ANY listed value (conservative)
                            "distractor_vids": sorted(covered - occ_any[occ])})
        if len(targets) >= max_questions:
            break
    return targets


# ---------- verification (llama3.3, decoupled from the generator) ----------
def verify_segment(doc, seg, topic, vcfg, kind="theme"):
    # topic-focused context over the whole window; chapter titles are NOT shown
    # (they are unreliable on some videos)
    ctx = topic_ctx(doc, seg, topic)
    if kind == "entity":
        spk = seg.get("dominant_speaker") or ""
        spk_line = (f"Diarization labels this segment's dominant speaker as: {spk}\n"
                    if spk and not spk.upper().startswith("SPEAKER_") else "")
        # retrieval semantics: the entity SPEAKS, is a SUBJECT, or is only PASSING
        user = ("Below are the most relevant transcript excerpts of one segment of an "
                "archival broadcast. A researcher is looking for segments substantively "
                f"involving '{topic}' - where they speak, are interviewed, or are a main "
                "subject of the segment. A single passing mention does not count. "
                'Reply JSON: {"role": "speaks" | "subject" | "passing" | "absent"}. '
                f"/no_think\n\n{spk_line}Transcript excerpts:\n{ctx}")
        out = json_out(chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                            temperature=0, max_tokens=64))
        return out.get("role") in ("speaks", "subject")
    user = ("Below are the most relevant transcript excerpts of one segment of an "
            f"archival broadcast. A researcher is looking for segments that cover "
            f"'{topic}'. Would they consider this segment relevant - is a substantial "
            "part of it about that topic (not merely a single passing mention)? "
            'Reply JSON: {"discussed": true|false}. /no_think\n\n'
            f"Transcript excerpts:\n{ctx}")
    out = json_out(chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                        temperature=0, max_tokens=64))
    return bool(out.get("discussed"))


def verify_occ_segment(doc, seg, occupation, people, vcfg):
    """Occupation-level check on a DISTRACTOR video's segment: does anyone of
    this occupation speak here? A positive means the catalog missed someone
    (grounding hole) -> completeness risk for the corpus question."""
    ctx = topic_ctx(doc, seg, occupation + " " + (people or ""))
    ppl_line = f"Named participants (diarization): {people}\n" if people else ""
    user = ("Below are transcript excerpts of one segment of an archival broadcast. "
            f"Using the excerpts AND your knowledge of who the named people are, does "
            f"a {occupation} speak substantively in it? Reply JSON: "
            '{"present": true|false}. /no_think\n\n'
            f"{ppl_line}Transcript excerpts:\n{ctx}")
    out = json_out(chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                        temperature=0, max_tokens=64))
    return bool(out.get("present"))


def verify_identity(doc, seg, canonical, description, vcfg):
    """Identity-plausibility check on a grounded participant: does the catalog's
    identification square with what this person actually talks about? Catches
    wrong-namesake grounding errors (observed: NewsHour civil-rights figure
    Roger Wilkins grounded to the Australian economist Roger Wilkins)."""
    ctx = topic_ctx(doc, seg, canonical)
    user = ("A catalog identifies a speaker in this archival US broadcast as: "
            f"{canonical} - {description}. Based on the transcript excerpts below "
            "(their topics, era, and context), is this identification PLAUSIBLE? "
            'Reply JSON: {"plausible": true|false}. /no_think\n\n'
            f"Transcript excerpts:\n{ctx}")
    out = json_out(chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                        temperature=0, max_tokens=64))
    return bool(out.get("plausible"))


def verify_person_segment(doc, seg, person, occupation, vcfg):
    ctx = topic_ctx(doc, seg, person)
    spk = seg.get("dominant_speaker") or ""
    spk_line = (f"Diarization labels this segment's dominant speaker as: {spk}\n"
                if spk and not spk.upper().startswith("SPEAKER_") else "")
    user = ("Below are the most relevant transcript excerpts of one segment of an "
            f"archival broadcast. Does {person} (a {occupation}) actually speak or get "
            "substantively featured in it? Reply JSON: {\"featured\": true|false}. "
            f"/no_think\n\n{spk_line}Transcript excerpts:\n{ctx}")
    out = json_out(chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                        temperature=0, max_tokens=64))
    return bool(out.get("featured"))


def set_roundtrip(question, gold_items, pool_items, vcfg, granularity="item"):
    """Verifier re-derives the answer set from a shuffled candidate pool.
    granularity="video" compares VIDEO sets: corpus questions ask 'which
    broadcasts...', so the video is the natural answer unit and the verifier
    replies with video ids."""
    pool = list(pool_items)
    RNG.shuffle(pool)
    # NOTE: chapter titles are deliberately omitted (unreliable on some videos);
    # the excerpt + speaker/people carry the signal
    listing = "\n".join(
        f"- {p['item_id']}: [{p['start_ms'] // 60000}m-{(p['end_ms'] or 0) // 60000}m]"
        + (f" (speaker: {p['speaker']})" if p.get("speaker") else "")
        + (f" (people: {p['people']})" if p.get("people") else "")
        + f" :: {p.get('excerpt') or p['snippet']}"
        for p in pool)
    user = ("You are selecting items from a catalog of broadcast segments. "
            f"Question: {question}\n\nCatalog:\n{listing}\n\n"
            "Use your own knowledge of who the named people are (their professions and "
            "roles) in addition to the excerpts. Reply with ONLY a JSON object, no "
            "explanation, listing the ids of ALL matching items (and only those): "
            '{"ids": ["..."]}. /no_think')
    raw = chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
               temperature=0, max_tokens=600)
    if os.environ.get("EXPLORE_DEBUG"):
        print(f"    [rt-debug] listing[:300]={listing[:300]!r}")
        print(f"    [rt-debug] raw={raw[:300]!r}")
    out = json_out(raw)

    def norm(i):
        return i.split(":")[0] if granularity == "video" else i

    got = {norm(i) for i in (out.get("ids") or [])}
    gold = {norm(g["item_id"]) for g in gold_items}
    if not got:
        return 0.0, []
    tp = len(got & gold)
    prec = tp / len(got)
    rec = tp / len(gold) if gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return round(f1, 3), sorted(got)


# ---------- question phrasing (gemma3:27b) ----------
def phrase_question(kind, payload, gcfg):
    sysm = ("You write ONE natural retrieval question for an archival-broadcast benchmark. "
            "The answer is a SET of segments/broadcasts (given), so the question must ask "
            "the reader to FIND or LIST them. Do not reveal how many there are, do not "
            "enumerate the answers, do not begin with 'Based on'. "
            'Return JSON: {"question": "..."}.')
    out = json_out(chat(gcfg["url"], gcfg["model"], gcfg["api_key"], sysm, payload,
                        temperature=0.7, max_tokens=150))
    return out.get("question")


def propose_theme(titles, gcfg):
    user = ("Segment titles of one archival broadcast:\n- " + "\n- ".join(titles) +
            "\n\nName ONE concrete theme or topic that connects SOME (2 or more, not all) "
            "of these segments - specific enough that a person could check each segment "
            'for it. Return JSON: {"theme": "..."}.')
    out = json_out(chat(gcfg["url"], gcfg["model"], gcfg["api_key"], "", user,
                        temperature=0.7, max_tokens=100))
    return out.get("theme")


# ---------- assembly ----------
def item_of(vid, seg, doc=None, topic=None, people=None):
    it = {"item_id": f"{vid}:{seg['seg_id']}", "video_id": vid,
          "seg_id": seg["seg_id"], "title": seg.get("title"),
          "start_ms": seg["start_ms"], "end_ms": seg["end_ms"],
          "snippet": (seg.get("snippet") or "")[:120]}
    if doc is not None and topic:
        it["excerpt"] = seg_excerpt(doc, seg, topic)
    if people:
        it["people"] = people
    spk = seg.get("dominant_speaker") or ""
    if spk and not spk.upper().startswith("SPEAKER_"):
        it["speaker"] = spk
    return it


def finish_row(row, gold, pool, near_miss_hits, checked, f1, rt_ids):
    row["qa"]["answer_set"] = gold
    row["qa"]["format"] = "retrieval_set"
    row["necessity"] = "robust_by_construction"
    # f1 is None where the set round-trip does not apply (corpus scope: the
    # distractor-sampling completeness check carries the precision gate instead)
    row["set_roundtrip"] = ({"f1": f1, "ids": rt_ids, "pass": f1 >= 0.8}
                            if f1 is not None else {"skipped": "corpus-scope"})
    row["completeness"] = {"near_miss_checked": checked,
                           "false_negatives": near_miss_hits,
                           "completeness_risk": near_miss_hits > 0}
    f1_ok = f1 is None or f1 >= 0.8
    row["keep"] = (2 <= len(gold) <= 10) and f1_ok and near_miss_hits == 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["program", "corpus", "both"], default="both")
    ap.add_argument("--only-video", default="")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--limit-questions", type=int, default=30, help="corpus-level cap")
    ap.add_argument("--max-targets", type=int, default=3, help="program-level per video")
    ap.add_argument("--dry-run", action="store_true", help="no LLM: targets + candidates only")
    ap.add_argument("--gen-url", default="http://localhost:11434/v1")
    ap.add_argument("--gen-model", default="gemma3:27b-it-qat")
    ap.add_argument("--verify-url", default="http://localhost:11434/v1")
    ap.add_argument("--verify-model", default="llama3.3:latest")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = json.load(open(CATALOG))
    gcfg = {"url": args.gen_url, "model": args.gen_model, "api_key": args.api_key}
    vcfg = {"url": args.verify_url, "model": args.verify_model, "api_key": args.api_key}

    kept = dropped = 0

    # ----- program-level -----
    if args.scope in ("program", "both"):
        vids = sorted(catalog["videos"])
        if args.only_video:
            vids = [v for v in vids if args.only_video in v]
        if args.limit_videos:
            vids = vids[:args.limit_videos]
        for vid in vids:
            idxp = IDX_DIR / f"{vid}.json"
            if not idxp.exists():
                continue
            doc = json.load(open(idxp))
            cv = catalog["videos"][vid]
            rows = []
            for t in program_targets(vid, cv, doc, args.max_targets):
                if t["kind"] == "theme":
                    if args.dry_run:
                        continue  # theme needs the LLM; skip in dry-run
                    theme = propose_theme(t["titles"], gcfg)
                    if not theme:
                        if args.verbose:
                            print(f"  [{vid[:24]}] DROP theme: no proposal")
                        continue
                    q_toks = _toks(theme)
                    t["topic"] = theme
                    t["candidates"] = [s for s in t["pool"]
                                       if seg_matches(doc, s, q_toks) >= 0.5]
                    if not 2 <= len(t["candidates"]) < len(t["pool"]):
                        if args.verbose:
                            print(f"  [{vid[:24]}] DROP theme '{theme[:40]}': "
                                  f"{len(t['candidates'])}/{len(t['pool'])} candidates")
                        continue
                topic = t["topic"]
                gold_segs, checked = [], 0
                if args.dry_run:
                    gold_segs = t["candidates"]
                    q = f"<PLACEHOLDER exploration/{t['kind']} :: {topic}>"
                    f1, rt_ids, near = 1.0, [], 0
                else:
                    # verify EVERY segment of the video (pool is chapter-sized,
                    # <= ~11 items): gold is exhaustive by construction, so no
                    # separate near-miss/completeness sampling is needed at
                    # program level. Lexical candidates only chose the topic.
                    for s in t["pool"]:
                        if verify_segment(doc, s, topic, vcfg, kind=t["kind"]):
                            gold_segs.append(s)
                    checked = len(t["pool"])
                    near = 0
                    if not 2 <= len(gold_segs) <= max(2, len(t["pool"]) // 2):
                        dropped += 1
                        if args.verbose:
                            print(f"  [{vid[:24]}] DROP {t['kind']} '{str(topic)[:36]}': "
                                  f"verified {len(gold_segs)}/{len(t['pool'])} of pool "
                                  f"(need 2..{max(2, len(t['pool']) // 2)})")
                        continue
                    setting = cv.get("provenance", {}).get("source") or "an archival broadcast"
                    what = (f"'{topic}' appears, speaks, or is a main subject"
                            if t["kind"] == "entity" else f"'{topic}' is covered")
                    q = phrase_question("program",
                                        f"Broadcast: {setting}, {cv.get('broadcast_date')}. "
                                        f"The answer set: the segments of THIS broadcast where "
                                        f"{what}. Ask which segments those are.",
                                        gcfg)
                    if not q:
                        dropped += 1
                        continue
                    gold = [item_of(vid, s, doc, topic) for s in gold_segs]
                    pool = [item_of(vid, s, doc, topic) for s in t["pool"]]
                    f1, rt_ids = set_roundtrip(q, gold, pool, vcfg)
                gold = [item_of(vid, s, doc, topic) for s in gold_segs]
                pool = [item_of(vid, s, doc, topic) for s in t["pool"]]
                row = finish_row({
                    "video_id": vid, "cell": "Exploration/L3", "w_role": "what",
                    "element": f"explore:{t['kind']}:{topic}",
                    "evidence": {"scope": "program"},
                    "qa": {"question": q, "rationale": f"{t['kind']}-based retrieval"},
                }, gold, pool, 0 if args.dry_run else near, checked, f1, rt_ids)
                row["completeness"]["exhaustive"] = not args.dry_run
                rows.append(row)
                kept += 1 if row["keep"] else 0
                if args.verbose:
                    print(f"  [{vid[:24]}] {t['kind']:9s} '{str(topic)[:40]}' "
                          f"gold={len(gold)} f1={f1} keep={row['keep']}")
            if rows:
                json.dump({"video_id": vid, "n": len(rows), "rows": rows},
                          open(OUT_DIR / f"{vid}.json", "w"), indent=2)

    # ----- corpus-level -----
    if args.scope in ("corpus", "both"):
        rows = []
        for t in corpus_targets(catalog, args.limit_questions):
            occ = t["occupation"]
            gold, pool, checked, near = [], [], 0, 0
            # candidate segment per hit: prefer the segment the person DOMINATES
            # (diarization), else the one with the most name mentions in ASR
            cands = []
            for h in t["hits"]:
                cv = catalog["videos"][h["video_id"]]
                segs = [s for s in cv["segments"] if s["content"]]
                if not segs:
                    continue
                dom = [s for s in segs
                       if s.get("dominant_speaker")
                       and set(_toks(h["name"])) & set(_toks(s["dominant_speaker"]))]
                if dom:
                    best = max(dom, key=lambda s: s.get("dur_ms") or 0)
                elif not args.dry_run:
                    hdoc = json.load(open(IDX_DIR / f"{h['video_id']}.json"))
                    best = max(segs, key=lambda s: seg_matches(hdoc, s, _toks(h["name"])))
                else:
                    best = max(segs, key=lambda s: s.get("dur_ms") or 0)
                cands.append((h, best))
            def participants_of(v):
                names = [p.get("grounded", {}).get("canonical") or p["name"]
                         for p in catalog["videos"][v]["participants"]
                         if p.get("speaking_ms", 0) > 30000]
                return ", ".join(names[:5])

            if args.dry_run:
                q = f"<PLACEHOLDER exploration/occupation :: {occ}>"
                gold = [dict(item_of(h["video_id"], s, people=participants_of(h["video_id"])),
                             person=h["person"]) for h, s in cands]
                f1, rt_ids = None, []
            else:
                # gold membership is verified by the INDEX: a named participant
                # with 30s+ of diarized speech and a description-corroborated
                # grounded occupation. (Roundtable guests are introduced once and
                # never re-named in ASR, so LLM text checks systematically miss
                # them.) The LLM check is kept as non-gating sanity metadata for
                # the known upstream risk: ASR-mangled speaker-name resolution.
                for h, s in cands:
                    doc = json.load(open(IDX_DIR / f"{h['video_id']}.json"))
                    # identity gate: the grounding may have picked the wrong
                    # namesake; check the description against the speech context
                    if not verify_identity(doc, s, h["person"], h["description"], vcfg):
                        if args.verbose:
                            print(f"  [corpus] identity-reject: {h['person']} "
                                  f"({h['description'][:50]}) in {h['video_id'][:30]}")
                        continue
                    dom = s.get("dominant_speaker") or ""
                    dom_match = bool(set(_toks(h["name"])) & set(_toks(dom)))
                    sanity = "diarization" if dom_match else (
                        "llm-pass" if verify_person_segment(
                            doc, s, h["person"], occ, vcfg) else "llm-unconfirmed")
                    gold.append(dict(item_of(h["video_id"], s, doc, h["person"],
                                             people=participants_of(h["video_id"])),
                                     person=h["person"], verified=sanity))
                if not 2 <= len(gold) <= 10:
                    dropped += 1
                    if args.verbose:
                        print(f"  [corpus] DROP {occ:24s} gold={len(gold)} out of bounds")
                    continue
                q = phrase_question("corpus",
                                    f"Collection: archival US broadcasts. The answer set: "
                                    f"broadcasts in the collection where a {occ} speaks "
                                    f"substantively. Ask EXACTLY that - which broadcasts "
                                    f"feature a {occ} speaking - without adding qualifiers "
                                    f"or narrowing the criterion.", gcfg)
                if not q:
                    dropped += 1
                    continue
                # completeness: sample catalog-covered videos with no participant
                # of this occupation; an occupation-level content check firing on
                # one means the catalog has a grounding hole for this question
                other = [v for v in t["distractor_vids"]
                         if v not in {g["video_id"] for g in gold}]
                distractors = []
                for ov in RNG.sample(other, min(8, len(other))):
                    segs = [s for s in catalog["videos"][ov]["segments"] if s["content"]]
                    if segs:
                        distractors.append(item_of(ov, max(segs, key=lambda s: s.get("dur_ms") or 0),
                                                   people=participants_of(ov)))
                checked = len(distractors)
                for dit in distractors:
                    ddoc = json.load(open(IDX_DIR / f"{dit['video_id']}.json"))
                    dseg = next(s for s in catalog["videos"][dit["video_id"]]["segments"]
                                if s["seg_id"] == dit["seg_id"])
                    if verify_occ_segment(ddoc, dseg, occ, dit.get("people"), vcfg):
                        near += 1
                pool = gold + distractors
                f1, rt_ids = None, []
            if not pool:
                pool = gold
            row = finish_row({
                "video_id": None, "cell": "Exploration/L4", "w_role": "who",
                "element": f"explore:occupation:{occ}",
                "evidence": {"scope": "corpus"},
                "qa": {"question": q, "rationale": "occupation-based corpus retrieval"},
            }, gold, pool, near, checked, f1, rt_ids)
            # corpus answers are defined relative to GROUNDED participants;
            # grounding abstains on obscure figures (~51%), so the answer set is
            # complete w.r.t. the catalog, not the raw footage. Documented scope.
            row["completeness"]["scope"] = "grounded-participants-only"
            rows.append(row)
            kept += 1 if row["keep"] else 0
            if args.verbose:
                print(f"  [corpus] {occ:28s} gold={len(gold)} "
                      f"holes={near}/{checked} keep={row['keep']}")
        # near-duplicate dedupe: co-occurring occupations of the same people
        # ("political adviser" / "political analyst") yield identical answer
        # sets; keep only the first question per distinct video set
        seen_sets = set()
        for r in rows:
            key = frozenset(g["video_id"] for g in r["qa"]["answer_set"])
            if r["keep"] and key in seen_sets:
                r["keep"] = False
                r["dedup"] = "duplicate_answer_set"
                kept -= 1
            elif r["keep"]:
                seen_sets.add(key)
        if rows:
            json.dump({"video_id": "_corpus", "n": len(rows), "rows": rows},
                      open(OUT_DIR / "_corpus.json", "w"), indent=2)

    print(f"\nkept: {kept} | dropped(gates): {dropped} | output -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
