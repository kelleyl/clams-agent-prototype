#!/usr/bin/env python3
"""Dense-paraphrasing self-containment pass (v6).

Rewrites each generated question so it is SELF-CONTAINED: a reader who has not
seen the video understands exactly what is being asked. Resolves vague
references ("this segment", "the report", "the speaker", "the demonstrations
referenced") into explicit names, program, event, and people, drawn from the
provenance setting + segment topic + transcript. Adds SETTING, not the ANSWER
(the answer must still require the video).

This is dense paraphrasing (Tu et al.) applied to the question rather than the
evidence. Runs on aristotle; use the SAME model as the generator to avoid
model-swap overhead.

Example:
  python scripts/densify_questions.py --video cpb-aacip-507-3x83j39n00 \
      --dp-url http://localhost:11434/v1 --dp-model gemma3:27b-it-qat
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
from utils.ctx_retrieval import retrieve_context

QA_DIR = Path("data/qa_needdown")
IDX_DIR = Path("data/video_indexes")
SAL_DIR = Path("data/salience_maps")
PROV = Path("data/v5_1_video_provenance.json")


def subject_anchor(sal):
    """A video-level who/what anchor so the rewrite can resolve generic
    references ('the president', 'his speech') to named people and events."""
    people = [p["name"] for p in sal.get("main_participants", [])
              if not p["name"].upper().startswith("SPEAKER_")][:6]
    topics = [s["title"] for s in sal.get("salient_segments", [])
              if s.get("title")][:5]
    parts = []
    if people:
        parts.append("Key people: " + ", ".join(people))
    if topics:
        parts.append("Topics/segments: " + "; ".join(topics))
    return ". ".join(parts)
DEIXIS = re.compile(r"\b(this segment|the segment|the report|the broadcast|the clip|the video|"
                    r"the speaker|referenced|the passage|the footage|the discussion|"
                    r"th(?:is|e) (?:news )?summary|this broadcast|this program|this episode|"
                    r"the program|the interview\b|the interviewer|the anchor\b|the guest\b)",
                    re.I)
# trailing program-attribution tic ("..., according to the discussion on PBS
# NewsHour?"): adds no identifying information the event description should not
# already carry, pads every question with the same formula, and hands the blind
# panel a program/era hint
ATTRIB = re.compile(r"\b(according to|as (?:mentioned|discussed|stated|described|reported)\b)"
                    r"[^,.?]*\b(broadcast|program|episode|newshour|news hour|segment|report|"
                    r"reports|discussion|transcript|video|coverage)\b", re.I)
# unresolved relative time ("news reports from today"): meaningless without the
# broadcast date; densify must resolve these to the date or drop them
RELTIME = re.compile(r"\b(today|yesterday|tonight|this (?:week|morning|evening)|"
                     r"earlier today)\b", re.I)


def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def asr_text(doc, start_ms, end_ms, max_chars=1200):
    parts = []
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        for it in items(doc["layers"][k]):
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


def setting_for(prov, vid):
    e = prov.get(vid, {}) if isinstance(prov, dict) else {}
    src = e.get("source") or e.get("collection") or "an archival public-broadcasting program"
    genre = (e.get("primary_genre") or "").replace("_", " ")
    era = e.get("era_estimate") or ""
    parts = [src]
    if genre:
        parts.append(f"a {genre}")
    if era:
        parts.append(f"({era})")
    return ", ".join(parts)


def topic_of(t):
    el = t.get("element", "")
    return el.split(":", 1)[1] if ":" in el else el


def densify(setting, anchor, topic, ctx, question, answer, url, model, api_key,
            timeout=90, bdate=""):
    sysm = ("You rewrite a benchmark question to be self-contained, without changing what it "
            "asks and without revealing the answer. You make the MINIMAL edit: if the original "
            "is already self-contained, return it UNCHANGED; otherwise change only what "
            "self-containment requires and keep the original's natural wording everywhere else.")
    user = (f"Setting: {setting}.\n"
            f"Broadcast date: {bdate or 'unknown'}.\n"
            f"Who and what this broadcast is about (BACKGROUND ONLY - the broadcast covers "
            f"several unrelated stories): {anchor or '(unknown)'}\n"
            f"Segment label (UNRELIABLE - news-summary segments mix stories and labels can be "
            f"misaligned): {topic}.\n"
            f"Transcript excerpt (the fact being asked about is stated in here):\n"
            f"{ctx or '(none)'}\n\n"
            f"Original question: {question}\n"
            f"The answer (do NOT include or hint at it): {answer}\n\n"
            "CRITICAL: identify the event/story from the TRANSCRIPT SENTENCES surrounding the "
            "asked fact. If the segment label or broadcast topics name a different story than "
            "those sentences, IGNORE them - attaching the wrong story makes the question "
            "factually misleading. If you cannot tell which story the fact belongs to, add no "
            "story framing at all (name only the people/roles). "
            "Never use relative time words (today, yesterday, tonight, this week) - resolve "
            "them to the broadcast date or an absolute description of the event.\n"
            "A question is self-contained ONLY if ALL of these hold: (a) it contains no "
            "deictic reference (no 'this news summary', 'this segment', 'the broadcast', "
            "'the speaker', 'the program', 'the interview'); (b) every person/organization is "
            "named, not described generically ('the president', 'the senator'); (c) no "
            "unresolved relative time ('today', 'yesterday'). "
            "If the original already satisfies ALL three, return it UNCHANGED. Otherwise "
            "rewrite it MINIMALLY: fix ONLY the failing parts and keep the original's natural "
            "wording everywhere else. Name the SUBJECT explicitly: replace generic references "
            "like 'the president', 'his speech', 'this segment', 'the speaker' with the specific "
            "named person and the specific named event/occasion, using the who/what anchor and "
            "transcript above (e.g. 'the president' -> 'George W. Bush', 'his speech' -> 'his 2001 "
            "inaugural address'). Add identifying context, but do NOT reveal or hint at the answer, "
            "and do NOT change what is being asked. Do NOT begin the question with 'Based on' or "
            "'According to the transcript'. Do NOT append attribution clauses such as 'according "
            "to the discussion on PBS NewsHour', 'as mentioned in the broadcast', or 'as discussed "
            "in the program' - identify the event itself (people, occasion, place, date), not the "
            "program that covered it. Keep it natural and a single sentence. "
            "Return JSON: {\"question\": \"...\"}.")
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0.3, "max_tokens": 200,
                                "messages": [{"role": "system", "content": sysm},
                                             {"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        if "</think>" in txt:
            txt = txt.split("</think>")[-1]
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0)).get("question")
    except Exception as ex:
        return f"__ERR__:{ex}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--qa-dir", default=str(QA_DIR),
                    help="sidecar dir (default data/qa_needdown)")
    ap.add_argument("--redo", action="store_true",
                    help="re-densify from question_raw even if already densified; "
                         "clears the round-trip verdict of rewritten rows")
    ap.add_argument("--dp-url", default="http://localhost:11434/v1")
    ap.add_argument("--dp-model", default="gemma3:27b-it-qat")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    qa_dir = Path(args.qa_dir)
    data = json.load(open(qa_dir / f"{args.video}.json"))
    doc = json.load(open(IDX_DIR / f"{args.video}.json"))
    prov = json.load(open(PROV)) if PROV.exists() else {}
    sal_path = SAL_DIR / f"{args.video}.json"
    sal = json.load(open(sal_path)) if sal_path.exists() else {}
    setting = setting_for(prov, args.video)
    anchor = subject_anchor(sal)
    print(f"setting: {setting}")
    print(f"anchor: {anchor[:160]}")

    bdate = doc.get("broadcast_date") or ""
    n = vague_before = vague_after = err = tic = 0
    for r in data.get("rows", []):
        qa = r.get("qa", {})
        q, a = qa.get("question"), qa.get("answer")
        if not q or qa.get("error"):
            continue
        if (r.get("element") or "").startswith("two_hop:"):
            # never rewrite two-hop questions: densify re-inserts the entity
            # name the necessity design deliberately hides
            continue
        if qa.get("densified"):
            if not args.redo:   # resume: already rewritten (not idempotent)
                continue
            q = qa.get("question_raw") or q     # restart from the raw question
        if DEIXIS.search(q):
            vague_before += 1
        ev = r.get("evidence", {})
        # answer-bearing passage FIRST (retrieved by question+answer tokens), the
        # window head second: the model must see the sentences that state the fact,
        # or it will frame the question with whichever story the window opens on
        focus = retrieve_context(doc, f"{q} {a}", k=5)
        head = asr_text(doc, ev["start_ms"], ev["end_ms"]) if ev.get("start_ms") is not None else ""
        ctx = (focus + "\n" + head).strip()[:1800]
        new = densify(setting, anchor, topic_of(r), ctx, q, a, args.dp_url, args.dp_model,
                      args.api_key, bdate=bdate)
        if new and not str(new).startswith("__ERR__") and (
                ATTRIB.search(new) or RELTIME.search(new) or DEIXIS.search(new)):
            # one retry with the offending pattern called out explicitly
            m = DEIXIS.search(new)
            gripe = ("appended a program-attribution clause; identify the event itself instead"
                     if ATTRIB.search(new) else
                     f"kept the vague reference '{m.group(0)}' - replace it with the specific "
                     "program name, date, or named people" if m else
                     "used a relative time word; resolve it to the broadcast date")
            retry = densify(setting, anchor, topic_of(r),
                            ctx + f"\n\nIMPORTANT: your previous rewrite {gripe}.",
                            q, a, args.dp_url, args.dp_model, args.api_key, bdate=bdate)
            if (retry and not str(retry).startswith("__ERR__")
                    and not ATTRIB.search(retry) and not RELTIME.search(retry)
                    and not DEIXIS.search(retry)):
                new = retry
        if not new or str(new).startswith("__ERR__"):
            err += 1
            continue
        qa["question_raw"] = q
        qa["question"] = new
        qa["densified"] = True
        if args.redo:
            r.pop("roundtrip", None)    # verdict was for the old phrasing
        n += 1
        if DEIXIS.search(new):
            vague_after += 1
            qa["self_contained_flag"] = "residual_deixis"
        if ATTRIB.search(new):
            tic += 1
            qa["self_contained_flag"] = "attribution_tic"
        elif RELTIME.search(new):
            tic += 1
            qa["self_contained_flag"] = "relative_time"
        if args.verbose and n <= 8:
            print(f"\n  BEFORE: {q[:95]}")
            print(f"  AFTER : {new[:130]}")

    json.dump(data, open(qa_dir / f"{args.video}.json", "w"), indent=2)
    print(f"\ndensified: {n} | vague(before): {vague_before} -> vague(after): {vague_after} "
          f"| attribution-tic(after retry): {tic} | errors: {err}")


if __name__ == "__main__":
    main()
