#!/usr/bin/env python3
"""Visually-grounded question generation (v6, the cross-modal typed minority).

Two target types, both salience-gated:
  visual_text   on-screen text (text_focus captions: chyrons, slates, signs,
                title cards) inside a salient segment. The richest source of
                visually-necessary facts in archival broadcast.
  visual_scene  a distinctive visual fact CORROBORATED by >=2 adjacent
                general-scene captions (single-frame captions hallucinate ~28%,
                so uncorroborated captions never become gold).

The modality gate is EMPIRICAL, two-sided (the honest version of v5.x's gamed
`modality_fit`): a verifier (different family from the generator) must
  (a) re-derive the answer from the VISUAL evidence   -> visual_pass
  (b) FAIL to derive it from the ASR of the same span -> not speech-answerable
Rows keep only if (a) and not (b)-fails, i.e. the question genuinely requires
watching. Speech-answerable rows are recorded and dropped (ordinary need-down
questions already cover speech).

Self-containment is built into generation (setting + named subjects, no
deixis); visual rows deliberately SKIP the densify pass.

Usage:
  python scripts/generate_qa_visual.py --only-video cpb-aacip-507-3x83j39n00 --dry-run --verbose
  python scripts/generate_qa_visual.py --only-video <vid> \
      --gen-url http://localhost:11434/v1 --gen-model gemma3:27b-it-qat \
      --verify-url http://localhost:11434/v1 --verify-model llama3.3:latest
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
import requests

from utils.answer_match import decide_match
from utils.ctx_retrieval import _toks

IDX_DIR = Path("data/video_indexes")
SAL_DIR = Path("data/salience_maps")
PROV = Path("data/v5_1_video_provenance.json")
OUT_DIR = Path("data/qa_visual")

TF_LAYER = "caption_qwen3vl-8b_text_focus"
GS_LAYER = "caption_qwen3vl-8b_general_scene"

NON_CONTENT_TITLES = {"introduction", "closing", "credits", "intro", "opening",
                      "end credits", "bumper", "promo"}

# reasoning-leak prefixes some VLM captions carry
LEAK_LINE = re.compile(r"^(drafting|the user wants|okay[, ]|let me|i need to|here'?s|"
                       r"\d+\.\s+\*\*|analyz|identify)", re.I)
TF_JUNK = re.compile(r"no (visible )?text|does not contain|unable to (read|determine)|"
                     r"^\d{1,2}:\d{2}(:\d{2})?$|^[a-z]{1,3}$", re.I)

# incidental-appearance trivia (plan C.1: clothing colors, cymbal brands):
# deterministic backstop behind the prompt instruction
TRIVIA = re.compile(r"\b(colou?r|wearing|worn|shirt|tie|suit|jacket|dress|blouse|hat\b|"
                    r"hairstyle|hair\b|glasses|necklace|earring|outfit|clothes|clothing|"
                    r"handwriting|cursive|font\b|typeface|style of (writing|text)|"
                    r"repeated .{0,12}times)\b", re.I)

# self-containment check for generated visual questions
V_DEIXIS = re.compile(r"\b(this (broadcast|segment|program|video|episode)|the video\b|"
                      r"the screen\b|the segment\b|the broadcast\b|shown during the segment|"
                      r"the man being interviewed|the graphic shown|the crowd\b)", re.I)


def _items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def clean_caption(text):
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    kept = [l for l in lines if not LEAK_LINE.match(l)]
    return " ".join(kept)[:500]


def asr_text(doc, start_ms, end_ms, max_chars=2000):
    parts = []
    seen = set()
    for k in doc.get("layers", {}):
        if not k.lower().startswith("asr"):
            continue
        for it in _items(doc["layers"][k]):
            if not isinstance(it, dict) or it.get("start_ms") is None:
                continue
            s, e = it["start_ms"], it.get("end_ms") or it["start_ms"]
            if s < end_ms and e > start_ms:
                t = (it.get("text") or "").strip()
                if t and t[:80].lower() not in seen:
                    seen.add(t[:80].lower())
                    parts.append((s, t))
    parts.sort()
    return " ".join(t for _, t in parts)[:max_chars]


def chat(url, model, api_key, sysm, user, temperature=0.2, max_tokens=300, timeout=120):
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


# ---------- target construction ----------
NO_TEXT_LABELS = {"neg", "bars"}    # SWT says no readable text in these frames


def tf_targets(doc, segs):
    """On-screen text inside salient segments: chyrons, slates, signs, cards.
    Only frames SWT labeled as text-bearing; deduped by normalized text."""
    out = []
    seen = set()
    for it in _items(doc.get("layers", {}).get(TF_LAYER)):
        if not isinstance(it, dict) or it.get("start_ms") is None:
            continue
        label = (it.get("scene_label") or "").strip()
        if label.lower() in NO_TEXT_LABELS:
            continue
        txt = (it.get("text") or "").strip()
        if not txt or txt.lower() in ("none",) or TF_JUNK.search(txt):
            continue
        txt = clean_caption(txt)
        if len(txt) < 12:               # single words carry no question
            continue
        key = re.sub(r"[^a-z0-9]", "", txt.lower())[:70]
        if key in seen:
            continue
        seen.add(key)
        seg = next((s for s in segs
                    if s["start_ms"] <= it["start_ms"] < (s["end_ms"] or 0)), None)
        if seg is None:
            continue
        out.append({"kind": "visual_text", "seg": seg, "text": txt,
                    "label": label or "text",
                    "start_ms": it["start_ms"], "end_ms": it.get("end_ms") or it["start_ms"]})
    return out


def gs_targets(doc, segs, min_agree=2):
    """Corroborated visual facts: a rare content token shared by >=min_agree
    adjacent general-scene captions inside one salient segment."""
    caps = []
    for it in _items(doc.get("layers", {}).get(GS_LAYER)):
        if not isinstance(it, dict) or it.get("start_ms") is None:
            continue
        txt = clean_caption(it.get("text") or "")
        if len(txt) > 30:
            caps.append({"start_ms": it["start_ms"], "end_ms": it.get("end_ms") or it["start_ms"],
                         "text": txt})
    caps.sort(key=lambda c: c["start_ms"])
    # generic caption vocabulary that corroborates nothing
    generic = set("man woman suit tie shirt camera speaking background image video frame "
                  "screen text displays shows appears sitting standing wearing jacket "
                  "person people blue red white black brown gray middleaged aged hair".split())
    out = []
    for seg in segs:
        window = [c for c in caps if seg["start_ms"] <= c["start_ms"] < (seg["end_ms"] or 0)]
        if len(window) < min_agree:
            continue
        vocab = Counter()
        for c in window:
            vocab.update(set(w for w in _toks(c["text"]) if w not in generic))
        # tokens seen in >=min_agree distinct captions but not everywhere (distinctive)
        agreed = [w for w, n in vocab.items() if min_agree <= n <= max(min_agree, len(window) // 2)]
        if not agreed:
            continue
        # anchor on the caption pair sharing the most agreed tokens
        best, best_score = None, 0
        for i in range(len(window) - 1):
            sc = len(set(_toks(window[i]["text"])) & set(_toks(window[i + 1]["text"])) & set(agreed))
            if sc > best_score:
                best, best_score = (window[i], window[i + 1]), sc
        if best is None:
            continue
        out.append({"kind": "visual_scene", "seg": seg,
                    "text": best[0]["text"] + " || " + best[1]["text"],
                    "label": "scene", "start_ms": best[0]["start_ms"],
                    "end_ms": best[1]["end_ms"]})
    return out


# ---------- generation + gates ----------
def gen_question(t, setting, gcfg, gripe=""):
    sysm = ("You write ONE free-text question about an archival broadcast for a research "
            "benchmark. The question must be answerable ONLY BY WATCHING the video (the "
            "answer is something SEEN on screen: on-screen text, a graphic, a setting, a "
            "visual event), not from the audio. It must be SELF-CONTAINED: name the program "
            "and subject, no 'this segment/broadcast', no relative time. The ANSWER must be "
            "concrete and short (the exact on-screen text, a name, an object). "
            "Ask about visually SIGNIFICANT content an archivist would catalog: on-screen "
            "text and titles, identified people and places shown, depicted events, graphics "
            "and their message. NEVER ask about incidental appearance (clothing, colors, "
            "hairstyles, furniture, what someone is wearing). If the evidence offers only "
            'incidental detail, return {"skip": true}. Do not begin with "Based on". '
            'Return JSON: {"question", "answer", "rationale"}.')
    what = ("on-screen text captured from the video" if t["kind"] == "visual_text"
            else "two consecutive frame descriptions from the video (facts appearing in "
                 "BOTH are reliable; facts in only one may be hallucinated)")
    user = (f"Program: {setting}.\nSegment topic: {t['seg'].get('title')}\n"
            f"Scene label: {t['label']}\n"
            f"Visual evidence ({what}):\n{t['text']}\n\n"
            f"Write the question and its answer grounded ONLY in this visual evidence."
            + (f"\nIMPORTANT: {gripe}" if gripe else ""))
    return json_out(chat(gcfg["url"], gcfg["model"], gcfg["api_key"], sysm, user,
                         temperature=0.7, max_tokens=300))


def reanswer(question, ctx, kind, vcfg):
    user = (f"Using ONLY the {kind} below, answer the question in a few words. "
            "If it does not contain the answer, reply UNANSWERABLE. /no_think\n\n"
            f"{kind}:\n{ctx}\n\nQuestion: {question}\nAnswer:")
    return chat(vcfg["url"], vcfg["model"], vcfg["api_key"], "", user,
                temperature=0, max_tokens=128)


def modality_gate(q, gold, t, doc, vcfg):
    """(a) visual round-trip must pass; (b) ASR re-answer must fail."""
    judge = {"url": vcfg["url"], "model": vcfg["model"], "api_key": vcfg["api_key"]}
    vis = reanswer(q, t["text"], "visual/frame evidence", vcfg)
    visual_pass = (not vis.startswith("__ERR__") and "unanswerable" not in vis.lower()
                   and decide_match(q, vis, gold, judge))
    # ASR context widened around the visual moment (+- 60s) plus the segment topic
    a0 = max(0, t["start_ms"] - 60000)
    a1 = (t["end_ms"] or t["start_ms"]) + 60000
    asr = reanswer(q, asr_text(doc, a0, a1), "transcript", vcfg)
    speech_answerable = (not asr.startswith("__ERR__") and "unanswerable" not in asr.lower()
                         and decide_match(q, asr, gold, judge))
    return {"visual_pass": bool(visual_pass), "visual_answer": vis[:120],
            "speech_answerable": bool(speech_answerable), "speech_answer": asr[:120],
            "keep": bool(visual_pass and not speech_answerable)}


def setting_for(prov, vid):
    e = prov.get(vid, {}) if isinstance(prov, dict) else {}
    src = e.get("source") or e.get("collection") or "an archival US broadcast"
    era = e.get("era_estimate") or ""
    return f"{src}{f' ({era})' if era else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-video", default="")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--max-per-video", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--gen-url", default="http://localhost:11434/v1")
    ap.add_argument("--gen-model", default="gemma3:27b-it-qat")
    ap.add_argument("--verify-url", default="http://localhost:11434/v1")
    ap.add_argument("--verify-model", default="llama3.3:latest")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gcfg = {"url": args.gen_url, "model": args.gen_model, "api_key": args.api_key}
    vcfg = {"url": args.verify_url, "model": args.verify_model, "api_key": args.api_key}
    prov = json.load(open(PROV)) if PROV.exists() else {}

    sal_files = sorted(SAL_DIR.glob("*.json"))
    if args.only_video:
        sal_files = [p for p in sal_files if args.only_video in p.stem]
    if args.limit_videos:
        sal_files = sal_files[:args.limit_videos]

    tot_targets = tot_kept = tot_speech = tot_visfail = tot_trivia = tot_deixis = 0
    for sp in sal_files:
        vid = sp.stem
        outp = OUT_DIR / f"{vid}.json"
        if args.skip_done and outp.exists():
            continue
        idxp = IDX_DIR / f"{vid}.json"
        if not idxp.exists():
            continue
        doc = json.load(open(idxp))
        sal = json.load(open(sp))
        segs = [s for s in sal.get("salient_segments", [])
                if (s.get("title") or "").strip().lower() not in NON_CONTENT_TITLES
                and (s.get("dur_ms") or 0) > 45000]
        if not segs:
            continue
        targets = tf_targets(doc, segs) + gs_targets(doc, segs)
        # prefer on-screen text (least hallucination-prone), cap per video
        targets = (sorted([t for t in targets if t["kind"] == "visual_text"],
                          key=lambda t: -len(t["text"]))
                   + [t for t in targets if t["kind"] == "visual_scene"])[:args.max_per_video]
        tot_targets += len(targets)
        setting = setting_for(prov, vid)

        rows = []
        seen_answers = set()
        for t in targets:
            if args.dry_run:
                rows.append({"video_id": vid, "cell": "Visual/L2", "w_role": "what",
                             "element": f"{t['kind']}:{t['label']}",
                             "evidence": {"start_ms": t["start_ms"], "end_ms": t["end_ms"],
                                          "modality": "visual",
                                          "visual_text": t["text"][:300]},
                             "qa": {"question": f"<PLACEHOLDER {t['kind']} :: {t['text'][:50]}>",
                                    "answer": "<answer>"}})
                continue
            out = gen_question(t, setting, gcfg)
            q, a = out.get("question"), out.get("answer")
            if q and V_DEIXIS.search(q):
                m = V_DEIXIS.search(q)
                out2 = gen_question(t, setting, gcfg,
                                    gripe=f"your previous question used the vague reference "
                                          f"'{m.group(0)}' - identify the program by name and "
                                          f"the segment by its topic instead")
                if out2.get("question") and not V_DEIXIS.search(out2["question"]):
                    out = out2
                    q, a = out.get("question"), out.get("answer")
            if not q or not a:
                continue
            if V_DEIXIS.search(q):
                tot_deixis += 1
                if args.verbose:
                    print(f"  [{vid[-14:]}] DEIXIS-drop: {str(q)[:70]!r}")
                continue
            if TRIVIA.search(f"{q} {a}"):
                tot_trivia += 1
                if args.verbose:
                    print(f"  [{vid[-14:]}] TRIVIA-drop: {str(q)[:70]!r}")
                continue
            if len(re.sub(r"[^A-Za-z0-9]", "", str(a))) < 2:   # degenerate ("D")
                continue
            norm_a = re.sub(r"[^a-z0-9]", "", str(a).lower())
            if norm_a in seen_answers:      # caption variants of the same card
                continue
            seen_answers.add(norm_a)
            gate = modality_gate(q, a, t, doc, vcfg)
            tot_kept += 1 if gate["keep"] else 0
            tot_speech += 1 if gate["speech_answerable"] else 0
            tot_visfail += 0 if gate["visual_pass"] else 1
            rows.append({"video_id": vid, "cell": "Visual/L2", "w_role": "what",
                         "element": f"{t['kind']}:{t['label']}",
                         "evidence": {"start_ms": t["start_ms"], "end_ms": t["end_ms"],
                                      "modality": "visual", "visual_text": t["text"][:400]},
                         "qa": {"question": q, "answer": a,
                                "rationale": out.get("rationale", "")},
                         "visual_gate": gate})
            if args.verbose:
                print(f"  [{vid[-14:]}] {t['kind']:12s} keep={gate['keep']} "
                      f"(vis={gate['visual_pass']} speech={gate['speech_answerable']}) "
                      f"{str(q)[:60]!r}")
        if rows:
            json.dump({"video_id": vid, "n": len(rows), "rows": rows},
                      open(outp, "w"), indent=2)

    print(f"\ntargets: {tot_targets} | kept (visual-necessary): {tot_kept} | "
          f"rejected speech-answerable: {tot_speech} | visual round-trip failed: {tot_visfail} "
          f"| trivia-dropped: {tot_trivia} | deixis-dropped: {tot_deixis}")
    print(f"output -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
