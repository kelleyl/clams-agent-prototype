#!/usr/bin/env python3
"""Need-down QA generator (v6 design).

Generates questions FROM salient elements (need-down) instead of extracting
whatever sits in a window (evidence-up). Implements the three-axis spec
(information need x cognitive level x 5W fact-role) with a deliberate skew
toward why/how and the interpretive/subject cells, plus a two-hop
grounding-document type gated by era-consistency, uniqueness, and necessity.

Inputs (already on disk):
  data/salience_maps/<vid>.json         (scripts/build_salience_map.py)
  data/entity_grounding_llm/<vid>.json  (scripts/ground_entities_llm.py)
  data/video_indexes/<vid>.json         (for ASR evidence context)

LLM generation runs on the aristotle vLLM (OpenAI-compatible). --dry-run does
NO LLM call: it builds targets, assembles prompts, and runs the gates, emitting
placeholder questions so the whole pipeline is testable without a model.

Examples:
  # placeholder test, no LLM
  python scripts/generate_qa_needdown.py --only-video cpb-aacip-507-154dn40c26 --dry-run --verbose
  # real generation on aristotle
  python scripts/generate_qa_needdown.py --vllm-url http://localhost:8200/v1 \
      --model Qwen/Qwen3-VL-8B-Instruct
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

IDX_DIR = Path("data/video_indexes")
SAL_DIR = Path("data/salience_maps")
GRD_DIR = Path("data/entity_grounding_llm")
DESC_DIR = Path("data/video_descriptions")
OUT_DIR = Path("data/qa_needdown")

# fact fields that are stable over time (safe for two-hop over old broadcasts)
TIME_INVARIANT = ["Occupation", "Field of work", "Country of citizenship",
                  "Educated at", "Industry"]
# fields that change over time -> risky/anachronistic for an old broadcast
TIME_VARYING = ["Employer", "Positions held", "Positions", "Position held", "Member of"]

NON_CONTENT_TITLES = {"introduction", "closing", "credits", "intro", "opening",
                      "end credits", "bumper", "promo"}


# ---------- loading / evidence ----------
def items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def asr_layers(doc):
    return [k for k in doc.get("layers", {}) if k.lower().startswith("asr")]


def asr_text(doc, start_ms, end_ms, max_chars=1500):
    parts = []
    for lk in asr_layers(doc):
        for it in items(doc["layers"][lk]):
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


def desc_window(desc, start_ms, end_ms, max_chars=4000):
    """Prose of the DP video-description beats overlapping [start,end] (arm B).
    Falls back to the scaffold if a beat has no prose."""
    out = []
    for b in desc.get("beats", []):
        s, e = b.get("start_ms"), b.get("end_ms") or b.get("start_ms")
        if s is None:
            continue
        if s < end_ms and e > start_ms:
            out.append((b.get("prose") or b.get("scaffold") or "").strip())
    return "\n".join(t for t in out if t)[:max_chars]


def evidence_text(doc, desc, start_ms, end_ms):
    """Evidence for a target window: raw ASR (slice mode) or DP prose (desc mode)."""
    if desc is not None:
        return desc_window(desc, start_ms, end_ms)
    return asr_text(doc, start_ms, end_ms)


def broadcast_year(doc):
    bd = doc.get("broadcast_date") or ""
    m = re.search(r"(19|20)\d{2}", bd) or re.search(r"(19|20)\d{2}", doc.get("video_id", ""))
    return int(m.group(0)) if m else None


# ---------- two-hop gates ----------
def parse_wd_fields(wd):
    fields = {}
    for part in re.split(r"\.\s+", wd or ""):
        m = re.match(r"\s*([A-Za-z ]+):\s*(.+)", part)
        if m:
            fields[m.group(1).strip()] = [v.strip() for v in m.group(2).split(",") if v.strip()]
    return fields


def entity_existence_year(wd):
    # "(1926-2018)", "born 1982", "Inception: 1972"
    m = re.search(r"\((?:born\s*)?((?:19|20)\d{2})", wd or "")
    if m:
        return int(m.group(1))
    m = re.search(r"(?:born|Inception:)\s*((?:19|20)\d{2})", wd or "")
    return int(m.group(1)) if m else None


def era_ok(wd, year):
    """Reject if the entity did not exist yet when the broadcast aired."""
    if year is None:
        return True
    ey = entity_existence_year(wd)
    return ey is None or ey <= year


def choose_two_hop_fact(wd):
    """Pick a time-invariant fact, interesting first. Country of citizenship is
    excluded (too generic/leaky). For multi-valued occupation/field we take the
    primary value as the answer and return the rest as `also_true` so distractor
    generation can avoid them (preserving a single defensible answer).
    Returns (field, value, also_true) or None.
    """
    fields = parse_wd_fields(wd)
    for f in ["Occupation", "Field of work", "Educated at", "Industry"]:
        vals = fields.get(f)
        if vals:
            return f, vals[0], vals[1:]
    return None


# ---------- target construction (need-down, why/how skew) ----------
def build_targets(sal, grounding, doc, max_per_cell):
    year = broadcast_year(doc)
    parts = sal.get("main_participants", [])
    named = [p for p in parts if p.get("kind") in ("speaker", "both")
             and not p["name"].upper().startswith("SPEAKER_")
             and p.get("speaking_ms", 0) > 20000]
    segs = [s for s in sal.get("salient_segments", [])
            if s.get("title") and s["title"].strip().lower() not in NON_CONTENT_TITLES
            and s.get("dur_ms", 0) > 45000]

    targets = []

    def add(cell, w, element, evidence, extra=None):
        targets.append({"cell": cell, "w_role": w, "element": element,
                        "evidence": evidence, "extra": extra or {}})

    # FACTUAL / retrieval (primary volume) -- a specific stated fact per salient segment
    for s in segs:
        add("Factual/L2", "what", f"fact:{s['title']}",
            {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "speaker": s.get("dominant_speaker")})

    # WHY (opt-in: top segments only; round-trip gate drops ones lacking a stated reason)
    for s in segs[:3]:
        add("Interpretive/L3", "why", f"segment:{s['title']}",
            {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "speaker": s.get("dominant_speaker")})

    # interpretive comparison (L4) -- if >=2 named participants
    if len(named) >= 2 and segs:
        a, b = named[0]["name"], named[1]["name"]
        add("Interpretive/L4", "how", f"compare:{a}|{b}",
            {"topic": segs[0]["title"], "a": a, "b": b,
             "start_ms": segs[0]["start_ms"], "end_ms": segs[0]["end_ms"]})

    # subject -- what the top segments are about (fewer)
    for s in segs[:max(2, max_per_cell // 2)]:
        add("Subject/L2", "what", f"segment:{s['title']}",
            {"start_ms": s["start_ms"], "end_ms": s["end_ms"]})

    # two-hop grounding-document (external knowledge) -- SMALL slice only.
    # Structurally leakage-prone: if the person is identifiable, world knowledge
    # supplies the external fact, so the necessity gate filters most of these.
    # Cap to a few rather than one per grounded entity.
    two_hop_added = 0
    for e in [x for x in grounding.get("entities", []) if x.get("grounded")]:
        if two_hop_added >= 3:
            break
        wd = e.get("wikidata") or ""
        if not era_ok(wd, year):
            continue
        fact = choose_two_hop_fact(wd)
        if not fact:
            continue
        field, value, also_true = fact
        add("Interpretive/L3", "what", f"two_hop:{e['text']}",
            {"entity": e["text"], "field": field, "answer": value, "also_true": also_true,
             "occupation": (parse_wd_fields(wd).get("Occupation") or [""])[0]})
        two_hop_added += 1

    # (factual handled above as the primary cell)

    # cataloging (few) -- participants + production
    if named:
        add("Cataloging/L2", "who", "participants",
            {"names": [p["name"] for p in named[:6]]})
    labels = sal.get("scene_labels", {})
    if labels.get("Slate") or labels.get("Credits"):
        add("Cataloging/L1", "what", "production", {"labels": labels})

    return targets


# ---------- prompt assembly ----------
def assemble_prompt(t, doc, desc=None):
    cell = t["cell"]
    w = t["w_role"]
    ev = t["evidence"]
    sysm = ("You write a single high-quality FREE-TEXT question about an archival "
            "broadcast for a research/cataloging benchmark (no multiple choice). The question "
            "must require the video evidence (not general knowledge), be natural, and have exactly "
            "one defensible short answer. "
            "The ANSWER must be concrete and specific and must unambiguously answer the question: "
            "a name, organization, date, number, place, title, or a specific reason/claim stated "
            "in the evidence. It may be a phrase or short clause, but not an open-ended "
            "interpretation, opinion, or multi-sentence explanation. Keep the question phrasing natural. "
            "Phrasing rules: do NOT begin the question with 'Based on'; do NOT use the words "
            "'primary' or 'primarily'; vary your opening wording. "
            "Return JSON: {question, answer, rationale}.")

    if cell.startswith("two_hop") or t["element"].startswith("two_hop"):
        # name hidden -> video hop required; fact = grounding hop
        occ = ev.get("occupation") or "person"
        user = (f"Two-hop question. In the broadcast, a {occ} appears (do NOT use their name; "
                f"refer to them by their role/contribution so the viewer must identify them from "
                f"the video). External fact to ask about ({ev['field']}): {ev['answer']}.\n"
                f"Write a question whose answer is '{ev['answer']}', requiring (1) identifying the "
                f"{occ} from the broadcast and (2) external knowledge of their "
                f"{ev['field'].lower()}.")
        return sysm, user, {"answer": ev["answer"]}

    if t["element"].startswith("compare:"):
        user = (f"Comparison question (how/why). Two participants, {ev['a']} and {ev['b']}, appear "
                f"in the segment on '{ev['topic']}'. Using the transcript, write a question about "
                f"how their positions or framing DIFFER.\n\nTranscript:\n{evidence_text(doc, desc, segs0(t), segs1(t))}")
        return sysm, user, {}

    if w in ("why", "how"):
        ctx = evidence_text(doc, desc, ev["start_ms"], ev["end_ms"])
        verb = "Why" if w == "why" else "How"
        user = (f"Write a reasoning-requiring question about the segment titled "
                f"'{t['element'].split(':',1)[1]}'. Phrase it naturally, beginning with the word "
                f"\"{verb}\" (e.g. 'Why does X disagree that ...'). Its ANSWER must be a specific "
                f"reason or cause EXPLICITLY STATED in the transcript that unambiguously answers the "
                f"question (a phrase or short clause is fine), not an open-ended interpretation. Only "
                f"write the question if the transcript states such a specific reason.\n\n"
                f"Transcript:\n{ctx}")
        return sysm, user, {}

    if cell.startswith("Subject"):
        ctx = evidence_text(doc, desc, ev["start_ms"], ev["end_ms"])
        user = (f"Write a SUBJECT question: what is this segment about, or its main point? "
                f"Avoid trivial detail.\n\nTranscript:\n{ctx}")
        return sysm, user, {}

    if cell.startswith("Factual"):
        ctx = evidence_text(doc, desc, ev["start_ms"], ev["end_ms"])
        spk = ev.get("speaker") or ""
        who = (f" (for instance something stated by {spk})"
               if spk and not spk.upper().startswith("SPEAKER_") else "")
        user = (f"Write a factual/retrieval question about a specific, important stated fact in this "
                f"segment{who}: a name, figure, date, organization, title, or specific claim. The "
                f"answer must be that concrete fact and must unambiguously answer the question.\n\n"
                f"Transcript:\n{ctx}")
        return sysm, user, {}

    if t["element"] == "participants":
        user = (f"Write a CATALOGING question about who appears/their roles. Participants: "
                f"{', '.join(ev['names'])}.")
        return sysm, user, {}

    if t["element"] == "production":
        user = (f"Write a CATALOGING question about production metadata (program/date/credits). "
                f"Scene labels present: {ev['labels']}.")
        return sysm, user, {}

    return sysm, "Write a question about the broadcast.", {}


def segs0(t):
    return t["evidence"].get("start_ms", 0)


def segs1(t):
    return t["evidence"].get("end_ms", 0)


# ---------- LLM (aristotle vLLM) ----------
def llm_generate(sysm, user, vllm_url, model, api_key, timeout=90):
    try:
        r = requests.post(f"{vllm_url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0.7, "max_tokens": 400,
                                "messages": [{"role": "system", "content": sysm},
                                             {"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        if "</think>" in txt:                       # strip thinking-model traces
            txt = txt.split("</think>")[-1]
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {"raw": txt}
    except Exception as ex:
        return {"error": str(ex)}


def necessity_ok(question, element):
    """Two-hop: the entity name must NOT appear in the question."""
    if not element.startswith("two_hop:"):
        return True
    name = element.split(":", 1)[1]
    return name.split()[0].lower() not in (question or "").lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-from", default="qa-data/benchmark/v5_2/benchmark_combined.jsonl")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only-video", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-per-cell", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="no LLM; build targets+prompts+gates")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip videos whose output file already has generated rows (resume)")
    ap.add_argument("--evidence-mode", choices=["slice", "description"], default="slice",
                    help="slice=raw ASR window (baseline A); description=DP prose window (arm B1)")
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8200/v1"))
    ap.add_argument("--model", default=os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        wanted = None
    else:
        wanted = set()
        for line in open(args.videos_from):
            line = line.strip()
            if line:
                try:
                    wanted.add(json.loads(line)["video_id"])
                except Exception:
                    pass

    sal_files = sorted(SAL_DIR.glob("*.json"))
    sal_files = [p for p in sal_files if wanted is None or p.stem in wanted]
    if args.only_video:
        sal_files = [p for p in sal_files if args.only_video in p.stem]
    if args.limit:
        sal_files = sal_files[:args.limit]

    cell_dist = Counter()
    w_dist = Counter()
    gate = Counter()
    n_targets = n_q = 0

    print(f"{'DRY-RUN' if args.dry_run else 'GENERATE'}: {len(sal_files)} videos "
          + ("" if args.dry_run else f"| model={args.model} @ {args.vllm_url}"), flush=True)

    for sp in sal_files:
        vid = sp.stem
        if args.skip_done:
            outp = OUT_DIR / f"{vid}.json"
            if outp.exists():
                try:
                    if json.load(open(outp)).get("rows"):
                        print(f"{vid}: done, skipping", flush=True)
                        continue
                except Exception:
                    pass
        sal = json.load(open(sp))
        idxp = IDX_DIR / f"{vid}.json"
        grdp = GRD_DIR / f"{vid}.json"
        if not idxp.exists():
            continue
        doc = json.load(open(idxp))
        grounding = json.load(open(grdp)) if grdp.exists() else {"entities": []}
        desc = None
        if args.evidence_mode == "description":
            dp = DESC_DIR / f"{vid}.json"
            if not dp.exists():
                print(f"{vid}: no description (run build_video_description.py), skipping", flush=True)
                continue
            desc = json.load(open(dp))

        targets = build_targets(sal, grounding, doc, args.max_per_cell)
        n_targets += len(targets)
        out_rows = []
        prior_qs = []
        for t in targets:
            if t["cell"].startswith("REJECTED"):
                gate[t["extra"].get("reason", "rejected")] += 1
                if args.verbose:
                    print(f"  [gate] {t['element']}: {t['extra'].get('reason')}")
                continue
            cell_dist[t["cell"]] += 1
            w_dist[t["w_role"]] += 1
            sysm, user, meta = assemble_prompt(t, doc, desc)
            if args.dry_run:
                label = t["element"]
                if label.startswith("two_hop:"):     # hide name so necessity passes
                    label = "two_hop:" + t["evidence"].get("field", "fact")
                q = {"question": f"<PLACEHOLDER {t['cell']}/{t['w_role']} :: {label}>",
                     "answer": meta.get("answer", "<answer>"),
                     "distractors": ["<d1>", "<d2>", "<d3>"]}
            else:
                duser = user
                if prior_qs:
                    duser = ("Questions already asked for this video. Use DIFFERENT opening "
                             "wording from all of these (do not repeat their stems):\n"
                             + "\n".join("- " + p for p in prior_qs[-8:]) + "\n\n" + user)
                q = llm_generate(sysm, duser, args.vllm_url, args.model, args.api_key)
                if q.get("question"):
                    prior_qs.append(q["question"])
            if not necessity_ok(q.get("question", ""), t["element"]):
                gate["necessity_failed(name_in_q)"] += 1
                continue
            n_q += 1
            out_rows.append({"video_id": vid, "cell": t["cell"], "w_role": t["w_role"],
                             "element": t["element"], "evidence": t["evidence"],
                             "prompt_user": user if args.dry_run else None, "qa": q})
        json.dump({"video_id": vid, "n": len(out_rows), "rows": out_rows},
                  open(OUT_DIR / f"{vid}.json", "w"), indent=2)
        if args.verbose and out_rows:
            print(f"\n=== {vid}: {len(out_rows)} questions ===")
            for r in out_rows[:8]:
                print(f"  [{r['cell']}/{r['w_role']}] {r['qa'].get('question','')[:80]}")

    print("\n=== summary ===")
    print(f"videos: {len(sal_files)} | targets: {n_targets} | questions kept: {n_q}")
    print("by cell:", dict(cell_dist))
    print("by 5W role:", dict(w_dist))
    interesting = w_dist["why"] + w_dist["how"]
    tot = sum(w_dist.values()) or 1
    print(f"why/how share: {100*interesting/tot:.1f}%  (v5.2 baseline was 6.5%)")
    print("gates fired:", dict(gate))
    print(f"output -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
