"""Lightweight context retrieval over a video index.

Given a query (a question or topic+entity string), return the most relevant
text passages from the index's grounded text layers (ASR + OCR), ACROSS
segments, not just a fixed anchor window. Lexical (token-overlap) scoring, no
embeddings, no network. Captions are excluded by default because single-frame
captions are hallucination-prone; pass include_captions=True to add them.

Used by benchmark construction (generation, dense paraphrasing, round-trip
verification) so each step sees the evidence actually relevant to the target,
which fixes unresolved cross-segment references and false round-trip rejects.
"""
import re

_STOP = set(
    "the a an of to in on for and or but is are was were be been it this that with as at by from "
    "his her their its he she they who what why how which when where will would can could has have "
    "had not no do does did about into over under than then them you your we our us i".split()
)


def _items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def _toks(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return [w for w in s.split() if len(w) > 2 and w not in _STOP]


def gather_text_items(doc, include_captions=False):
    out = []
    for k, lay in doc.get("layers", {}).items():
        kl = k.lower()
        if kl.startswith("asr"):
            src = "asr"
        elif "ocr" in kl:
            src = "ocr"
        elif include_captions and kl.startswith("caption"):
            src = "caption"
        else:
            continue
        for it in _items(lay):
            if isinstance(it, dict):
                t = (it.get("text") or "").strip()
                if t:
                    out.append({"start_ms": it.get("start_ms"), "end_ms": it.get("end_ms"),
                                "text": t, "layer": src})
    return out


def retrieve_context(doc, query, k=6, max_chars=1500, include_captions=False):
    """Return up to k most query-relevant passages, joined and time-ordered."""
    q = set(_toks(query))
    if not q:
        return ""
    scored = []
    for it in gather_text_items(doc, include_captions=include_captions):
        toks = set(_toks(it["text"]))
        if not toks:
            continue
        overlap = len(q & toks)
        if overlap:
            # normalize slightly by passage length so long ASR turns don't dominate
            scored.append((overlap / (1 + len(toks) ** 0.5), overlap, it))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    top = [it for _, _, it in scored[:k]]
    top.sort(key=lambda it: it.get("start_ms") or 0)
    parts, seen, total = [], set(), 0
    for it in top:
        key = it["text"][:60]
        if key in seen:
            continue
        seen.add(key)
        chunk = f"[{it['layer']}] {it['text']}"
        parts.append(chunk)
        total += len(chunk)
        if total > max_chars:
            break
    return "\n".join(parts)
