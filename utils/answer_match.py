"""Shared free-text answer matcher for v6 (no multiple choice).

Decides whether a predicted free-text answer matches the gold answer. Used by
both QC gates: the blind check (does a no-video model produce the answer ->
leaky) and the round-trip check (does re-answering from evidence reproduce it).
Deterministic first (normalization + containment + ANLS*-style fuzzy + token
overlap); an LLM judge is only needed as a fallback for paraphrased phrase
answers, handled by the caller.
"""
import difflib
import re

_ARTICLES = {"a", "an", "the"}


def normalize(s):
    s = (s or "").lower().replace(",", "")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _tokset(s):
    return set(normalize(s).split())


def match(pred, gold, ratio_thresh=0.72, token_thresh=0.6):
    """True if pred and gold are the same answer (deterministic)."""
    np_, ng = normalize(pred), normalize(gold)
    if not np_ or not ng:
        return False
    if np_ == ng:
        return True
    # containment: short gold inside a longer/verbose prediction (or vice versa)
    if ng in np_ or np_ in ng:
        return True
    # ANLS*-style normalized similarity
    if difflib.SequenceMatcher(None, np_, ng).ratio() >= ratio_thresh:
        return True
    # most of gold's content words appear in the prediction (paraphrase)
    g, p = _tokset(gold), _tokset(pred)
    if g and len(g & p) / len(g) >= token_thresh:
        return True
    return False


def uncertain(pred, gold):
    """Near-miss zone where an LLM judge is worth calling (deterministic said
    no, but there is partial overlap)."""
    g, p = _tokset(gold), _tokset(pred)
    if not g or not p:
        return False
    return 0.2 <= len(g & p) / len(g) < 0.6


def judge_match(question, pred, gold, url, model, api_key="EMPTY", timeout=60):
    """Ask an LLM whether two answers mean the same thing for this question.
    Returns True/False, or None on error."""
    import requests
    user = ("For the question below, do the two answers mean the same thing? "
            "Answer YES or NO only. /no_think\n\n"
            f"Question: {question}\nAnswer A: {pred}\nAnswer B: {gold}\nSame meaning?")
    try:
        r = requests.post(f"{url.rstrip('/')}/chat/completions",
                          headers={"Authorization": f"Bearer {api_key}",
                                   "Content-Type": "application/json"},
                          json={"model": model, "temperature": 0, "max_tokens": 8,
                                "messages": [{"role": "user", "content": user}]},
                          timeout=timeout)
        r.raise_for_status()
        ans = r.json()["choices"][0]["message"]["content"]
        ans = (ans.split("</think>")[-1] if "</think>" in ans else ans).strip().upper()
        return "YES" in ans
    except Exception:
        return None


def decide_match(question, pred, gold, judge_cfg=None):
    """Clear cases by string; ambiguous fuzzy cases by LLM judge if configured.
    Catches both fuzzy false-negatives (name variants) and false-positives
    (overlapping-but-different answers)."""
    np_, ng = normalize(pred), normalize(gold)
    if not np_ or not ng:
        return False
    if np_ == ng or ng in np_ or np_ in ng:
        return True
    if judge_cfg:
        j = judge_match(question, pred, gold, **judge_cfg)
        if j is not None:
            return j
    return match(pred, gold)
