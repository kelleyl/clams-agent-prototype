"""LLM-as-judge quality metric for cross-modal QA questions.

Scores each generated question on 4 taxonomy-grounded dimensions:
1. Cognitive Level (L1-L4, adapted from DramaQA + Bloom's)
2. Naturalness (colleague test)
3. Groundedness (evidence citation quality)
4. Information Need (cataloging/factual/subject/interpretive)

The judge model should be stronger than the generation model.

GEPA integration notes (DSPy 3.x):
- Metric functions must accept 5 positional params:
  (example, prediction, trace, pred_name, pred_trace)
- For richer optimization signal, return dspy.Prediction(score=..., feedback=...)
  instead of a bare float. GEPA uses the feedback text to guide prompt mutations.
- See: https://dspy.ai/api/optimizers/GEPA/overview/
"""
import json
import logging
import requests

import dspy

logger = logging.getLogger(__name__)

# Default judge config — override via environment or constructor
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_API_BASE = "http://localhost:8100/v1"
DEFAULT_BACKEND = "vllm"

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of cross-modal video comprehension questions.

You will score a question-answer pair on 4 dimensions using the taxonomy below.
Each dimension is scored 0-5. Return a JSON object with your scores.

## Dimension 1: Cognitive Level (0-5)
Based on DramaQA's cognitive hierarchy and Bloom's taxonomy:
- 0-1: L1 (Identify) — perceives a fact from a single source. Trivial.
- 2: L2 (Retrieve) — locates specific information across sources.
- 3: L3 (Integrate) — combines information from multiple modalities. BEST.
- 4: L3-L4 transition — integration with some reasoning.
- 5: L4 (Analyze) — reasons about relationships across modalities and segments.

We WANT questions at L2-L3 primarily, with some L4. Pure L1 should score 0-1.

## Dimension 2: Naturalness (0-5)
The "colleague test" — would you phrase it this way to a colleague watching the video?
- 0: Database-style query ("enumerate all named entities"), over-technical
- 1: Stilted or formulaic phrasing
- 2: Understandable but awkward
- 3: Reasonable but somewhat formulaic
- 4: Sounds like a researcher or journalist would ask this
- 5: Completely natural, specific, and purposeful

Filters out: database queries, over-specific technical questions, obviously answerable questions, questions about artifacts.

## Dimension 3: Groundedness (0-5)
Is the answer verifiable from the cited evidence?
- 0: No grounding provided, or grounding is fabricated
- 1: Grounding exists but doesn't match the evidence
- 2: Grounding from one modality only, weakly connected
- 3: Grounding exists from one modality, well connected
- 4: Grounding cites evidence from 2 modalities
- 5: Grounding cites specific evidence from 2+ modalities with clear connection

## Dimension 4: Information Need (0-5)
Does this serve a real information need? Types from archival reference:
- 0: No practical use case
- 1: Trivially obvious or about artifacts
- 2: Marginally useful
- 3: Useful for subject access or factual retrieval
- 4: Directly useful for research or media analysis
- 5: Directly useful for cataloging, fact-checking, or scholarly analysis

Return ONLY a JSON object:
{
  "cognitive_level": <0-5>,
  "naturalness": <0-5>,
  "groundedness": <0-5>,
  "information_need": <0-5>,
  "brief_rationale": "<one sentence>"
}
"""


def _call_judge(prompt: str, model: str = DEFAULT_JUDGE_MODEL,
                api_base: str = DEFAULT_API_BASE,
                backend: str = DEFAULT_BACKEND) -> dict:
    """Call the judge LLM and parse its JSON response.

    Supports both vLLM (OpenAI-compatible /v1/chat/completions) and
    Ollama (/api/chat) backends.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    try:
        if backend == "vllm":
            url = f"{api_base.rstrip('/')}/chat/completions"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 500,
                    "response_format": {"type": "json_object"},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
        else:  # ollama
            url = f"{api_base.rstrip('/')}/api/chat"
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0, "num_predict": 500},
                },
                timeout=120,
            )
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "")

        return json.loads(text)
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Judge call failed: {e}")
        return {}


def judge_question(question_dict: dict, segment_evidence: str,
                   model: str = DEFAULT_JUDGE_MODEL,
                   api_base: str = DEFAULT_API_BASE,
                   backend: str = DEFAULT_BACKEND) -> dict:
    """Score a single question against the taxonomy.

    Args:
        question_dict: Dict with question, answer, modalities_required,
                       reasoning_type, grounding fields.
        segment_evidence: The source evidence the question was generated from.
        model: Judge model name.
        api_base: API base URL.
        backend: "vllm" or "ollama".

    Returns:
        Dict with cognitive_level, naturalness, groundedness, information_need
        scores (each 0-5), or empty dict on failure.
    """
    grounding = question_dict.get("grounding", {})
    if isinstance(grounding, dict):
        grounding_str = json.dumps(grounding, indent=2)
    else:
        grounding_str = str(grounding)

    prompt = f"""Score this cross-modal video comprehension question.

SEGMENT EVIDENCE (abbreviated):
{segment_evidence[:1500]}

QUESTION: {question_dict.get('question', '')}
ANSWER: {question_dict.get('answer', '')}
MODALITIES REQUIRED: {question_dict.get('modalities_required', [])}
REASONING TYPE: {question_dict.get('reasoning_type', '')}
GROUNDING:
{grounding_str}

Score on all 4 dimensions (0-5 each). Return JSON only."""

    return _call_judge(prompt, model=model, api_base=api_base, backend=backend)


def compute_question_score(scores: dict) -> float:
    """Compute normalized composite score from dimension scores.

    Returns value in [0, 1].
    """
    dims = ["cognitive_level", "naturalness", "groundedness", "information_need"]
    values = [scores.get(d, 0) for d in dims]
    total = sum(values)
    max_total = len(dims) * 5  # 20
    return total / max_total


def _format_feedback(all_scores: list[dict]) -> str:
    """Format dimension scores into human-readable feedback for GEPA."""
    if not all_scores:
        return "No questions were generated."

    dims = ["cognitive_level", "naturalness", "groundedness", "information_need"]
    dim_avgs = {}
    for dim in dims:
        vals = [s.get(dim, 0) for s in all_scores if dim in s]
        dim_avgs[dim] = sum(vals) / len(vals) if vals else 0

    # Identify weakest dimension for targeted feedback
    weakest = min(dim_avgs, key=dim_avgs.get)
    strongest = max(dim_avgs, key=dim_avgs.get)

    lines = [f"Scored {len(all_scores)} questions."]
    for dim in dims:
        lines.append(f"  {dim}: {dim_avgs[dim]:.1f}/5")

    if dim_avgs[weakest] < 3.0:
        hints = {
            "cognitive_level": "Questions are too simple (L1). Generate questions requiring cross-modal integration (L2-L3).",
            "naturalness": "Questions sound formulaic. Phrase them as a researcher or journalist would ask.",
            "groundedness": "Answers lack evidence citations. Include specific excerpts from speech, visual, and OCR.",
            "information_need": "Questions lack practical value. Target cataloging, fact-checking, or research needs.",
        }
        lines.append(f"Weakest: {weakest} ({dim_avgs[weakest]:.1f}/5). {hints.get(weakest, '')}")

    rationales = [s.get("brief_rationale", "") for s in all_scores if s.get("brief_rationale")]
    if rationales:
        lines.append(f"Judge notes: {rationales[0]}")

    return " ".join(lines)


def qa_quality_metric(example, prediction, trace=None,
                      pred_name=None, pred_trace=None,
                      judge_model: str = DEFAULT_JUDGE_MODEL,
                      judge_api_base: str = DEFAULT_API_BASE,
                      backend: str = DEFAULT_BACKEND) -> float:
    """GEPA-compatible metric: score generated questions with LLM judge.

    Accepts 5 positional params as required by GEPA's metric protocol.
    Returns a plain float — GEPA's installed version (gepa 0.0.7) asserts
    that feedback scores exactly equal module-level scores, which breaks
    when returning dspy.Prediction(score=..., feedback=...) because the
    LLM judge is non-deterministic. Plain floats avoid this assertion.

    Args:
        example: DSPy Example with segment_evidence field.
        prediction: DSPy Prediction with questions field (list of QuestionOutput).
        trace: Optional trace (unused).
        pred_name: Name of the predictor being evaluated (GEPA param).
        pred_trace: Trace of the predictor (GEPA param).

    Returns:
        Float in [0, 1] — average composite quality across generated questions.
    """
    questions = getattr(prediction, "questions", [])
    if not questions:
        return 0.0

    segment_evidence = getattr(example, "segment_evidence", "")
    score_values = []

    for q in questions:
        if hasattr(q, "model_dump"):
            q_dict = q.model_dump()
        elif isinstance(q, dict):
            q_dict = q
        else:
            continue

        scores = judge_question(
            q_dict, segment_evidence,
            model=judge_model, api_base=judge_api_base,
            backend=backend,
        )
        if scores:
            score_values.append(compute_question_score(scores))

    if not score_values:
        return 0.0

    return sum(score_values) / len(score_values)


def qa_quality_metric_cached(judge_model: str = DEFAULT_JUDGE_MODEL,
                             judge_api_base: str = DEFAULT_API_BASE,
                             backend: str = DEFAULT_BACKEND):
    """Return a metric function with judge config baked in.

    The returned function has the 5-parameter GEPA signature and returns
    a plain float (not dspy.Prediction) to avoid GEPA's score-mismatch
    assertion with non-deterministic LLM judges.

    Usage:
        metric = qa_quality_metric_cached(judge_model="Qwen/Qwen3.5-9B")
        gepa = dspy.GEPA(metric=metric, ...)
    """
    def metric_fn(example, prediction, trace=None,
                  pred_name=None, pred_trace=None):
        return qa_quality_metric(
            example, prediction, trace, pred_name, pred_trace,
            judge_model=judge_model, judge_api_base=judge_api_base,
            backend=backend,
        )
    metric_fn.__name__ = "qa_quality_metric"
    return metric_fn
