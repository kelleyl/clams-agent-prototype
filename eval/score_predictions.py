#!/usr/bin/env python3
"""Score model predictions against the dual-format benchmark.

Supports:
- Multiple-choice scoring (exact match on option letter)
- Free-text scoring via LLM-as-judge (Video-ChatGPT protocol: 5 dimensions)
- Keyword overlap (fallback, no LLM needed)
- Auto-detection of format from benchmark questions

Usage:
    # Score against dual-format benchmark (auto-detects MC vs free-text)
    python eval/score_predictions.py \
        --predictions eval/results/predictions.jsonl \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --judge-model qwen3.5:9b

    # Score MC-only predictions
    python eval/score_predictions.py \
        --predictions eval/results/mc_predictions.jsonl \
        --mode mc-exact

    # Score free-text with keyword overlap (no LLM needed)
    python eval/score_predictions.py \
        --predictions eval/results/ft_predictions.jsonl \
        --mode keyword
"""
import argparse
import json
import os
import sys
import requests
from collections import defaultdict
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


# ---------- Scoring functions ----------

def score_exact_match(predicted, expected):
    """Exact string match (for MC answers like 'A', 'B', 'C', 'D')."""
    pred = predicted.strip().upper()
    exp = expected.strip().upper()
    # Also handle "A)" or "A." formats
    if len(pred) > 1:
        pred = pred[0]
    return pred == exp


def score_retrieval_set(predicted, expected, iou_thresh=0.3):
    """Set F1 for retrieval_set questions. A predicted segment matches a gold
    segment if same video_id and temporal IoU >= iou_thresh (or same seg_id).
    Accepts predicted as list of dicts, or a string of 'video_id start-end' /
    seg_id mentions (best-effort parse for free-form model output)."""
    def as_segs(x):
        if isinstance(x, list):
            return [s for s in x if isinstance(s, dict)]
        segs = []
        import re as _re
        for m in _re.finditer(r"([\w.\-]*cpb[\w.\-]+|[A-Za-z0-9_\-]{6,})[^\d]{0,12}(\d+)m",
                              str(x)):
            segs.append({"video_id": m.group(1), "start_ms": int(m.group(2)) * 60000,
                         "end_ms": (int(m.group(2)) + 1) * 60000})
        return segs

    def iou(a, b):
        s1, e1 = a.get("start_ms") or 0, a.get("end_ms") or 0
        s2, e2 = b.get("start_ms") or 0, b.get("end_ms") or 0
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = max(e1, e2) - min(s1, s2)
        return inter / union if union > 0 else 0.0

    P, G = as_segs(predicted), as_segs(expected)
    if not G:
        return 0.0, 0.0, 0.0
    if not P:
        return 0.0, 0.0, 0.0
    matched_g = set()
    tp = 0
    for p in P:
        for gi, g in enumerate(G):
            if gi in matched_g:
                continue
            same_vid = (p.get("video_id") or "")[:20] == (g.get("video_id") or "")[:20]
            if same_vid and (p.get("seg_id") == g.get("seg_id") or iou(p, g) >= iou_thresh):
                matched_g.add(gi)
                tp += 1
                break
    prec = tp / len(P)
    rec = tp / len(G)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec


def score_keyword_overlap(predicted, expected):
    """Score based on keyword overlap between predicted and expected."""
    if not predicted or not expected:
        return 0.0
    pred_words = set(predicted.lower().split())
    exp_words = set(expected.lower().split())
    stops = {"the", "a", "an", "is", "was", "were", "are", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this", "it"}
    pred_words -= stops
    exp_words -= stops
    if not exp_words:
        return 0.0
    overlap = pred_words & exp_words
    precision = len(overlap) / max(1, len(pred_words))
    recall = len(overlap) / max(1, len(exp_words))
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 3)


# Video-ChatGPT evaluation protocol (adapted)
# Scores on 5 dimensions, each 0-5, then averages
JUDGE_PROMPT = """You are evaluating a video question-answering system. Given a question about a broadcast news video, the expected answer, the model's predicted answer, and the evidence gathered by the system, score the prediction.

Question: {question}
Required modalities: {modalities}
Reference answer: {expected}
Predicted answer: {predicted}
Evidence gathered:
{evidence}

Score each dimension 0-5:
1. **Correctness**: Does the prediction match the key facts in the reference? (0=completely wrong, 5=all facts correct)
2. **Completeness**: Does it cover all important points from the reference? (0=nothing covered, 5=fully complete)
3. **Relevance**: Is the prediction focused on answering the question? (0=off-topic, 5=directly answers)
4. **Cross-modal grounding**: Is the answer supported by the gathered evidence, and does it appear to use the required modalities when multiple modalities are needed? (0=unsupported, 5=clearly grounded in the gathered evidence)
5. **Specificity**: Does it provide specific details rather than vague generalities? (0=completely vague, 5=precise and specific)

Return a JSON object with:
- "correctness": 0-5
- "completeness": 0-5
- "relevance": 0-5
- "cross_modal": 0-5
- "specificity": 0-5
- "explanation": brief overall assessment (1-2 sentences)
"""


def _format_judge_evidence(pred):
    evidence = pred.get("evidence", []) or []
    if evidence:
        chunks = []
        for ev in evidence[:4]:
            tool = ev.get("tool", "tool")
            params = ev.get("params", {})
            result = (ev.get("result", "") or "")[:250]
            chunks.append(f"[{tool} {params}] {result}")
        return "\n".join(chunks)
    tool_trace = pred.get("tool_trace", []) or []
    if tool_trace:
        return "\n".join(f"[{tc.get('tool', 'tool')}] params={tc.get('params', {})}" for tc in tool_trace[:4])
    return "No evidence recorded."


def score_llm_judge(predicted, expected, question, judge_url, judge_model, pred_row=None, benchmark_row=None):
    """Multi-dimensional LLM-as-judge scoring (Video-ChatGPT protocol)."""
    pred_row = pred_row or {}
    benchmark_row = benchmark_row or {}
    prompt = JUDGE_PROMPT.format(
        question=question,
        modalities=", ".join(benchmark_row.get("modalities_required", pred_row.get("modalities_required", []))) or "unknown",
        expected=expected,
        predicted=predicted,
        evidence=_format_judge_evidence(pred_row),
    )

    try:
        resp = requests.post(
            f"{judge_url}/api/chat",
            json={
                "model": judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 300},
            },
            timeout=60,
        )
        resp.raise_for_status()
        result = json.loads(resp.json()["message"]["content"])

        dims = {
            "correctness": result.get("correctness", 0),
            "completeness": result.get("completeness", 0),
            "relevance": result.get("relevance", 0),
            "cross_modal": result.get("cross_modal", 0),
            "specificity": result.get("specificity", 0),
        }
        # Clamp to 0-5
        for k in dims:
            dims[k] = max(0, min(5, dims[k]))

        avg_score = sum(dims.values()) / len(dims)
        explanation = result.get("explanation", "")

        return round(avg_score, 2), dims, explanation
    except Exception as e:
        return -1, {}, f"Judge error: {e}"


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Score QA predictions (dual-format)")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Predictions JSONL (must have predicted_answer field)")
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Benchmark JSONL for auto-format detection and reference answers")
    parser.add_argument("--mode", choices=["auto", "keyword", "llm-judge", "mc-exact"],
                        default="auto",
                        help="Scoring mode (auto uses benchmark format field)")
    parser.add_argument("--judge-url", type=str, default=OLLAMA_URL)
    parser.add_argument("--judge-model", type=str, default="qwen3.5:9b")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with open(args.predictions) as f:
        predictions = [json.loads(l) for l in f]

    # Load benchmark for reference answers and format info
    benchmark_by_id = {}
    if args.benchmark:
        with open(args.benchmark) as f:
            for line in f:
                q = json.loads(line)
                benchmark_by_id[q["id"]] = q

    print(f"Scoring {len(predictions)} predictions...")
    if benchmark_by_id:
        mc_count = sum(1 for q in benchmark_by_id.values() if q.get("format") == "multiple_choice")
        ft_count = sum(1 for q in benchmark_by_id.values() if q.get("format") in ("free_text", "freetext"))
        rs_count = sum(1 for q in benchmark_by_id.values() if q.get("format") == "retrieval_set")
        print(f"Benchmark: {mc_count} MC + {ft_count} free-text + {rs_count} retrieval-set questions")

    # Score each prediction
    mc_scores = []
    ft_scores = []
    ft_dims_all = defaultdict(list)
    by_type = defaultdict(lambda: {"mc": [], "ft": []})
    by_video = defaultdict(lambda: {"mc": [], "ft": []})
    tool_usage = defaultdict(int)
    model_usage = defaultdict(int)
    task_mode_usage = defaultdict(int)

    for i, pred in enumerate(predictions):
        qid = pred.get("question_id", pred.get("id", f"q-{i}"))
        predicted = pred.get("predicted_answer", "") or ""
        question = pred.get("question", "")
        reasoning_type = pred.get("reasoning_type", "unknown")
        video_id = pred.get("video_id", "")
        for tc in pred.get("tool_trace", []) or []:
            tool_usage[tc.get("tool", "unknown")] += 1
            params = tc.get("params", {}) or {}
            if params.get("model"):
                model_usage[params["model"]] += 1
            if params.get("task_mode"):
                task_mode_usage[params["task_mode"]] += 1

        # Determine format and expected answer
        bench_q = benchmark_by_id.get(qid, {})
        fmt = bench_q.get("format", "")

        if args.mode == "auto":
            if fmt == "multiple_choice":
                mode = "mc-exact"
                expected = bench_q.get("mc_correct", pred.get("expected_answer", ""))
            elif fmt == "retrieval_set":
                mode = "retrieval-set"
                expected = bench_q.get("answer", pred.get("expected_answer", []))
            elif fmt in ("free_text", "freetext"):
                mode = "llm-judge"
                expected = bench_q.get("reference_answer",
                                       bench_q.get("answer",
                                                   pred.get("expected_answer", "")))
            else:
                # Fallback: guess from prediction content
                if predicted.strip().upper() in ("A", "B", "C", "D"):
                    mode = "mc-exact"
                    expected = pred.get("expected_answer", "")
                else:
                    mode = "llm-judge"
                    expected = pred.get("expected_answer", "")
        else:
            mode = args.mode
            expected = pred.get("expected_answer",
                        bench_q.get("reference_answer",
                        bench_q.get("mc_correct", "")))

        # Score
        if mode == "mc-exact":
            correct = score_exact_match(predicted, expected)
            score = 1.0 if correct else 0.0
            pred["score"] = score
            pred["score_correct"] = correct
            pred["score_format"] = "mc"
            mc_scores.append(score)
            by_type[reasoning_type]["mc"].append(score)
            by_video[video_id]["mc"].append(score)

        elif mode == "llm-judge":
            avg_score, dims, explanation = score_llm_judge(
                predicted, expected, question,
                args.judge_url, args.judge_model,
                pred_row=pred, benchmark_row=bench_q
            )
            pred["score"] = avg_score
            pred["score_dims"] = dims
            pred["score_explanation"] = explanation
            pred["score_format"] = "ft"
            if avg_score >= 0:
                ft_scores.append(avg_score)
                for dim, val in dims.items():
                    ft_dims_all[dim].append(val)
                by_type[reasoning_type]["ft"].append(avg_score)
                by_video[video_id]["ft"].append(avg_score)

        elif mode == "retrieval-set":
            f1, prec, rec = score_retrieval_set(predicted, expected)
            pred["score"] = f1
            pred["score_dims"] = {"precision": prec, "recall": rec}
            pred["score_format"] = "retrieval_set"
            ft_scores.append(f1)
            by_type[reasoning_type]["ft"].append(f1)
            by_video[video_id]["ft"].append(f1)

        elif mode == "keyword":
            f1 = score_keyword_overlap(predicted, expected)
            score = round(f1 * 5, 1)
            pred["score"] = score
            pred["score_f1"] = f1
            pred["score_format"] = "ft"
            ft_scores.append(score)
            by_type[reasoning_type]["ft"].append(score)

        if (i + 1) % 10 == 0:
            mc_avg = sum(mc_scores) / max(1, len(mc_scores)) if mc_scores else 0
            ft_avg = sum(ft_scores) / max(1, len(ft_scores)) if ft_scores else 0
            print(f"  [{i+1}/{len(predictions)}] MC acc: {mc_avg:.1%}  FT avg: {ft_avg:.2f}/5")

    # ---------- Summary ----------
    print(f"\n{'='*60}")
    print(f"Scoring Results")
    print(f"{'='*60}")

    if mc_scores:
        mc_acc = sum(mc_scores) / len(mc_scores)
        print(f"\nMultiple Choice (n={len(mc_scores)}):")
        print(f"  Accuracy: {mc_acc:.1%} ({sum(int(s) for s in mc_scores)}/{len(mc_scores)})")

    if ft_scores:
        ft_avg = sum(ft_scores) / len(ft_scores)
        ft_errors = sum(1 for s in ft_scores if s < 0)
        print(f"\nFree-Text (n={len(ft_scores)}):")
        print(f"  Overall average: {ft_avg:.2f}/5")
        if ft_dims_all:
            print(f"  By dimension:")
            for dim in ["correctness", "completeness", "relevance", "cross_modal", "specificity"]:
                vals = ft_dims_all.get(dim, [])
                if vals:
                    davg = sum(vals) / len(vals)
                    print(f"    {dim:15s} {davg:.2f}/5")

    if by_type:
        print(f"\nBy reasoning type:")
        print(f"  {'type':20s} {'MC acc':>8s} {'FT avg':>8s}")
        for rtype in sorted(by_type.keys()):
            data = by_type[rtype]
            mc_str = f"{sum(data['mc'])/len(data['mc']):.1%}" if data["mc"] else "  -"
            ft_str = f"{sum(data['ft'])/len(data['ft']):.2f}" if data["ft"] else "  -"
            n_mc = len(data["mc"])
            n_ft = len(data["ft"])
            print(f"  {rtype:20s} {mc_str:>8s} (n={n_mc})  {ft_str:>8s} (n={n_ft})")

    if tool_usage:
        print(f"\nTool usage:")
        for tool, count in sorted(tool_usage.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {tool:20s} {count}")
    if model_usage:
        print(f"\nModel usage:")
        for model, count in sorted(model_usage.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {model:20s} {count}")
    if task_mode_usage:
        print(f"\nTask mode usage:")
        for mode_name, count in sorted(task_mode_usage.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {mode_name:20s} {count}")

    # Grounding metrics (if predictions include verification data)
    grounding_verified = [p for p in predictions if p.get("verification", {}).get("any_verified")]
    multi_modal = [p for p in predictions if p.get("verification", {}).get("multi_modal_citation")]
    has_verification = any(p.get("verification") for p in predictions)

    if has_verification:
        total_with_v = sum(1 for p in predictions if p.get("verification"))
        print(f"\nGrounding Metrics (n={total_with_v}):")
        print(f"  Evidence verified:     {len(grounding_verified):4d} ({len(grounding_verified)/max(1,total_with_v):.0%})")
        print(f"  Multi-modal citations: {len(multi_modal):4d} ({len(multi_modal)/max(1,total_with_v):.0%})")

        # Grounded accuracy (only count answers with verified grounding)
        if mc_scores and grounding_verified:
            grounded_mc = [p for p in predictions
                          if p.get("score_format") == "mc"
                          and p.get("verification", {}).get("any_verified")]
            if grounded_mc:
                grounded_acc = sum(p.get("score", 0) for p in grounded_mc) / len(grounded_mc)
                print(f"  Grounded MC accuracy:  {grounded_acc:.1%} (n={len(grounded_mc)})")

    # Save scored predictions
    out_path = args.output or args.predictions.with_suffix(".scored.jsonl")
    with open(out_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    print(f"\nScored predictions: {out_path}")


if __name__ == "__main__":
    main()
