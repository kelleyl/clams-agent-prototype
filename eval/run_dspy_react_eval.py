#!/usr/bin/env python3
"""DSPy ReAct eval harness for V7 warm-index tools with optional
metadata-rich tool descriptions.

Lets us A/B handcrafted descriptions (mode='plain') against:
  - mode='metadata': adds CLAMS app provenance from data/app_directory.json
  - mode='reliability': adds empirical stats from data/tool_stats.json
  - mode='full': metadata + reliability + per-video coverage

Granularity:
  - 'collapsed' (5 tools): one tool per task; model selected via parameter
  - 'split' (13 tools): one tool per (task, model) — discrete choice for GEPA

Output format mirrors run_policy_answerer_eval.py so existing scoring
scripts (score_grounding.py, etc.) work unchanged.

Usage:
    python eval/run_dspy_react_eval.py \\
        --benchmark qa-data/benchmark/v5_1/v5_1_benchmark.jsonl \\
        --output eval/results/v5_1_dspy_react_full.jsonl \\
        --vllm-url http://localhost:8889/v1 \\
        --model Qwen/Qwen3.5-9B \\
        --description-mode full --granularity collapsed \\
        --coverage-source auto --max-turns 8 --shard 0/4 --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy

from eval.v7_tools import V7Executor
from eval.dspy_v7_tools import make_dspy_tools, CoverageTracker


# ============================================================
# DSPy signature for the QA task
# ============================================================

class VideoQA(dspy.Signature):
    """Answer a question about a video by calling tools to gather evidence,
then synthesizing a final answer.

For multiple-choice questions, the final answer must be exactly one of the
provided option letters (e.g. 'A', 'B', 'C', or 'D').
For free-text questions, give a concise answer in 1-2 sentences.
"""
    question: str = dspy.InputField(desc="The question to answer")
    options: str = dspy.InputField(desc="MC options if any, e.g. 'A. Foo\\nB. Bar\\n...' or '(free-text)'")
    video_context: str = dspy.InputField(desc="Brief context about the video (id, duration)")
    answer: str = dspy.OutputField(desc="Final answer. For MC: just the letter (A/B/C/D).")


# ============================================================
# Per-question runner
# ============================================================

def format_options(q: dict) -> str:
    opts = q.get("mc_options") or {}
    if not opts:
        return "(free-text)"
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))


def expected_answer(q: dict) -> str:
    if q.get("format") == "multiple_choice":
        return str(q.get("mc_correct") or q.get("expected_answer") or "")
    return str(q.get("answer") or q.get("expected_answer") or "")


def extract_letter(raw: str) -> str:
    """Extract the chosen letter from a free-form answer.

    Priority: a single-letter answer; an explicit 'Answer: X' tag; a letter
    immediately followed by a closing brace (X. or X)); a standalone letter.
    """
    import re
    s = (raw or "").strip().upper()
    if not s:
        return ""
    # Single letter only
    if len(s) == 1 and s in "ABCD":
        return s
    # Explicit answer tag
    m = re.search(r"\bANSWER\s*[:=\-]\s*([ABCD])\b", s)
    if m:
        return m.group(1)
    # X. or X) or X: at end or after whitespace
    m = re.search(r"(?:^|[\s\(\[])([ABCD])(?:[\.\)\]\:]|$)", s)
    if m:
        return m.group(1)
    # Standalone letter surrounded by non-letters
    m = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", s)
    if m:
        return m.group(1)
    return s[:1]


def run_one_question(q: dict, indexes: dict, *,
                     description_mode: str,
                     granularity: str,
                     coverage_source: str,
                     tracker: CoverageTracker,
                     max_turns: int,
                     lm: dspy.LM,
                     module_kind: str = "react",
                     deno_command: list = None,
                     enable_network: list = None,
                     sub_lm: dspy.LM = None) -> dict:
    video_id = q.get("video_id")
    idx = indexes.get(video_id)
    if not idx:
        return {
            "question_id": q.get("id"), "video_id": video_id,
            "error": f"no index for {video_id}",
        }

    executor = V7Executor(idx, video_id=video_id, execution_mode="warm_index")
    tools = make_dspy_tools(
        executor,
        mode=description_mode,
        granularity=granularity,
        video_id=video_id,
        tracker=tracker,
        coverage_source=coverage_source,
    )

    if module_kind == "rlm":
        interp = None
        if deno_command:
            interp = dspy.PythonInterpreter(
                deno_command=deno_command,
                enable_network_access=enable_network or [],
                tools={t.name: t.func for t in tools},
            )
        agent = dspy.RLM(
            VideoQA, tools=tools,
            max_iterations=max_turns,
            max_llm_calls=max_turns * 2, max_output_chars=10000,
            sub_lm=sub_lm or lm,
            interpreter=interp,
        )
    else:
        agent = dspy.ReAct(VideoQA, tools=tools, max_iters=max_turns)

    duration_s = (idx.get("duration_ms") or 0) / 1000.0
    video_context = (
        f"video_id={video_id}; duration={duration_s:.0f}s "
        f"({duration_s/60:.1f} min)"
    )

    t0 = time.time()
    raw_answer = ""
    react_trace = None
    err = None
    try:
        with dspy.context(lm=lm):
            pred = agent(
                question=q["question"],
                options=format_options(q),
                video_context=video_context,
            )
        raw_answer = str(pred.answer)
        react_trace = getattr(pred, "trajectory", None) or getattr(pred, "repl_history", None)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    elapsed = time.time() - t0
    fmt = q.get("format", "multiple_choice")
    expected = expected_answer(q)
    if fmt == "multiple_choice":
        predicted = extract_letter(raw_answer)
        is_correct = predicted == expected.strip().upper()[:1]
    else:
        predicted = raw_answer.strip()
        is_correct = False  # free-text needs posthoc scoring

    # Reconstruct tool_calls / tool_trace from the executor evidence list
    tool_calls = [
        {"tool": ev.tool, "time_range": list(ev.time_range)}
        for ev in executor.evidence
    ]
    return {
        "question_id": q.get("id"),
        "video_id": video_id,
        "question": q["question"],
        "format": fmt,
        "expected_answer": expected,
        "predicted_answer": predicted,
        "raw_answer": str(raw_answer)[:500],
        "agent_mode": "dspy_react",
        "description_mode": description_mode,
        "granularity": granularity,
        "tool_calls": [tc["tool"] for tc in tool_calls],
        "tool_trace": tool_calls,
        "evidence": [
            {"tool": ev.tool, "time_range": list(ev.time_range),
             "observation": (ev.observation or "")[:500]}
            for ev in executor.evidence
        ],
        "num_tool_calls": len(tool_calls),
        "evidence_count": len(executor.evidence),
        "reasoning_type": q.get("reasoning_type", ""),
        "modalities_required": q.get("modalities_required", []),
        "source_segment_times": q.get("source_segment_times", []),
        "inline_correct": is_correct,
        "needs_posthoc_scoring": fmt != "multiple_choice",
        "v7_metrics": executor.get_metrics(),
        "elapsed_s": round(elapsed, 1),
        "error": err,
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vllm-url", default=os.environ.get("VLLM_URL", "http://localhost:8889/v1"))
    ap.add_argument("--model", default="openai/Qwen/Qwen3.5-9B",
                    help="LiteLLM-style model id. Use 'openai/<hf_name>' for vLLM.")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--top-p", type=float, default=None,
                    help="Qwen3.5 instruct: 0.8; reasoning: 1.0")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Qwen3.5 instruct: 20; reasoning: 40")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="Qwen3.5 instruct: 1.5; reasoning: 2.0 — reduces repetition loops")
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--description-mode", choices=["plain", "metadata", "reliability", "full"],
                    default="full")
    ap.add_argument("--granularity", choices=["collapsed", "split"], default="collapsed")
    ap.add_argument("--coverage-source", choices=["auto", "tracker", "index"], default="auto")
    ap.add_argument("--coverage-path", type=Path,
                    default=Path("data/runtime_coverage.json"))
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of questions to run in parallel (uses thread pool).")
    ap.add_argument("--module", choices=["react", "rlm"], default="react",
                    help="DSPy module to use as the agent.")
    ap.add_argument("--deno-path", type=str, default=None,
                    help="Path to deno binary for RLM PythonInterpreter. "
                         "If unset, RLM uses the default deno on PATH.")
    ap.add_argument("--enable-network", nargs="*", default=None,
                    help="Whitelist for sandbox network access (RLM mode).")
    ap.add_argument("--sub-vllm-url", default=None,
                    help="Optional separate vLLM URL for RLM's sub_lm. Defaults to --vllm-url.")
    ap.add_argument("--sub-model", default=None,
                    help="Optional separate model id for sub_lm.")
    ap.add_argument("--shard", default=None,
                    help="N/M format, e.g. 0/4 to take every 4th question starting at 0.")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    questions = [json.loads(l) for l in args.benchmark.open() if l.strip()]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        questions = [q for k, q in enumerate(questions) if k % n == i]
        print(f"[shard {args.shard}] {len(questions)} questions")
    if args.max_questions:
        questions = questions[: args.max_questions]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    already = set()
    if args.resume and args.output.exists():
        for line in args.output.open():
            try:
                already.add(json.loads(line).get("question_id"))
            except Exception:
                pass
        questions = [q for q in questions if q.get("id") not in already]
        print(f"[resume] skipping {len(already)}; {len(questions)} remaining")

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        if f.name.endswith(".bak"):
            continue
        try:
            indexes[f.stem] = json.loads(f.read_text())
        except Exception:
            pass

    tracker = CoverageTracker(path=args.coverage_path)

    extra_lm_kwargs = {}
    if args.top_p is not None: extra_lm_kwargs["top_p"] = args.top_p
    if args.top_k is not None: extra_lm_kwargs["top_k"] = args.top_k
    if args.presence_penalty is not None:
        extra_lm_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p is not None: extra_lm_kwargs["min_p"] = args.min_p

    lm = dspy.LM(
        model=args.model,
        api_base=args.vllm_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        **extra_lm_kwargs,
    )
    sub_lm = None
    if args.module == "rlm":
        sub_lm = dspy.LM(
            model=args.sub_model or args.model,
            api_base=args.sub_vllm_url or args.vllm_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            **extra_lm_kwargs,
        )

    deno_command = [args.deno_path] if args.deno_path else None

    n_done = n_correct = n_err = 0
    t_start = time.time()
    open_mode = "a" if (args.resume and already) else "w"

    def _process(q):
        row = run_one_question(
            q, indexes,
            description_mode=args.description_mode,
            granularity=args.granularity,
            coverage_source=args.coverage_source,
            tracker=tracker,
            max_turns=args.max_turns,
            lm=lm,
            module_kind=args.module,
            deno_command=deno_command,
            enable_network=args.enable_network,
            sub_lm=sub_lm,
        )
        row["module"] = args.module
        return row

    from concurrent.futures import ThreadPoolExecutor, as_completed
    fout_lock = None
    if args.concurrency > 1:
        import threading
        fout_lock = threading.Lock()

    with args.output.open(open_mode, buffering=1) as fout:
        if args.concurrency <= 1:
            for i, q in enumerate(questions, 1):
                row = _process(q)
                fout.write(json.dumps(row, default=str) + "\n")
                n_done += 1
                if row.get("inline_correct"): n_correct += 1
                if row.get("error"): n_err += 1
                if i % 10 == 0:
                    rate = i / max(1, time.time() - t_start)
                    print(f"  [{i}/{len(questions)}] correct={n_correct} err={n_err} "
                          f"({rate:.2f} q/s)")
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = {ex.submit(_process, q): q for q in questions}
                for fut in as_completed(futures):
                    row = fut.result()
                    with fout_lock:
                        fout.write(json.dumps(row, default=str) + "\n")
                        fout.flush()
                    n_done += 1
                    if row.get("inline_correct"): n_correct += 1
                    if row.get("error"): n_err += 1
                    if n_done % 5 == 0 or n_done == len(questions):
                        rate = n_done / max(1, time.time() - t_start)
                        print(f"  [{n_done}/{len(questions)}] correct={n_correct} "
                              f"err={n_err} ({rate:.2f} q/s)", flush=True)

    elapsed = time.time() - t_start
    print(f"\nDone. {n_done} processed, {n_correct} MC-correct, {n_err} errors. "
          f"{elapsed:.0f}s ({n_done/elapsed:.2f} q/s).")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
