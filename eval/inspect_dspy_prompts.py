"""Render exactly what DSPy is sending to the LM for one question.

Useful for understanding why prompts blow up / answers get truncated.
Saves to a markdown file with the prompt body for ReAct and RLM modes.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dspy
from eval.run_dspy_react_eval import VideoQA, format_options
from eval.v7_tools import V7Executor
from eval.dspy_v7_tools import make_dspy_tools, CoverageTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    ap.add_argument("--vllm-url", default="http://localhost:8890/v1")
    ap.add_argument("--model", default="openai/Qwen/Qwen3.5-9B")
    ap.add_argument("--question-index", type=int, default=0)
    ap.add_argument("--description-mode", default="full")
    ap.add_argument("--granularity", default="collapsed")
    ap.add_argument("--module", choices=["react", "rlm"], default="react")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    qs = [json.loads(l) for l in args.benchmark.open() if l.strip()]
    q = qs[args.question_index]
    video_id = q["video_id"]
    idx = json.loads((args.index_dir / f"{video_id}.json").read_text())

    executor = V7Executor(idx, video_id=video_id)
    tools = make_dspy_tools(executor, mode=args.description_mode,
                            granularity=args.granularity, video_id=video_id,
                            tracker=CoverageTracker(path=Path("/tmp/_inspect_cov.json")))

    lm = dspy.LM(model=args.model, api_base=args.vllm_url, api_key="EMPTY",
                 temperature=0.0, max_tokens=10)  # tiny so we don't actually wait

    if args.module == "rlm":
        agent = dspy.RLM(VideoQA, tools=tools, max_iterations=3,
                         max_llm_calls=5, sub_lm=lm)
    else:
        agent = dspy.ReAct(VideoQA, tools=tools, max_iters=3)

    duration = (idx.get("duration_ms") or 0) / 1000.0
    video_context = f"video_id={video_id}; duration={duration:.0f}s"

    # Try to call (will likely fail given max_tokens=10) — but it captures the prompt
    try:
        with dspy.context(lm=lm):
            agent(question=q["question"], options=format_options(q),
                  video_context=video_context)
    except Exception as e:
        print(f"(expected truncation/parse error: {type(e).__name__})", file=sys.stderr)

    # Pull from history
    hist = lm.history if hasattr(lm, "history") else []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# DSPy prompt inspection\n\n")
        f.write(f"- module: `{args.module}`\n- description-mode: `{args.description_mode}`"
                f"\n- granularity: `{args.granularity}`\n- video: `{video_id}`\n")
        f.write(f"- question: {q['question']}\n\n")
        f.write(f"- num LM calls captured: {len(hist)}\n\n")
        for i, h in enumerate(hist):
            f.write(f"\n## Call {i+1}\n\n")
            messages = h.get("messages") or []
            for m in messages:
                role = m.get("role", "?")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(c.get("text", str(c)) for c in content)
                f.write(f"\n### `{role}` ({len(content)} chars)\n\n")
                f.write("```\n" + content + "\n```\n")

    n_total = sum(len(m.get("content","")) for h in hist for m in (h.get("messages") or []))
    print(f"Wrote {args.out}")
    print(f"  {len(hist)} call(s), total {n_total} chars in messages")


if __name__ == "__main__":
    main()
