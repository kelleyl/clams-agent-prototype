"""Run one RLM question and dump the model's actual reasoning + code at each iteration.

Captures the full LM call history (prompts + completions) so we can see what
the model is producing.
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
    ap.add_argument("--max-iters", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    qs = [json.loads(l) for l in args.benchmark.open() if l.strip()]
    # Apply same shard logic as the smoke
    qs_shard = [q for k, q in enumerate(qs) if k % 73 == 0]
    q = qs_shard[args.question_index]
    video_id = q["video_id"]
    idx = json.loads((args.index_dir / f"{video_id}.json").read_text())

    executor = V7Executor(idx, video_id=video_id)
    tools = make_dspy_tools(executor, mode="full",
                            granularity="collapsed", video_id=video_id,
                            tracker=CoverageTracker(path=Path("/tmp/_rlm_inspect_cov.json")))

    lm = dspy.LM(model=args.model, api_base=args.vllm_url, api_key="EMPTY",
                 temperature=args.temperature, max_tokens=args.max_tokens,
                 top_p=1.0, top_k=40, presence_penalty=2.0)

    agent = dspy.RLM(VideoQA, tools=tools, max_iterations=args.max_iters,
                     max_llm_calls=args.max_iters * 2, max_output_chars=10000,
                     sub_lm=lm)

    duration = (idx.get("duration_ms") or 0) / 1000.0
    video_context = f"video_id={video_id}; duration={duration:.0f}s"

    print(f"Q: {q['question']}", file=sys.stderr)
    print(f"Expected: {q.get('mc_correct')}", file=sys.stderr)

    pred = None
    err = None
    try:
        with dspy.context(lm=lm):
            pred = agent(question=q["question"], options=format_options(q),
                         video_context=video_context)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"Error: {err}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# RLM trace inspection\n\n")
        f.write(f"- video: `{video_id}`\n")
        f.write(f"- question: {q['question']}\n")
        f.write(f"- options:\n```\n{format_options(q)}\n```\n")
        f.write(f"- expected: {q.get('mc_correct')}\n")
        f.write(f"- predicted: {getattr(pred, 'answer', None)}\n")
        f.write(f"- error: {err}\n\n")

        hist = lm.history if hasattr(lm, "history") else []
        f.write(f"## {len(hist)} LM calls\n\n")
        for i, h in enumerate(hist):
            f.write(f"\n## Call {i+1}\n\n")
            messages = h.get("messages") or []
            for m in messages[-2:]:  # only last system + user (skip earlier examples)
                role = m.get("role", "?")
                content = m.get("content", "")
                if isinstance(content, list):
                    content = "\n".join(c.get("text", str(c)) for c in content)
                f.write(f"\n### `{role}` prompt ({len(content)} chars)\n\n")
                f.write("```\n" + content[-3000:] + "\n```\n")  # last 3K chars
            # Completion
            completion = h.get("outputs", h.get("response", ""))
            if isinstance(completion, list):
                completion = "\n".join(str(c) for c in completion)
            elif isinstance(completion, dict):
                # Try to extract text
                completion = str(completion)
            f.write(f"\n### model completion ({len(str(completion))} chars)\n\n")
            f.write("```\n" + str(completion)[:5000] + "\n```\n")

    print(f"\nWrote {args.out}", file=sys.stderr)
    print(f"  {len(hist)} LM calls", file=sys.stderr)


if __name__ == "__main__":
    main()
