#!/usr/bin/env python3
"""Eval harness using native RLM over actual MMIF files.

This is the CLAMS RLM scaffold we want to train against: the model gets a
Python REPL with MMIF-backed helpers and can inspect the actual MMIF output for
the video under question. It intentionally does not load or depend on derived
video indexes.

Usage:
    python eval/run_rlm_native_eval.py \\
        --benchmark qa-data/benchmark/v5_2/benchmark_combined.jsonl \\
        --output eval/results/v5_2_rlm_native_mmif_smoke.jsonl \\
        --vllm-url http://localhost:8891/v1 \\
        --model mit-oasys/rlm-qwen3-8b-v0.1 \\
        --shard 0/73 --max-questions 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clams_haystack.mmif_inventory import MmifInventory
from utils.rlm_sandbox import RLMVideoWorkspace

try:
    from rlm import RLM
except ModuleNotFoundError:
    RLM = None


PROMPT_TEMPLATE = """You are answering a question about an archival video using the actual CLAMS MMIF file.

The REPL has MMIF-backed custom tools. The most important helpers are:
{cheat_sheet}

VIDEO_ID: {video_id}
MMIF_PATH: {mmif_path}
VIDEO_METADATA:
{video_info}

VIEW SUMMARY:
{view_summary}

Initial context blocks below are rendered from the same MMIF file. Use the REPL
helpers to inspect raw views, annotations, alignments, and linked text when the
question needs more detail.

=== ASR / TRANSCRIPT TEXT ===
{asr}

=== VISUAL DESCRIPTIONS ===
{visual}

=== ON-SCREEN TEXT / OCR-LIKE TEXT ===
{ocr}

=== SPEAKER ANNOTATIONS ===
{speakers}

=== QUESTION ===
{question}

=== OPTIONS ===
{options}

Use Python to navigate the MMIF structure, search text, inspect timeframes, and
walk Alignment edges as needed. When done, call FINAL(letter) where letter is A,
B, C, or D."""


def format_options(q: dict) -> str:
    opts = q.get("mc_options") or {}
    if not opts:
        return "(free-text)"
    return "\n".join(f"{k}. {v}" for k, v in sorted(opts.items()))


def extract_letter(text: str) -> str:
    import re

    if not text:
        return ""
    match = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.I)
    if match:
        return match.group(1).upper()
    upper = text.strip().upper()
    match = re.search(r"\bANSWER\s*[:=]\s*([ABCD])\b", upper)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[\s\(\[])([ABCD])(?:[\.\)\]\:]|$)", upper)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", upper)
    if match:
        return match.group(1)
    return upper[:1] if upper and upper[0] in "ABCD" else ""


def build_prompt(workspace: RLMVideoWorkspace, q: dict) -> str:
    blocks = workspace.context_blocks()
    view_summary = json.dumps(workspace.views(), indent=2, default=str)
    video_info = json.dumps(workspace.video_info(), indent=2, default=str)
    return PROMPT_TEMPLATE.format(
        video_id=workspace.video_id,
        mmif_path=str(workspace.path),
        video_info=video_info,
        view_summary=view_summary,
        question=q["question"],
        options=format_options(q),
        cheat_sheet=workspace.cheat_sheet(),
        **blocks,
    )


def run_one_question(
    q: dict,
    mmif_paths: dict[str, str],
    *,
    backend_kwargs: dict,
    max_iterations: int,
    max_tokens: int,
    verbose: bool = False,
) -> dict:
    qid = q.get("id") or q.get("question_id")
    video_id = q.get("video_id")
    mmif_path = mmif_paths.get(video_id)
    if not mmif_path:
        return {
            "question_id": qid,
            "video_id": video_id,
            "error": f"no MMIF file for {video_id}",
            "module": "rlm_native_mmif",
        }
    if RLM is None:
        return {
            "question_id": qid,
            "video_id": video_id,
            "error": "native rlm package is not installed",
            "module": "rlm_native_mmif",
        }

    workspace = RLMVideoWorkspace.from_path(mmif_path, video_id=video_id)
    tools = workspace.custom_tools()
    prompt = build_prompt(workspace, q)

    rlm = RLM(
        backend="vllm",
        backend_kwargs=backend_kwargs,
        environment="local",
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        custom_tools=tools,
        compaction=True,
        compaction_threshold_pct=0.85,
        verbose=verbose,
    )

    t0 = time.time()
    raw = ""
    err = None
    try:
        result = rlm.completion(prompt)
        raw = result.response if hasattr(result, "response") else str(result)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        try:
            rlm.close()
        except Exception:
            pass

    elapsed = time.time() - t0
    fmt = q.get("format", "multiple_choice")
    expected = str(q.get("mc_correct") or q.get("expected_answer") or "")
    if fmt == "multiple_choice":
        predicted = extract_letter(raw)
        is_correct = predicted == expected.strip().upper()[:1]
    else:
        predicted = raw.strip()[:200]
        is_correct = False

    evidence = workspace.evidence()
    tool_calls = [item.get("tool", "") for item in evidence]
    return {
        "question_id": qid,
        "video_id": video_id,
        "question": q["question"],
        "format": fmt,
        "expected_answer": expected,
        "predicted_answer": predicted,
        "raw_answer": str(raw)[:1000],
        "agent_mode": "rlm_native_mmif",
        "tool_calls": tool_calls,
        "tool_trace": evidence,
        "evidence": [
            {
                "tool": item.get("tool"),
                "time_range": item.get("time_range"),
                "observation": (item.get("observation") or "")[:500],
            }
            for item in evidence
        ],
        "num_tool_calls": len(tool_calls),
        "evidence_count": len(evidence),
        "modalities_required": q.get("modalities_required", []),
        "source_segment_times": q.get("source_segment_times", []),
        "inline_correct": is_correct,
        "needs_posthoc_scoring": fmt != "multiple_choice",
        "mmif_metrics": {
            "view_count": len(workspace.views()),
            "document_count": len(workspace.documents),
            "mmif_path": mmif_path,
        },
        "elapsed_s": round(elapsed, 1),
        "error": err,
        "module": "rlm_native_mmif",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--mmif-roots", default=None, help="MMIF root path(s), comma- or pathsep-separated")
    parser.add_argument("--index-dir", type=Path, default=None, help="Deprecated; ignored. This harness uses MMIF files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm-url", default="http://localhost:8891/v1")
    parser.add_argument("--model", default="mit-oasys/rlm-qwen3-8b-v0.1")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--shard", default=None, help="N/M format")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.benchmark.open() if line.strip()]
    if args.shard:
        shard_idx, shard_count = (int(x) for x in args.shard.split("/"))
        questions = [q for k, q in enumerate(questions) if k % shard_count == shard_idx]
        print(f"[shard {args.shard}] {len(questions)} questions", flush=True)
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
        print(f"[resume] skipping {len(already)}; {len(questions)} remaining", flush=True)

    inventory = MmifInventory(args.mmif_roots)
    mmif_paths = {
        video.video_id: video.mmif_path
        for video in inventory.list_videos()
        if video.mmif_path
    }
    print(f"[mmif] {len(mmif_paths)} videos from {inventory.roots_display()}", flush=True)

    backend_kwargs = {
        "base_url": args.vllm_url,
        "model_name": args.model,
        "api_key": "EMPTY",
    }

    n_done = n_correct = n_err = 0
    t_start = time.time()
    open_mode = "a" if (args.resume and already) else "w"
    with args.output.open(open_mode, buffering=1) as fout:
        for i, q in enumerate(questions, 1):
            row = run_one_question(
                q,
                mmif_paths,
                backend_kwargs=backend_kwargs,
                max_iterations=args.max_iterations,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            fout.write(json.dumps(row, default=str) + "\n")
            n_done += 1
            if row.get("inline_correct"):
                n_correct += 1
            if row.get("error"):
                n_err += 1
            if i % 5 == 0 or i == len(questions):
                rate = i / max(1, time.time() - t_start)
                print(
                    f"  [{i}/{len(questions)}] correct={n_correct} err={n_err} "
                    f"({rate:.2f} q/s)",
                    flush=True,
                )

    elapsed = time.time() - t_start
    print(
        f"\nDone. {n_done} processed, {n_correct} MC-correct, {n_err} errors. "
        f"{elapsed:.0f}s ({n_done / max(1, elapsed):.2f} q/s)."
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
