"""Compute per-tool empirical reliability stats from prior eval runs.

Produces structured stats that can be embedded into DSPy tool descriptions
or rendered as a runtime "tool reliability sheet" for the agent.

Stats produced per (tool, optional model):
  - call_freq: how often this tool was used per question
  - call_when_modality_needed: recall against modalities_required
  - correctness_when_called: P(MC correct | tool was called)
  - mean_cost_per_call: from v7_metrics processed_duration
  - mean_evidence_items_per_call: signal of usefulness

Per-video coverage stats (separate, from video_indexes):
  - has_layer: does the index contain non-empty data for the tool's layer?

Usage:
    python eval/tool_stats.py \\
        --results eval/results/v5_1_base_policy_*.jsonl \\
        --benchmark qa-data/benchmark/v5_1/v5_1_benchmark.jsonl \\
        --output data/tool_stats.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


# Map tool → primary modality(s) the question must require for the tool to be "needed".
TOOL_TO_MODALITIES = {
    "get_transcript": {"speech", "asr"},
    "get_text_on_screen": {"ocr", "visual_text"},
    "get_visual_description": {"visual", "vlm"},
    "get_speakers": {"speaker", "diarization"},
    "get_video_info": set(),  # always free, never "needed"
}

# Map tool → index layer key prefix (for per-video coverage check).
TOOL_TO_LAYER = {
    "get_transcript": ["asr", "asr_whisper"],
    "get_text_on_screen": ["caption_qwen3vl-8b_text_focus", "caption_qwen-8b_text_focus",
                           "caption_qwen-30b_text_focus", "caption_smolvlm_text_focus"],
    "get_visual_description": ["caption_qwen3vl-8b_general_scene", "caption_qwen-8b_general_scene",
                                "caption_qwen-30b_general_scene"],
    "get_speakers": ["speakers"],
    "get_video_info": [],
}


def load_benchmark(path: Path) -> dict:
    by_id = {}
    if not path or not path.exists():
        return by_id
    for line in path.open():
        if not line.strip():
            continue
        r = json.loads(line)
        qid = r.get("id") or r.get("question_id")
        if qid:
            by_id[qid] = r
    return by_id


def aggregate_tool_stats(result_files: list[Path], benchmark: dict) -> dict:
    """Aggregate per-tool stats across one or more result files."""
    n_questions = 0
    n_correct = 0
    tool_calls = defaultdict(int)
    tool_questions = defaultdict(set)  # tool → set of qids that called it
    tool_correct_calls = defaultdict(int)  # tool was called AND answer correct
    tool_modality_recall_num = defaultdict(int)  # called when modality required
    tool_modality_recall_den = defaultdict(int)  # modality required (regardless of call)
    cost_by_modality = defaultdict(list)

    for fpath in result_files:
        if not fpath.exists():
            continue
        for line in fpath.open():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_questions += 1
            qid = r.get("question_id")
            correct = bool(r.get("inline_correct"))
            if correct:
                n_correct += 1

            mods_needed = set(r.get("modalities_required") or [])
            tools_used = list(r.get("tool_calls") or [])
            unique_tools = set(tools_used)

            for tname, tmods in TOOL_TO_MODALITIES.items():
                if tmods and (tmods & mods_needed):
                    tool_modality_recall_den[tname] += 1
                    if tname in unique_tools:
                        tool_modality_recall_num[tname] += 1

            for t in tools_used:
                tool_calls[t] += 1
                tool_questions[t].add(qid)
            for t in unique_tools:
                if correct:
                    tool_correct_calls[t] += 1

            v7m = r.get("v7_metrics") or {}
            for mod, dur_ms in (v7m.get("processed_duration_ms") or {}).items():
                cost_by_modality[mod].append(dur_ms / 60000.0)

    out = {}
    for tname in TOOL_TO_MODALITIES:
        n_used = len(tool_questions[tname])
        out[tname] = {
            "call_freq_per_question": round(tool_calls[tname] / max(1, n_questions), 3),
            "questions_using_tool": n_used,
            "frac_questions_using": round(n_used / max(1, n_questions), 3),
            "correctness_when_called": round(
                tool_correct_calls[tname] / max(1, n_used), 3
            ),
            "modality_recall": (
                round(tool_modality_recall_num[tname] / max(1, tool_modality_recall_den[tname]), 3)
                if tool_modality_recall_den[tname] else None
            ),
            "n_modality_relevant_qs": tool_modality_recall_den[tname],
        }

    return {
        "n_questions": n_questions,
        "overall_correct": round(n_correct / max(1, n_questions), 3),
        "n_runs": len(result_files),
        "per_tool": out,
    }


def compute_per_video_coverage(index_dir: Path) -> dict:
    """For each video index, record which tool-relevant layers have content."""
    coverage = {}
    for f in sorted(index_dir.glob("*.json")):
        if f.name.endswith(".bak"):
            continue
        try:
            idx = json.loads(f.read_text())
        except Exception:
            continue
        vid = idx.get("video_id") or f.stem
        layers = idx.get("layers", {}) or {}
        per_video = {}
        for tool, layer_keys in TOOL_TO_LAYER.items():
            available = []
            for lk in layer_keys:
                lyr = layers.get(lk)
                if not lyr:
                    continue
                if isinstance(lyr, dict):
                    items = lyr.get("items") or []
                else:
                    items = lyr if isinstance(lyr, list) else []
                if items:
                    available.append(lk)
            per_video[tool] = available
        coverage[vid] = per_video
    return coverage


def coverage_summary(coverage: dict) -> dict:
    """Aggregate per-video coverage to a tool → fraction-of-videos-covered map."""
    n_videos = len(coverage)
    summary = {}
    for tool in TOOL_TO_LAYER:
        n_have = sum(1 for v in coverage.values() if v.get(tool))
        summary[tool] = {
            "n_videos_with_layer": n_have,
            "frac_videos_with_layer": round(n_have / max(1, n_videos), 3),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="*", default=[],
                    help="Glob patterns or paths for result jsonl files")
    ap.add_argument("--benchmark", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path,
                    default=Path("data/video_indexes"))
    ap.add_argument("--output", type=Path,
                    default=Path("data/tool_stats.json"))
    args = ap.parse_args()

    files = []
    for p in args.results:
        if "*" in p:
            files.extend(Path(x) for x in glob.glob(p))
        else:
            files.append(Path(p))

    benchmark = load_benchmark(args.benchmark) if args.benchmark else {}
    agg = aggregate_tool_stats(files, benchmark) if files else {
        "n_questions": 0, "per_tool": {}, "n_runs": 0,
    }

    cov = compute_per_video_coverage(args.index_dir) if args.index_dir.exists() else {}
    cov_sum = coverage_summary(cov) if cov else {}

    out = {
        "aggregate": agg,
        "coverage_summary": cov_sum,
        "per_video_coverage": cov,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    print(f"Wrote {args.output}")
    print(f"  questions analyzed: {agg.get('n_questions', 0)}")
    print(f"  per-tool aggregate stats:")
    for t, s in (agg.get("per_tool") or {}).items():
        print(f"    {t:30s}  used={s['frac_questions_using']:.0%}  "
              f"recall@modality={s['modality_recall']}  "
              f"correctness={s['correctness_when_called']:.0%}")
    if cov_sum:
        print(f"  per-video coverage (n={len(cov)}):")
        for t, s in cov_sum.items():
            print(f"    {t:30s}  {s['n_videos_with_layer']}/{len(cov)} videos")


if __name__ == "__main__":
    main()
