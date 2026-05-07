"""Wrap V7Executor's warm-index tools as DSPy Tools with metadata-rich
descriptions sourced from:

  1. CLAMS App Directory (data/app_directory.json) — what app the tool
     reads from, and what CLAMS-declared inputs/outputs it has
  2. tool_stats.json — empirical reliability from prior eval runs
     (recall@modality, correctness when called, frac videos with coverage)
  3. Per-video coverage map — runtime hint about whether THIS video has
     non-empty layer data for the tool

The descriptions are templated so we can A/B:

  - 'plain': hand-written description (current baseline)
  - 'metadata': adds CLAMS app provenance
  - 'reliability': adds empirical stats
  - 'full': metadata + reliability + per-video hint

Granularity:

  - 'collapsed' (default): one tool per task, model variants in description
  - 'split': one dspy.Tool per (task, model) — better for GEPA tool selection

Returns: list[dspy.Tool] ready to drop into a dspy.ReAct module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP_DIR = REPO_ROOT / "data" / "app_directory.json"
DEFAULT_STATS = REPO_ROOT / "data" / "tool_stats.json"
DEFAULT_RUNTIME_COVERAGE = REPO_ROOT / "data" / "runtime_coverage.json"

# Hard cap on each tool observation handed to the LLM so the ReAct
# trajectory doesn't blow past the model's context window. V7 tools
# (especially OCR / visual_description) often return >2K-char captions.
DEFAULT_MAX_OBS_CHARS = 1200


# A token used in observations so the agent can see coverage updates
# in its own ReAct history (no need to re-render tool descriptions).
COVERAGE_NOTE_PREFIX = "[coverage]"


class CoverageTracker:
    """Tracks per-video tool coverage as tools are actually called.

    Two intended uses:
      1. New video / new collection: start with empty coverage; populate
         as tools return non-empty data. Future questions on the same
         video get richer hints in tool descriptions.
      2. Existing corpus with prebuilt indexes: optional warm-start from
         the index layers (the static signal we already have).

    Persists to a JSON file so coverage learned in one ReAct trajectory
    carries forward to later ones on the same video.
    """

    EMPTY_MARKERS = (
        "no ocr data", "no text detected", "no transcript", "no speech",
        "no speakers", "no visual data", "empty", "(no output)",
        "no caption", "text scenes detected but no ocr",
        "no on-screen text", "no on screen text", "no text on screen",
        "no shots detected", "no scenes detected", "no caption found",
        "no description available", "nothing detected", "no asr",
        "no diarization data", "no results", "no output",
    )

    def __init__(self, path: Path = DEFAULT_RUNTIME_COVERAGE,
                 readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._data = {}
        if path and path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                self._data = {}

    def get_video(self, video_id: str) -> dict:
        return self._data.get(video_id, {})

    def warm_start_from_index(self, video_id: str,
                              per_video_coverage: dict) -> None:
        """Seed runtime coverage from a prebuilt static coverage map.

        Uses 'inferred' rather than 'observed' as the source so we can tell
        what came from a real call vs an offline index check.
        """
        existing = self._data.setdefault(video_id, {})
        for tool, layers in per_video_coverage.items():
            if tool in existing:
                continue
            existing[tool] = {
                "non_empty_calls": 0,
                "empty_calls": 0,
                "any_data": bool(layers),
                "source": "index_warm_start",
                "layers": list(layers) if layers else [],
            }
        self._save()

    def record_call(self, video_id: str, tool: str, observation: str) -> str:
        """Record one tool call. Returns an inline coverage note for the agent.

        The note is a short string appended to the observation, e.g.
        '[coverage] get_text_on_screen returned empty for this video.'
        """
        rec = self._data.setdefault(video_id, {}).setdefault(tool, {
            "non_empty_calls": 0,
            "empty_calls": 0,
            "any_data": None,
            "source": "observed",
            "layers": [],
        })
        is_empty = self._is_empty_observation(observation)
        if is_empty:
            rec["empty_calls"] += 1
        else:
            rec["non_empty_calls"] += 1
            rec["any_data"] = True
            rec["source"] = "observed"
        self._save()
        return self._build_note(tool, rec, is_empty)

    def _is_empty_observation(self, observation: str) -> bool:
        if not observation:
            return True
        s = observation.strip().lower()
        if not s:
            return True
        if len(s) < 20:
            return True
        return any(m in s for m in self.EMPTY_MARKERS)

    def _build_note(self, tool: str, rec: dict, was_empty: bool) -> str:
        n_e = rec["empty_calls"]
        n_ne = rec["non_empty_calls"]
        if was_empty and n_ne == 0:
            return (f"{COVERAGE_NOTE_PREFIX} {tool} has returned empty "
                    f"{n_e}/{n_e+n_ne} times on this video so far. "
                    "Consider an alternative tool.")
        if was_empty and n_ne > 0:
            return (f"{COVERAGE_NOTE_PREFIX} {tool} returned empty for "
                    f"this range, but has produced data elsewhere on this "
                    "video.")
        return ""  # non-empty + first useful return: no note needed

    def coverage_for_descriptions(self, video_id: str) -> dict:
        """Map tool → list of layers/data sources, formatted for description rendering."""
        out = {}
        for tool, rec in self.get_video(video_id).items():
            if rec.get("any_data"):
                if rec.get("layers"):
                    out[tool] = rec["layers"]
                else:
                    out[tool] = [f"observed ({rec.get('non_empty_calls',0)} non-empty calls)"]
            else:
                out[tool] = []
        return out

    def _save(self):
        if self.readonly or not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))


# Map our warm-index tool → CLAMS app(s) it reads from. Some tools span
# multiple apps (e.g. text_on_screen depends on SWT + OCR captioner).
TOOL_TO_CLAMS_APPS = {
    "get_transcript": ["whisper-wrapper"],  # we also have parakeet variant
    "get_text_on_screen": ["swt-detection", "qwen3vl-captioner"],
    "get_visual_description": ["qwen3vl-captioner"],
    "get_speakers": ["app-speaker-diarization"],  # not in public directory
    "get_video_info": [],
}

# Per-tool model variants and short reliability priors (rough defaults
# until per-variant stats are computed).
MODEL_VARIANTS = {
    "get_text_on_screen": [
        ("qwen3vl-8b", {"speed": "medium", "cost_per_min": 0.15, "quality": "good"}),
        ("qwen-30b", {"speed": "slow", "cost_per_min": 0.35, "quality": "best"}),
        ("qwen-8b", {"speed": "medium", "cost_per_min": 0.15, "quality": "good"}),
        ("qwen-small", {"speed": "fast", "cost_per_min": 0.05, "quality": "ok"}),
        ("smolvlm", {"speed": "fast", "cost_per_min": 0.05, "quality": "weak"}),
    ],
    "get_visual_description": [
        ("qwen3vl-8b", {"speed": "medium", "cost_per_min": 0.15, "quality": "good"}),
        ("qwen-30b", {"speed": "slow", "cost_per_min": 0.35, "quality": "best"}),
        ("qwen-small", {"speed": "fast", "cost_per_min": 0.05, "quality": "ok"}),
        ("smolvlm", {"speed": "fast", "cost_per_min": 0.05, "quality": "weak"}),
    ],
    "get_transcript": [
        ("parakeet", {"speed": "fast", "cost_per_min": 0.01, "quality": "ok"}),
        ("whisper", {"speed": "slow", "cost_per_min": 0.10, "quality": "best"}),
    ],
}


@dataclass
class ToolDescriptor:
    name: str
    base_desc: str
    clams_apps: list[str] = field(default_factory=list)
    clams_metadata: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    per_video_layers: list[str] = field(default_factory=list)
    model_variant: Optional[str] = None
    variant_meta: dict = field(default_factory=dict)


def _load_app_directory(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_tool_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _build_description(d: ToolDescriptor, mode: str) -> str:
    """Render the tool description according to A/B mode."""
    parts = [d.base_desc]

    if mode in ("metadata", "full") and d.clams_apps:
        app_blurbs = []
        for app in d.clams_apps:
            md = (d.clams_metadata.get(app) or {}).get("metadata", {})
            desc = md.get("description") if isinstance(md, dict) else None
            ver = (d.clams_metadata.get(app) or {}).get("latest_version", "")
            if desc:
                app_blurbs.append(f"{app} ({ver}): {desc}")
            else:
                app_blurbs.append(f"{app}")
        parts.append("Backed by: " + "; ".join(app_blurbs))

    if mode in ("reliability", "full") and d.stats:
        s = d.stats
        bits = []
        rel = s.get("modality_recall")
        if rel is not None:
            bits.append(f"recall@modality={rel:.0%}")
        corr = s.get("correctness_when_called")
        if corr is not None:
            bits.append(f"answer-correct-when-called={corr:.0%}")
        used = s.get("frac_questions_using")
        if used is not None:
            bits.append(f"used-on={used:.0%}-of-questions")
        if bits:
            parts.append("Empirical (prior runs): " + ", ".join(bits))

    # Per-video coverage is only meaningful for tools that read a layer.
    is_layer_tool = d.name in {"get_transcript", "get_text_on_screen",
                                "get_visual_description", "get_speakers"}
    if mode == "full" and is_layer_tool:
        if d.per_video_layers:
            parts.append(f"This video has data: {', '.join(d.per_video_layers)}.")
        else:
            parts.append("WARNING: this video has no indexed layer for this tool — "
                         "expect empty output.")

    if d.variant_meta:
        v = d.variant_meta
        parts.append(
            f"Model variant '{d.model_variant}': speed={v.get('speed')}, "
            f"cost~${v.get('cost_per_min')}/min, quality={v.get('quality')}."
        )

    return "\n".join(parts)


def _make_dspy_tool(executor, tool_name: str, base_desc: str,
                    descriptor: ToolDescriptor, mode: str,
                    fixed_model: Optional[str] = None,
                    tracker: Optional["CoverageTracker"] = None,
                    video_id: Optional[str] = None):
    """Wrap V7Executor.execute(...) as a dspy.Tool."""
    import dspy

    description = _build_description(descriptor, mode)
    dspy_name = descriptor.name  # may include __variant suffix in split mode

    def _truncate(s: str, n: int = DEFAULT_MAX_OBS_CHARS) -> str:
        if not s or len(s) <= n:
            return s
        return s[:n] + f"\n…[truncated, {len(s)-n} more chars]"

    def _wrap_obs(obs: str) -> str:
        obs = _truncate(obs)
        if not (tracker and video_id):
            return obs
        note = tracker.record_call(video_id, tool_name, obs)
        return f"{obs}\n{note}".strip() if note else obs

    def _runner(start_time: str, end_time: str,
                model: Optional[str] = None,
                task_mode: Optional[str] = None) -> str:
        params = {"start_time": start_time, "end_time": end_time}
        if fixed_model:
            params["model"] = fixed_model
        elif model:
            params["model"] = model
        if task_mode and tool_name == "get_visual_description":
            params["task_mode"] = task_mode
        result = executor.execute(tool_name, params)
        obs = result.observation if hasattr(result, "observation") else str(result)
        return _wrap_obs(obs)

    def _runner_no_args() -> str:
        result = executor.execute(tool_name, {})
        obs = result.observation if hasattr(result, "observation") else str(result)
        return _wrap_obs(obs)

    if tool_name == "get_video_info":
        return dspy.Tool(_runner_no_args, name=dspy_name, desc=description)
    if tool_name == "get_speakers":
        def _sp(start_time: str, end_time: str) -> str:
            result = executor.execute(tool_name,
                                      {"start_time": start_time, "end_time": end_time})
            obs = result.observation if hasattr(result, "observation") else str(result)
            return _wrap_obs(obs)
        return dspy.Tool(_sp, name=dspy_name, desc=description)
    return dspy.Tool(_runner, name=dspy_name, desc=description)


def make_dspy_tools(executor,
                    *,
                    mode: str = "full",
                    granularity: str = "collapsed",
                    app_dir_path: Path = DEFAULT_APP_DIR,
                    stats_path: Path = DEFAULT_STATS,
                    video_id: Optional[str] = None,
                    tracker: Optional[CoverageTracker] = None,
                    coverage_source: str = "auto") -> list:
    """Build dspy.Tool list from a V7Executor.

    Args:
        executor: V7Executor instance bound to a video_id + index_data
        mode: 'plain' | 'metadata' | 'reliability' | 'full'
        granularity: 'collapsed' (1 tool per task) | 'split' (1 per task,model)
        app_dir_path: path to data/app_directory.json
        stats_path: path to data/tool_stats.json
        video_id: if given, the per-video coverage hint is included
        tracker: CoverageTracker that updates as tools are called. When
            present, observations get an inline coverage note appended.
        coverage_source: 'auto' (prefer tracker observed, fall back to index),
            'tracker' (only runtime observations — required for new
            videos / new collections), 'index' (only static index check)
    """
    from eval.v7_tools import V7_TOOL_SCHEMAS

    app_dir = _load_app_directory(app_dir_path)
    stats_blob = _load_tool_stats(stats_path)
    per_tool_stats = (stats_blob.get("aggregate") or {}).get("per_tool", {})

    # Resolve which coverage signal to render in descriptions
    index_cov = (stats_blob.get("per_video_coverage") or {}).get(video_id, {}) if video_id else {}
    tracker_cov = tracker.coverage_for_descriptions(video_id) if (tracker and video_id) else {}
    if coverage_source == "tracker":
        per_video_cov = tracker_cov
    elif coverage_source == "index":
        per_video_cov = index_cov
    else:
        per_video_cov = {}
        for t in set(list(index_cov.keys()) + list(tracker_cov.keys())):
            per_video_cov[t] = tracker_cov.get(t) or index_cov.get(t) or []

    tools = []
    for schema in V7_TOOL_SCHEMAS:
        fn = schema["function"]
        tool_name = fn["name"]
        base_desc = fn["description"]
        clams_apps = TOOL_TO_CLAMS_APPS.get(tool_name, [])
        clams_meta = {a: app_dir.get(a, {}) for a in clams_apps}
        stats = per_tool_stats.get(tool_name, {})
        per_video_layers = per_video_cov.get(tool_name, [])

        if granularity == "split" and tool_name in MODEL_VARIANTS:
            # one dspy.Tool per (task, model)
            for variant_model, variant_meta in MODEL_VARIANTS[tool_name]:
                desc = ToolDescriptor(
                    name=f"{tool_name}__{variant_model}",
                    base_desc=base_desc,
                    clams_apps=clams_apps,
                    clams_metadata=clams_meta,
                    stats=stats,
                    per_video_layers=per_video_layers,
                    model_variant=variant_model,
                    variant_meta=variant_meta,
                )
                tools.append(_make_dspy_tool(executor, tool_name, base_desc,
                                             desc, mode, fixed_model=variant_model,
                                             tracker=tracker, video_id=video_id))
        else:
            desc = ToolDescriptor(
                name=tool_name,
                base_desc=base_desc,
                clams_apps=clams_apps,
                clams_metadata=clams_meta,
                stats=stats,
                per_video_layers=per_video_layers,
            )
            tools.append(_make_dspy_tool(executor, tool_name, base_desc, desc, mode,
                                         tracker=tracker, video_id=video_id))

    return tools


def render_descriptions_only(*, mode: str = "full",
                             video_id: Optional[str] = None,
                             app_dir_path: Path = DEFAULT_APP_DIR,
                             stats_path: Path = DEFAULT_STATS) -> dict:
    """Inspect what the descriptions look like without building dspy.Tools.
    Useful for prompt-comparison tables and CI-style diffs.
    """
    from eval.v7_tools import V7_TOOL_SCHEMAS

    app_dir = _load_app_directory(app_dir_path)
    stats_blob = _load_tool_stats(stats_path)
    per_tool_stats = (stats_blob.get("aggregate") or {}).get("per_tool", {})
    per_video_cov = (stats_blob.get("per_video_coverage") or {}).get(video_id, {}) if video_id else {}

    out = {}
    for schema in V7_TOOL_SCHEMAS:
        fn = schema["function"]
        tool_name = fn["name"]
        clams_apps = TOOL_TO_CLAMS_APPS.get(tool_name, [])
        desc = ToolDescriptor(
            name=tool_name,
            base_desc=fn["description"],
            clams_apps=clams_apps,
            clams_metadata={a: app_dir.get(a, {}) for a in clams_apps},
            stats=per_tool_stats.get(tool_name, {}),
            per_video_layers=per_video_cov.get(tool_name, []),
        )
        out[tool_name] = _build_description(desc, mode)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full",
                    choices=["plain", "metadata", "reliability", "full"])
    ap.add_argument("--video-id", default=None)
    args = ap.parse_args()

    descs = render_descriptions_only(mode=args.mode, video_id=args.video_id)
    for name, d in descs.items():
        print(f"\n=== {name} ===")
        print(d)
