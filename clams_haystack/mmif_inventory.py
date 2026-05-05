"""Read-only inventory over MMIF outputs for the Chainlit tool palette.

The thesis outputs are spread across several repo-local directories on
different machines. This module treats those directories as one inventory,
groups intermediate MMIFs by their source video, and prefers final/enriched
MMIFs over pipeline step files when multiple candidates exist.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MMIF_ROOTS = (
    "data/chicago",
    "data/enriched",
    "data/combined_outputs",
    "data/pipeline_outputs",
    "data/scene_summaries_smoke",
    "data/qwen_outputs",
    "data/face_tracker_outputs",
    "data/face_tracker_outputs_v2",
)

_STEP_SUFFIX_RE = re.compile(r"_step(?P<step>\d+)_.+$")
_STRIP_SUFFIXES = (
    "_enriched",
    "_seed",
    "_caption",
    "_ocr",
    "_faces_v2",
    "_faces",
    "_transnet",
    "_merged",
)


@dataclass
class TimeFrameInfo:
    annotation_id: str
    label: str
    start_ms: int | None
    end_ms: int | None
    text_preview: str = ""
    source_unit: str = "milliseconds"
    raw_start: float | int | None = None
    raw_end: float | int | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.start_ms is None or self.end_ms is None:
            return None
        return self.end_ms - self.start_ms

    def display(self) -> str:
        ts = _fmt_span(
            self.start_ms,
            self.end_ms,
            self.raw_start,
            self.raw_end,
            self.source_unit,
        )
        text = f" - {self.text_preview[:40]}" if self.text_preview else ""
        return f"[{self.label}] {ts}{text}"


@dataclass
class ViewInfo:
    view_id: str
    app: str
    contains: list[str]
    timeunit: str = "milliseconds"
    label_counts: dict[str, int] = field(default_factory=dict)
    labelset: list[str] = field(default_factory=list)
    annotation_count: int = 0

    @property
    def short_app(self) -> str:
        return self.app.split("/")[-2] if "/" in self.app else self.app

    def display(self) -> str:
        types = ", ".join(t.split("/")[-2] for t in self.contains) or "(no contains)"
        return (
            f"{self.view_id} · {self.short_app} · {types} · "
            f"{self.annotation_count} ann · {self.timeunit}"
        )


@dataclass
class MmifSnapshot:
    video_id: str
    mmif_path: str | None
    views: list[ViewInfo]
    source_root: str | None = None

    def views_by_app(self, needle: str) -> list[ViewInfo]:
        n = needle.lower()
        return [v for v in self.views if n in v.app.lower()]

    def views_with_type(self, type_name: str) -> list[ViewInfo]:
        return [
            v
            for v in self.views
            if any(type_name.lower() in c.lower() for c in v.contains)
        ]


@dataclass
class VideoInfo:
    video_id: str
    label: str
    mmif_path: str | None = None
    source_root: str | None = None


@dataclass(frozen=True)
class _TimeValue:
    raw: float
    unit: str
    ms: int | None


class MmifInventory:
    """Catalog-aware lookup over one or more MMIF output roots."""

    def __init__(
        self,
        mmif_roots: Path | str | Iterable[Path | str] | None = None,
        repository=None,
    ):
        self.mmif_roots = normalise_roots(mmif_roots)
        self.repository = repository
        self._cache: dict[str, dict] = {}

    @classmethod
    def default_roots(cls) -> list[Path]:
        return normalise_roots(None)

    def roots_display(self) -> str:
        return os.pathsep.join(str(p) for p in self.mmif_roots)

    def list_videos(self) -> list[VideoInfo]:
        videos: dict[str, VideoInfo] = {}

        for path in self._iter_mmif_paths():
            vid = video_id_from_path(path)
            existing = videos.get(vid)
            if existing is None or self._prefer(path, existing.mmif_path):
                videos[vid] = VideoInfo(
                    video_id=vid,
                    label=_humanize(vid),
                    mmif_path=str(path),
                    source_root=str(path.parent),
                )

        if self.repository is not None:
            try:
                for record in self.repository.list_records():
                    if record.video_id in videos:
                        continue
                    videos[record.video_id] = VideoInfo(
                        video_id=record.video_id,
                        label=_humanize(record.video_id),
                    )
            except Exception:
                pass

        return sorted(videos.values(), key=lambda v: v.label.lower())

    def snapshot(self, video_id: str) -> MmifSnapshot:
        path = self._video_path(video_id)
        if path is None or not path.exists():
            return MmifSnapshot(video_id=video_id, mmif_path=None, views=[])

        mmif = self._load(path)
        views = []
        for v in mmif.get("views", []):
            md = v.get("metadata", {})
            contains = md.get("contains", {}) or {}
            timeunit = _view_timeunit(v)
            labelset: list[str] = []
            for type_meta in contains.values():
                if isinstance(type_meta, dict) and "labelset" in type_meta:
                    labelset = list(type_meta["labelset"])
            label_counts: Counter[str] = Counter()
            for ann in v.get("annotations", []):
                lbl = ann.get("properties", {}).get("label")
                if lbl:
                    label_counts[lbl] += 1
            views.append(
                ViewInfo(
                    view_id=v.get("id", "?"),
                    app=md.get("app", ""),
                    contains=list(contains.keys()),
                    timeunit=timeunit,
                    label_counts=dict(label_counts),
                    labelset=labelset,
                    annotation_count=len(v.get("annotations", [])),
                )
            )
        return MmifSnapshot(
            video_id=video_id,
            mmif_path=str(path),
            source_root=str(path.parent),
            views=views,
        )

    def labels_in_view(self, video_id: str, view_id: str) -> list[tuple[str, int]]:
        snap = self.snapshot(video_id)
        for v in snap.views:
            if v.view_id == view_id:
                return sorted(v.label_counts.items(), key=lambda kv: kv[1], reverse=True)
        return []

    def timeframes(
        self,
        video_id: str,
        view_id: str,
        labels: Iterable[str] | None = None,
        limit: int = 50,
    ) -> list[TimeFrameInfo]:
        path = self._video_path(video_id)
        if path is None or not path.exists():
            return []

        mmif = self._load(path)
        fps = _extract_fps(mmif)
        tp_index = _index_timepoints(mmif, fps)
        target_view = None
        for v in mmif.get("views", []):
            if v.get("id") == view_id:
                target_view = v
                break
        if target_view is None:
            return []

        timeunit = _view_timeunit(target_view)
        wanted = set(l.lower() for l in (labels or [])) or None
        out: list[TimeFrameInfo] = []
        for ann in target_view.get("annotations", []):
            if "TimeFrame" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            label = props.get("label", "")
            if wanted and label.lower() not in wanted:
                continue
            ann_id = props.get("id") or ann.get("id") or ""
            start_ms, end_ms, raw_start, raw_end, source_unit = _resolve_span(
                props,
                tp_index,
                timeunit,
                fps,
            )
            out.append(
                TimeFrameInfo(
                    annotation_id=_annotation_ref(view_id, ann_id),
                    label=label,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    text_preview=_text_value(props.get("text") or ""),
                    source_unit=source_unit,
                    raw_start=raw_start,
                    raw_end=raw_end,
                )
            )
            if len(out) >= limit:
                break
        return out

    def _iter_mmif_paths(self) -> Iterable[Path]:
        seen: set[Path] = set()
        for root in self.mmif_roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.mmif")):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path

    def _video_path(self, video_id: str) -> Path | None:
        best: Path | None = None
        direct_names = (
            f"{video_id}.mmif",
            f"{video_id}_enriched.mmif",
            f"{video_id}_seed.mmif",
        )
        for root in self.mmif_roots:
            for name in direct_names:
                path = root / name
                if path.exists() and (best is None or self._prefer(path, str(best))):
                    best = path
        for path in self._iter_mmif_paths():
            if video_id_from_path(path) == video_id or path.stem == video_id:
                if best is None or self._prefer(path, str(best)):
                    best = path
        return best

    def _load(self, path: Path) -> dict:
        cached = self._cache.get(str(path))
        if cached is None:
            with path.open() as f:
                cached = json.load(f)
            self._cache[str(path)] = cached
        return cached

    def _prefer(self, candidate: Path, existing: str | None) -> bool:
        if existing is None:
            return True
        return _mmif_rank(candidate, self.mmif_roots) > _mmif_rank(
            Path(existing),
            self.mmif_roots,
        )


def normalise_roots(
    roots: Path | str | Iterable[Path | str] | None,
) -> list[Path]:
    if roots is None:
        raw = os.environ.get("CLAMS_HAYSTACK_MMIF_ROOTS")
        if raw is None:
            raw = os.environ.get("CLAMS_HAYSTACK_PIPELINE_OUTPUTS")
        items: Iterable[Path | str]
        if raw:
            items = _split_root_string(raw)
        else:
            items = DEFAULT_MMIF_ROOTS
    elif isinstance(roots, (str, Path)):
        items = _split_root_string(str(roots)) if isinstance(roots, str) else [roots]
    else:
        items = roots

    out: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        if not item:
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        if path not in seen:
            out.append(path)
            seen.add(path)
    return out


def video_id_from_path(path: Path) -> str:
    stem = path.stem
    step_match = _STEP_SUFFIX_RE.search(stem)
    if step_match:
        return stem[: step_match.start()]
    for suffix in _STRIP_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _split_root_string(raw: str) -> list[str]:
    return [p.strip() for p in raw.replace(",", os.pathsep).split(os.pathsep) if p.strip()]


def _mmif_rank(path: Path, roots: list[Path]) -> tuple[int, int, str]:
    stem = path.stem
    suffix_score = 900
    step_match = _STEP_SUFFIX_RE.search(stem)
    if stem.endswith("_enriched"):
        suffix_score = 1000
    elif step_match:
        suffix_score = 100 + int(step_match.group("step"))
    elif stem.endswith("_seed"):
        suffix_score = 10
    elif stem.endswith(("_caption", "_ocr", "_faces", "_faces_v2", "_transnet", "_merged")):
        suffix_score = 60

    root_score = 0
    parent = path.parent.resolve()
    for idx, root in enumerate(roots):
        if parent == root.resolve():
            root_score = len(roots) - idx
            break
    return suffix_score, root_score, str(path)


def _view_timeunit(view: dict) -> str:
    contains = view.get("metadata", {}).get("contains", {}) or {}
    for type_meta in contains.values():
        if isinstance(type_meta, dict) and type_meta.get("timeUnit"):
            return str(type_meta["timeUnit"])
    return "milliseconds"


def _index_timepoints(mmif: dict, fps: float | None) -> dict[str, _TimeValue]:
    index: dict[str, _TimeValue] = {}
    for v in mmif.get("views", []):
        vid = v.get("id")
        unit = _view_timeunit(v)
        for ann in v.get("annotations", []):
            if "TimePoint" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            tp_id = props.get("id") or ann.get("id")
            t = props.get("timePoint")
            if tp_id is None or t is None:
                continue
            value = _time_value(t, unit, fps)
            index[str(tp_id)] = value
            if vid:
                short_id = str(tp_id).split(":")[-1]
                index[f"{vid}:{short_id}"] = value
    return index


def _resolve_span(
    props: dict,
    tp_index: dict[str, _TimeValue],
    view_timeunit: str,
    fps: float | None,
) -> tuple[int | None, int | None, float | None, float | None, str]:
    if "start" in props and "end" in props:
        unit = str(props.get("timeUnit") or view_timeunit)
        try:
            start_value = _time_value(props["start"], unit, fps)
            end_value = _time_value(props["end"], unit, fps)
            return (
                start_value.ms,
                end_value.ms,
                start_value.raw,
                end_value.raw,
                unit,
            )
        except (TypeError, ValueError):
            pass

    targets = props.get("targets") or []
    values = [tp_index[ref] for ref in targets if ref in tp_index]
    if not values:
        return None, None, None, None, view_timeunit

    raw_start = min(v.raw for v in values)
    raw_end = max(v.raw for v in values)
    units = {v.unit for v in values}
    source_unit = values[0].unit if len(units) == 1 else "mixed"
    if all(v.ms is not None for v in values):
        ms_values = [v.ms for v in values if v.ms is not None]
        return min(ms_values), max(ms_values), raw_start, raw_end, source_unit
    return None, None, raw_start, raw_end, source_unit


def _time_value(raw: Any, unit: str, fps: float | None) -> _TimeValue:
    value = float(raw)
    return _TimeValue(raw=value, unit=unit, ms=_to_ms(value, unit, fps))


def _to_ms(value: float, unit: str, fps: float | None) -> int | None:
    normalised = unit.lower()
    if normalised in {"millisecond", "milliseconds", "ms"}:
        return int(round(value))
    if normalised in {"second", "seconds", "sec", "secs", "s"}:
        return int(round(value * 1000))
    if normalised in {"frame", "frames"}:
        if fps and fps > 0:
            return int(round((value / fps) * 1000))
        return None
    return int(round(value))


def _extract_fps(mmif: dict) -> float | None:
    keys = ("fps", "frameRate", "frame_rate", "framerate")
    candidates: list[Any] = []
    for doc in mmif.get("documents", []):
        props = doc.get("properties", {}) or {}
        candidates.extend(props.get(k) for k in keys if k in props)
        technical = props.get("technical_metadata") or props.get("technicalMetadata") or {}
        if isinstance(technical, dict):
            candidates.extend(technical.get(k) for k in keys if k in technical)
    for view in mmif.get("views", []):
        md = view.get("metadata", {}) or {}
        params = md.get("parameters") or {}
        config = md.get("appConfiguration") or {}
        if isinstance(params, dict):
            candidates.extend(params.get(k) for k in keys if k in params)
        if isinstance(config, dict):
            candidates.extend(config.get(k) for k in keys if k in config)
    for candidate in candidates:
        fps = _parse_fps(candidate)
        if fps:
            return fps
    return None


def _parse_fps(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        try:
            return float(num) / float(den)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def _annotation_ref(view_id: str, ann_id: str) -> str:
    if not ann_id:
        return view_id
    ann_id = str(ann_id)
    if ":" in ann_id:
        return ann_id
    return f"{view_id}:{ann_id}"


def _text_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("@value") or value.get("value") or "")
    return str(value)


def _humanize(video_id: str) -> str:
    return video_id.replace("_", " ").strip()


def _fmt_span(
    start_ms: int | None,
    end_ms: int | None,
    raw_start: float | int | None = None,
    raw_end: float | int | None = None,
    unit: str = "milliseconds",
) -> str:
    if start_ms is not None and end_ms is not None:
        return f"{_fmt_ts(start_ms)}-{_fmt_ts(end_ms)}"
    if raw_start is not None and raw_end is not None:
        return f"{_fmt_raw(raw_start)}-{_fmt_raw(raw_end)} {unit}"
    return "??:??-??:??"


def _fmt_raw(value: float | int) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _fmt_ts(ms: int) -> str:
    s, ms_rem = divmod(int(ms), 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}.{ms_rem:03d}"
    return f"{m:02d}:{s:02d}.{ms_rem:03d}"
