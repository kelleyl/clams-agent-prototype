#!/usr/bin/env python3
"""Simulate CLAMS tool outputs from existing MMIF files and video indexes.

Instead of re-running tools, extracts what each tool actually produced
from the MMIF pipeline outputs. Used to create realistic tool observations
for orchestration training data.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MMIFOutputSimulator:
    """Extracts real tool outputs from existing MMIF files."""

    def __init__(self, mmif_dir: Path, index_dir: Path):
        self.mmif_dir = Path(mmif_dir)
        self.index_dir = Path(index_dir)
        self._mmif_cache: Dict[str, dict] = {}
        self._index_cache: Dict[str, dict] = {}

    def _load_mmif(self, video_id: str) -> Optional[dict]:
        if video_id in self._mmif_cache:
            return self._mmif_cache[video_id]
        path = self.mmif_dir / f"{video_id}.mmif"
        if not path.exists():
            return None
        doc = json.load(open(path))
        self._mmif_cache[video_id] = doc
        return doc

    def _load_index(self, video_id: str) -> Optional[dict]:
        if video_id in self._index_cache:
            return self._index_cache[video_id]
        path = self.index_dir / f"{video_id}.json"
        if not path.exists():
            return None
        doc = json.load(open(path))
        self._index_cache[video_id] = doc
        return doc

    def _find_view(self, mmif: dict, app_keyword: str,
                   params_match: dict = None) -> Optional[dict]:
        """Find a MMIF view matching the app keyword and optional parameters."""
        for view in mmif.get("views", []):
            app = view.get("metadata", {}).get("app", "")
            if app_keyword not in app:
                continue
            if params_match:
                view_params = view.get("metadata", {}).get("parameters", {})
                if all(view_params.get(k) == v for k, v in params_match.items()):
                    return view
            else:
                return view
        return None

    def _filter_annotations_by_time(self, annotations: list,
                                     start_ms: int, end_ms: int) -> list:
        """Filter annotations to those overlapping a time range."""
        filtered = []
        for ann in annotations:
            props = ann.get("properties", {})
            ann_start = props.get("start", props.get("timePoint", 0))
            ann_end = props.get("end", ann_start + 1000)
            if ann_end > start_ms and ann_start < end_ms:
                filtered.append(ann)
        return filtered

    def simulate_video_info(self, video_id: str) -> str:
        """Simulate ffmpeg:get_video_info output."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "Video not found"})
        return json.dumps({
            "video_id": video_id,
            "video_path": idx.get("video_path", ""),
            "duration_ms": idx.get("duration_ms", 0),
            "duration_formatted": _format_time(idx.get("duration_ms", 0)),
        })

    def simulate_swt_detection(self, video_id: str,
                                time_range: Tuple[int, int] = None) -> str:
        """Simulate swt-detection output: TimeFrame annotations with labels."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "No index for video"})

        # Extract SWT-labeled segments from the index
        timeframes = []
        for seg in idx.get("segments", []):
            label = seg.get("scene_label", "Neg")
            if label in ("Neg", "shot"):
                continue  # Skip non-text scenes
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue
            timeframes.append({
                "start_ms": start,
                "end_ms": end,
                "label": label,
                "duration_s": round((end - start) / 1000, 1),
                "time_formatted": f"{_format_time(start)} - {_format_time(end)}",
            })

        # Also check MMIF for richer SWT data
        mmif = self._load_mmif(video_id)
        if mmif and not timeframes:
            view = self._find_view(mmif, "swt-detection")
            if view:
                for ann in view.get("annotations", []):
                    if "TimeFrame" not in ann.get("@type", ""):
                        continue
                    props = ann.get("properties", {})
                    start = props.get("start", 0)
                    end = props.get("end", 0)
                    label = props.get("label", "")
                    if label and label != "Neg":
                        if time_range and (end < time_range[0] or start > time_range[1]):
                            continue
                        timeframes.append({
                            "start_ms": start, "end_ms": end,
                            "label": label,
                            "duration_s": round((end - start) / 1000, 1),
                        })

        return json.dumps({
            "tool": "swt-detection",
            "text_scenes_found": len(timeframes),
            "timeframes": timeframes[:30],  # Cap for prompt length
        }, indent=2)

    def simulate_asr(self, video_id: str,
                     time_range: Tuple[int, int] = None) -> str:
        """Simulate parakeet-wrapper output: timestamped transcript."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "No index for video"})

        segments = []
        full_transcript = []
        for seg in idx.get("segments", []):
            asr = seg.get("asr_transcript", "")
            if not asr:
                continue
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue
            segments.append({
                "start_ms": start,
                "end_ms": end,
                "time_formatted": f"{_format_time(start)} - {_format_time(end)}",
                "text": asr[:300],
            })
            full_transcript.append(asr)

        speaker_names = idx.get("speaker_names", {})

        return json.dumps({
            "tool": "parakeet-wrapper",
            "transcript_segments": len(segments),
            "total_words": sum(len(s["text"].split()) for s in segments),
            "speakers_identified": list(speaker_names.values())[:5],
            "segments": segments[:20],  # Cap for prompt length
        }, indent=2)

    def simulate_shot_detection(self, video_id: str) -> str:
        """Simulate transnet-wrapper output: shot boundaries."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "No index for video"})

        shots = []
        for seg in idx.get("segments", []):
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            shots.append({
                "start_ms": start,
                "end_ms": end,
                "duration_s": round((end - start) / 1000, 1),
            })

        return json.dumps({
            "tool": "transnet-wrapper",
            "shots_detected": len(shots),
            "average_shot_duration_s": round(
                sum(s["duration_s"] for s in shots) / max(1, len(shots)), 1
            ),
            "shots": shots[:30],
        }, indent=2)

    def simulate_captioner(self, video_id: str, config: str,
                           time_range: Tuple[int, int] = None) -> str:
        """Simulate qwen3vl-captioner output for either OCR or captioning."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "No index for video"})

        is_ocr = "swt_transcription" in config
        results = []

        for seg in idx.get("segments", []):
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue

            if is_ocr:
                # OCR mode: extract on-screen text
                ocr = seg.get("ocr_text", "") or ""
                if not ocr or ocr.startswith("The image") or ocr == "None":
                    continue
                results.append({
                    "start_ms": start,
                    "end_ms": end,
                    "time_formatted": f"{_format_time(start)} - {_format_time(end)}",
                    "on_screen_text": ocr[:200],
                    "scene_label": seg.get("scene_label", ""),
                })
            else:
                # Caption mode: visual descriptions
                caption = seg.get("visual_caption", "") or ""
                if not caption:
                    continue
                results.append({
                    "start_ms": start,
                    "end_ms": end,
                    "time_formatted": f"{_format_time(start)} - {_format_time(end)}",
                    "visual_description": caption[:200],
                })

        mode = "OCR (on-screen text extraction)" if is_ocr else "visual captioning"
        return json.dumps({
            "tool": "qwen3vl-captioner",
            "config": config,
            "mode": mode,
            "results_count": len(results),
            "results": results[:20],
        }, indent=2)

    def simulate_ner(self, video_id: str,
                     time_range: Tuple[int, int] = None) -> str:
        """Simulate spacy-wrapper NER output."""
        idx = self._load_index(video_id)
        if not idx:
            return json.dumps({"error": "No index for video"})

        all_entities = []
        seen = set()
        for seg in idx.get("segments", []):
            start = seg.get("start_ms", 0)
            end = seg.get("end_ms", 0)
            if time_range and (end < time_range[0] or start > time_range[1]):
                continue
            for ent in seg.get("named_entities", []):
                key = (ent["text"], ent["type"])
                if key not in seen:
                    seen.add(key)
                    all_entities.append({
                        "text": ent["text"],
                        "type": ent["type"],
                        "first_seen_ms": start,
                    })

        return json.dumps({
            "tool": "spacy-wrapper",
            "entities_found": len(all_entities),
            "entities": all_entities[:30],
        }, indent=2)

    def simulate_tool(self, video_id: str, tool_name: str,
                      parameters: dict = None,
                      time_range: Tuple[int, int] = None) -> str:
        """Dispatch to the right simulator based on tool name."""
        if tool_name == "swt-detection":
            return self.simulate_swt_detection(video_id, time_range)
        elif tool_name in ("parakeet-wrapper", "whisper-wrapper"):
            return self.simulate_asr(video_id, time_range)
        elif tool_name == "transnet-wrapper":
            return self.simulate_shot_detection(video_id)
        elif tool_name == "qwen3vl-captioner":
            config = (parameters or {}).get("config", "config/transnet_shots.yaml")
            return self.simulate_captioner(video_id, config, time_range)
        elif tool_name == "spacy-wrapper":
            return self.simulate_ner(video_id, time_range)
        elif tool_name == "ffmpeg:get_video_info":
            return self.simulate_video_info(video_id)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS or HH:MM:SS."""
    total_s = ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
