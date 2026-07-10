#!/usr/bin/env python3
"""Construct trajectories reflecting actual CLAMS tool execution decisions.

Trajectories follow a locate-then-inspect pattern: the agent first finds
the relevant time region in the video (via search_transcript, search_ocr,
or browse_timeline), then inspects that region with run_asr, caption_frame,
or extract_text. This mirrors the coarse-to-fine reasoning that makes
the RAG baseline effective (~75% accuracy) while preserving the agent's
ability to choose *which* tools and models to apply.

Tool palette (matching actual CLAMS apps):
- search_transcript(query, top_k): Keyword search across all ASR segments.
- search_ocr(query, top_k): Keyword search across all OCR items.
- run_asr(start_time, end_time, model): Get ASR in a time range. Requires time bounds.
- detect_text_scenes(): Run SWT to find frames with text.
- extract_text(timestamp, model): Run OCR on a frame. model=qwen-small|qwen-8b|qwen-30b.
- caption_frame(timestamp, model, task_mode): Run VLM on a frame.
    model=smolvlm|qwen-small|qwen-8b|qwen-30b.
    task_mode=general_scene|text_focus|person_identity|action_event.
- identify_speakers(start_time, end_time): Speaker diarization in a time range.
- browse_timeline(start_time, end_time): Compact summary of a time range.

Noise injection:
- Sometimes the agent tries a cheaper tool first (SWT) that misses things, then escalates to VLM
- Sometimes the agent runs a tool that returns unhelpful results and has to try a different approach
- These "failed then recovered" trajectories teach adaptation

Usage:
    python training_data/construct_tool_trajectories.py \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/trajectories_tool_execution.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

random.seed(42)

# Stopwords for keyword search scoring
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "don", "now", "and", "but", "or", "if", "while", "that", "this",
    "what", "which", "who", "whom", "it", "its", "he", "she", "they",
    "them", "his", "her", "their", "my", "your", "our", "me", "we", "you",
    "i", "about", "up",
}

# Maps index layer sources to the CLAMS tool that produced them
SOURCE_TO_TOOL = {
    "parakeet-wrapper": "run_asr",
    "swt-detection": "detect_text_scenes",
    "qwen3vl-captioner/swt_transcription": "extract_text",
    "qwen3vl-captioner/transnet_shots": "caption_frame",
    "pyannote+llm": "identify_speakers",
    "spacy+grounding": "extract_entities",
    "transnet-wrapper": "detect_shots",
    "diarization/llm": "segment_topics",
}

ASR_MODELS = ["parakeet", "whisper"]
OCR_MODELS = ["qwen-small", "qwen-8b", "qwen-30b", "qwen3vl-8b"]
VLM_MODELS = ["smolvlm", "qwen-small", "qwen-8b", "qwen-30b", "qwen3vl-8b"]
CAPTION_TASK_MODES = ["general_scene", "text_focus", "person_identity", "action_event"]


# Tool descriptions for the system prompt
TOOL_DESCRIPTIONS = {
    "search_transcript": {
        "description": "Keyword search across all ASR segments in the video. Returns matching segments with timestamps and surrounding context.",
        "cost": "free",
        "accuracy": "N/A",
        "parameters": {
            "query": "Keywords to search for in the transcript",
            "top_k": "Number of results to return (default 5)",
        },
    },
    "search_ocr": {
        "description": "Keyword search across all OCR items (chyrons, slates, credits). Returns matching on-screen text with timestamps and scene labels.",
        "cost": "free",
        "accuracy": "N/A",
        "parameters": {
            "query": "Keywords to search for in on-screen text",
            "top_k": "Number of results to return (default 5)",
        },
    },
    "run_asr": {
        "description": "Get speech transcript in a specific time range. Must specify start and end time.",
        "cost": "low",
        "accuracy": "3/5",
        "models": ASR_MODELS,
        "parameters": {
            "start_time": "Start time (e.g., '15:00')",
            "end_time": "End time (e.g., '16:00')",
            "model": "ASR model (parakeet=fast, whisper=accurate on noisy audio)",
        },
    },
    "detect_text_scenes": {
        "description": "Detect frames containing text (chyrons, slates, credits, graphics) using scene classification. Fast but may miss some text.",
        "cost": "low",
        "accuracy": "2/5",
        "parameters": {},
    },
    "extract_text": {
        "description": "Run OCR on a specific frame to read on-screen text. Use after finding a relevant timestamp via search or browsing.",
        "cost": "low-high (depends on model)",
        "accuracy": "3-5/5 (depends on model)",
        "models": OCR_MODELS,
        "parameters": {
            "timestamp": "time in the video (e.g., '15:47')",
            "model": "VLM for OCR (qwen-small=fast, qwen-8b=good for complex text, qwen-30b=strongest when available)",
        },
    },
    "caption_frame": {
        "description": "Analyze a video frame with a vision-language model. Supports task-specific modes.",
        "cost": "low-high (depends on model)",
        "accuracy": "3-5/5 (depends on model)",
        "models": VLM_MODELS,
        "task_modes": CAPTION_TASK_MODES,
        "parameters": {
            "timestamp": "time in the video (e.g., '15:47')",
            "model": "VLM (smolvlm=fastest, qwen-small=fast, qwen-8b=detailed, qwen-30b=strongest when available)",
            "task_mode": "What to look for: general_scene, text_focus, person_identity, or action_event.",
        },
    },
    "identify_speakers": {
        "description": "Run speaker diarization to identify who speaks when in a time range.",
        "cost": "medium",
        "accuracy": "3/5",
        "parameters": {
            "start_time": "Start time (e.g., '15:00')",
            "end_time": "End time (e.g., '16:00')",
        },
    },
    "browse_timeline": {
        "description": "Get a compact summary of a time range: key speech topics, speaker names, chyron text, notable timestamps.",
        "cost": "free",
        "accuracy": "N/A",
        "parameters": {
            "start_time": "start time (e.g., '15:00')",
            "end_time": "end time (e.g., '16:00')",
        },
    },
}

# Valid models per tool (mirrors run_grpo_env.py)
TOOL_MODELS = {
    "run_asr": ASR_MODELS,
    "extract_text": OCR_MODELS,
    "caption_frame": VLM_MODELS,
}

# Default (cheapest) model per tool
TOOL_MODEL_DEFAULTS = {
    "run_asr": "parakeet",
    "extract_text": "qwen3vl-8b",
    "caption_frame": "qwen3vl-8b",
}

# Valid task_modes for caption_frame. All known precomputed caption variants are
# exposed so verified evidence spans can replay the exact layer they used.

# Tool schemas for Qwen3.5 native format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Keyword search across all ASR segments in the video. Returns matching segments with timestamps and surrounding context. Use this first to locate relevant parts of the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for in the transcript"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Keyword search across all OCR items (chyrons, slates, credits). Returns matching on-screen text with timestamps and scene labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Keywords to search for in on-screen text"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)"},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Get speech transcript in a specific time range. You must first locate a relevant region (via search_transcript or browse_timeline) before calling this.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '16:00')"},
                    "model": {
                        "type": "string",
                        "enum": ASR_MODELS,
                        "description": "ASR model. 'parakeet' is fast but may miss unclear speech. 'whisper' is slower but more accurate on noisy audio."
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_text_scenes",
            "description": "Detect frames containing text (chyrons, slates, credits). Fast but may miss some. Returns timestamps and scene types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text",
            "description": "Run OCR on a specific frame to read on-screen text. Use after finding a relevant timestamp via search or browsing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"},
                    "model": {
                        "type": "string",
                        "enum": OCR_MODELS,
                        "description": "VLM for OCR. 'qwen-small' is fastest. 'qwen-8b' is good for complex or multi-line text. 'qwen-30b' is strongest when available."
                    },
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "caption_frame",
            "description": "Analyze video frames with a vision-language model. Provide a single timestamp for one frame, or start_time+end_time to caption all shots in a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Single frame time (e.g., '15:47'). Use this OR start_time+end_time."},
                    "start_time": {"type": "string", "description": "Start of time range (e.g., '1:00'). Use with end_time to caption all shots in the range."},
                    "end_time": {"type": "string", "description": "End of time range (e.g., '1:30'). Use with start_time."},
                    "model": {
                        "type": "string",
                        "enum": VLM_MODELS,
                        "description": "VLM to use. 'qwen3vl-8b' has best coverage. 'qwen-8b' gives detailed output. 'qwen-30b' is strongest when available."
                    },
                    "task_mode": {
                        "type": "string",
                        "enum": CAPTION_TASK_MODES,
                        "description": "What to look for: 'general_scene' for broad visual description, 'text_focus' for on-screen text, 'person_identity' for visible people, 'action_event' for actions/events."
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_speakers",
            "description": "Run speaker diarization to identify who speaks when in a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '16:00')"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_timeline",
            "description": "Get a compact summary of a time range: key speech topics, speaker names, chyron text, notable timestamps. Use to explore the video before inspecting specific moments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '16:00')"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
]


def get_tool_schemas(allow_browse=True):
    """Return tool schemas.

    allow_browse is kept for compatibility with older scripts, but all tools are
    now exposed so experiments do not silently remove capabilities.
    """
    schemas = json.loads(json.dumps(TOOL_SCHEMAS))
    return schemas


def format_time(ms):
    """Format milliseconds as MM:SS."""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def get_layer_items_at(layers, layer_name, start_ms, end_ms):
    """Get items from a layer overlapping a time range."""
    items = layers.get(layer_name, {}).get("items", [])
    return [i for i in items if i.get("start_ms", 0) < end_ms and i.get("end_ms", 0) > start_ms]


def simulate_tool_output(tool_name, index_data, timestamp_ms=None, context=None,
                         layer_override=None, model=None, task_mode=None,
                         query=None, top_k=5, start_ms=None, end_ms=None):
    """Simulate what a CLAMS tool would return, using actual index data.

    Args:
        tool_name: Name of the tool to simulate.
        index_data: Video index data dict with "layers" and "duration_ms".
        timestamp_ms: Optional timestamp in milliseconds for time-specific tools.
        context: Optional context dict (e.g., {"start_ms": ..., "end_ms": ...}).
        layer_override: Optional explicit layer name override (e.g., "asr_whisper").
        model: Optional model name controlling output detail level.
        task_mode: Optional task mode for caption_frame.
        query: Search query string (for search_transcript, search_ocr).
        top_k: Number of search results to return (default 5).
        start_ms: Start of time range in milliseconds (for run_asr, browse_timeline, identify_speakers).
        end_ms: End of time range in milliseconds (for run_asr, browse_timeline, identify_speakers).
    """
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    if tool_name == "search_transcript":
        if not query:
            return "Error: query required"
        query = str(query)  # handle int/float passed as query
        # Keyword search across all ASR segments
        query_words = set(query.lower().split()) - STOPWORDS
        if not query_words:
            query_words = set(query.lower().split())  # fall back to all words
        all_items = []
        for asr_layer in ["asr", "asr_whisper"]:
            all_items.extend(layers.get(asr_layer, {}).get("items", []))
        if not all_items:
            return "No transcript data available."
        # Score each segment by word overlap
        scored = []
        for item in all_items:
            text = item.get("text", "")
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return f"No transcript segments matching '{query}'."
        lines = []
        seen_times = set()
        for score, item in scored[:top_k]:
            t = format_time(item["start_ms"])
            if t in seen_times:
                continue
            seen_times.add(t)
            pct = int(item["start_ms"] / max(1, duration_ms) * 100)
            lines.append(f"[{t} | {pct}%] {item.get('text', '')[:200]}")
        return f"Found {len(scored)} matching segments (showing top {len(lines)}):\n" + "\n".join(lines)

    elif tool_name == "search_ocr":
        if not query:
            return "Error: query required"
        query = str(query)
        # Keyword search across all OCR/text-focus items.
        query_words = set(query.lower().split()) - STOPWORDS
        if not query_words:
            query_words = set(query.lower().split())
        all_items = []
        for layer_name, layer in layers.items():
            if layer_name.endswith("_text_focus"):
                for item in layer.get("items", []):
                    item = dict(item)
                    item.setdefault("_layer_name", layer_name)
                    all_items.append(item)
        if not all_items:
            return "No OCR data available."
        scored = []
        for item in all_items:
            text = item.get("text", "")
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, item))
        scored.sort(key=lambda x: -x[0])
        if not scored:
            return f"No on-screen text matching '{query}'."
        lines = []
        for score, item in scored[:top_k]:
            t = format_time(item.get("start_ms", 0))
            label = item.get("scene_label", "") or item.get("_layer_name", "")
            text = item.get("text", "")[:200]
            lines.append(f"[{t}] [{label}] {text}")
        return f"Found {len(scored)} matching OCR items (showing top {len(lines)}):\n" + "\n".join(lines)

    elif tool_name == "run_asr":
        # Requires start_ms and end_ms
        if start_ms is None or end_ms is None:
            return "Error: start_time and end_time are required. Use search_transcript first to find relevant time regions."
        if layer_override:
            asr_layer = layer_override
        elif model == "whisper":
            asr_layer = "asr_whisper"
        else:
            asr_layer = "asr"
        items = get_layer_items_at(layers, asr_layer, start_ms, end_ms)
        if not items:
            return "No speech detected in this range."
        lines = []
        for item in items[:15]:
            t = format_time(item["start_ms"])
            pct = int(item["start_ms"] / max(1, duration_ms) * 100)
            lines.append(f"[{t} | {pct}%] {item.get('text', '')[:200]}")
        return "\n".join(lines)

    elif tool_name == "detect_text_scenes":
        items = layers.get("scenes", {}).get("items", [])
        text_scenes = [i for i in items if i.get("label") in ("Chyron", "Slate", "Credits", "Other-text")]
        if not text_scenes:
            return "No text scenes detected."
        lines = []
        for item in text_scenes[:8]:
            t = format_time(item["start_ms"])
            lines.append(f"[{t}] {item.get('label', '')} ({format_time(item['end_ms'] - item['start_ms'])} duration)")
        return f"Found {len(text_scenes)} text scenes:\n" + "\n".join(lines)

    elif tool_name == "extract_text":
        if timestamp_ms is None:
            return "Error: timestamp required"
        variant_layer = f"caption_{model}_text_focus" if model else None
        if variant_layer and variant_layer in layers:
            all_items = layers[variant_layer].get("items", [])
            nearby = [
                item for item in all_items
                if abs(item.get("start_ms", 0) - timestamp_ms) <= 15000
                and item.get("text", "") and not item["text"].startswith("Error")
            ]
            if nearby:
                nearby.sort(key=lambda x: abs(x.get("start_ms", 0) - timestamp_ms))
                parts = []
                for item in nearby[:3]:
                    t = format_time(item["start_ms"])
                    parts.append(f"[{t}] {item['text']}")
                return "\n".join(parts)
        return f"No {model or 'OCR'} text data available for this frame."

    elif tool_name == "caption_frame":
        variant_layer = f"caption_{model}_{task_mode}" if model and task_mode else None

        # Range mode: return all items overlapping [start_ms, end_ms]
        if start_ms is not None and end_ms is not None:
            if variant_layer and variant_layer in layers:
                items = get_layer_items_at(layers, variant_layer, start_ms, end_ms)
                items = [i for i in items if i.get("text") and not i["text"].startswith("Error")]
                if items:
                    parts = []
                    for item in items:
                        t = format_time(item["start_ms"])
                        parts.append(f"[{t}] {item['text']}")
                    return "\n".join(parts)
            return f"No {model or 'caption'} data available for this range."

        # Single-timestamp mode: return closest items within +-15s
        if timestamp_ms is None:
            return "Error: provide timestamp or start_time+end_time"

        CAPTION_WINDOW_MS = 15000
        if variant_layer and variant_layer in layers:
            all_items = layers[variant_layer].get("items", [])
            nearby = [item for item in all_items
                      if abs(item.get("start_ms", 0) - timestamp_ms) <= CAPTION_WINDOW_MS
                      and item.get("text", "") and not item["text"].startswith("Error")]
            if nearby:
                nearby.sort(key=lambda x: abs(x.get("start_ms", 0) - timestamp_ms))
                parts = []
                for item in nearby[:3]:
                    t = format_time(item["start_ms"])
                    parts.append(f"[{t}] {item['text']}")
                return "\n".join(parts)

        return f"No {model or 'caption'} data available for this frame."

    elif tool_name == "identify_speakers":
        if start_ms is not None and end_ms is not None:
            items = get_layer_items_at(layers, "speakers", start_ms, end_ms)
        elif timestamp_ms:
            items = get_layer_items_at(layers, "speakers", timestamp_ms - 15000, timestamp_ms + 15000)
        else:
            return "Error: start_time and end_time are required."
        if not items:
            return "No speaker information available in this range."
        seen = set()
        lines = []
        for item in items:
            name = item.get("speaker_name", "Unknown")
            if name not in seen:
                seen.add(name)
                t = format_time(item["start_ms"])
                lines.append(f"[{t}] {name}: {item.get('text', '')[:100]}")
        return "\n".join(lines)

    elif tool_name == "browse_timeline":
        if start_ms is None or end_ms is None:
            if context and "start_ms" in context and "end_ms" in context:
                start_ms = context["start_ms"]
                end_ms = context["end_ms"]
            else:
                return "Error: start_time and end_time are required."
        # Build compact summary of the time range
        parts = []
        # Speech: key phrases (first sentence of each segment)
        asr_items = get_layer_items_at(layers, "asr", start_ms, end_ms)
        if not asr_items:
            asr_items = get_layer_items_at(layers, "asr_whisper", start_ms, end_ms)
        if asr_items:
            speech_snippets = []
            for item in asr_items[:6]:
                text = item.get("text", "")
                # Take first sentence or first 60 chars
                snippet = text.split(". ")[0] if ". " in text else text[:60]
                if snippet:
                    t = format_time(item["start_ms"])
                    speech_snippets.append(f"  [{t}] {snippet[:80]}")
            if speech_snippets:
                parts.append("Speech:\n" + "\n".join(speech_snippets))
        # OCR: any chyron/slate text from text_focus layers
        ocr_items = []
        for layer_name in layers:
            if layer_name.endswith("_text_focus"):
                ocr_items.extend(get_layer_items_at(layers, layer_name, start_ms, end_ms))
        if ocr_items:
            ocr_snippets = []
            for item in ocr_items[:3]:
                label = item.get("scene_label", "")
                text = item.get("text", "")[:80]
                t = format_time(item.get("start_ms", 0))
                if text and not text.startswith("The "):
                    ocr_snippets.append(f"  [{t}] [{label}] {text}")
            if ocr_snippets:
                parts.append("On-screen text:\n" + "\n".join(ocr_snippets))
        # Speakers
        speaker_items = get_layer_items_at(layers, "speakers", start_ms, end_ms)
        if speaker_items:
            names = []
            seen = set()
            for item in speaker_items:
                name = item.get("speaker_name", "Unknown")
                if name not in seen:
                    seen.add(name)
                    names.append(name)
            if names:
                parts.append(f"Speakers: {', '.join(names[:5])}")
        # Scenes
        scene_items = get_layer_items_at(layers, "scenes", start_ms, end_ms)
        if scene_items:
            labels = [item.get("label", "") for item in scene_items if item.get("label")]
            if labels:
                parts.append(f"Scene types: {', '.join(labels[:5])}")
        if not parts:
            return f"No data available for {format_time(start_ms)}-{format_time(end_ms)}."
        header = f"Summary for {format_time(start_ms)}-{format_time(end_ms)}:"
        # Cap total output to ~500 chars per 5-minute window
        summary = header + "\n" + "\n".join(parts)
        window_minutes = max(1, (end_ms - start_ms) / 60000)
        char_budget = int(500 * window_minutes / 5)
        if len(summary) > char_budget:
            summary = summary[:char_budget] + "..."
        return summary

    elif tool_name == "extract_entities":
        if timestamp_ms:
            items = get_layer_items_at(layers, "entities", timestamp_ms - 15000, timestamp_ms + 15000)
        else:
            items = layers.get("entities", {}).get("items", [])[:10]
        if not items:
            return "No entities found."
        lines = []
        for item in items[:8]:
            lines.append(f"{item.get('text', '')} ({item.get('type', '')})")
        return ", ".join(lines)

    return f"Unknown tool: {tool_name}"


def _available_caption_models(layers):
    """Return list of caption models that have real (non-error) pre-computed data."""
    return sorted({model for model, _task_mode in _available_caption_pairs(layers)})


def _available_caption_pairs(layers):
    """Return available (model, task_mode) caption variant pairs."""
    available = set()
    for model in VLM_MODELS:
        for task_mode in CAPTION_TASK_MODES:
            layer = layers.get(f"caption_{model}_{task_mode}", {})
            items = layer.get("items", [])
            if items and not items[0].get("text", "").startswith("Error"):
                available.add((model, task_mode))
    return available


def _available_asr_models(layers):
    """Return list of ASR models that have real data in this index."""
    available = []
    if layers.get("asr", {}).get("items"):
        available.append("parakeet")
    if layers.get("asr_whisper", {}).get("items"):
        available.append("whisper")
    return available or ["parakeet"]  # always allow parakeet as fallback


def _pick_asr_model(grounding, available_models=None):
    """Pick ASR model based on grounding hints.

    Uses whisper when grounding suggests degraded or unclear audio;
    defaults to parakeet (cheaper) otherwise.
    """
    speech = grounding.get("speech_excerpt") or ""
    reasoning = grounding.get("reasoning") or ""
    # Upgrade to whisper for noisy/unclear/degraded audio indicators
    degraded_hints = ["unclear", "noisy", "garbled", "hard to hear",
                      "background noise", "degraded", "poor audio",
                      "difficult to make out", "faint"]
    combined = (speech + " " + reasoning).lower()
    if any(h in combined for h in degraded_hints):
        if available_models is None or "whisper" in available_models:
            return "whisper"
    return "parakeet"


def _pick_ocr_model(grounding):
    """Pick OCR (extract_text) model based on grounding hints.

    Upgrades from qwen-small when grounding suggests degraded/complex text.
    """
    ocr = grounding.get("ocr_reference") or ""
    reasoning = grounding.get("reasoning") or ""
    combined = (ocr + " " + reasoning).lower()
    degraded_hints = ["degraded", "blurry", "small text", "hard to read",
                      "faded", "low resolution", "low quality", "partial",
                      "truncated", "overlapping"]
    complex_hints = ["multiple lines", "credits", "slate", "several names",
                     "dense text", "paragraph"]
    if any(h in combined for h in degraded_hints) or any(h in combined for h in complex_hints):
        return "qwen-8b"
    return "qwen-small"


def _pick_caption_model_and_mode(
    question,
    grounding,
    available_models=None,
    available_pairs=None,
):
    """Pick caption_frame model and task_mode based on question and grounding.

    Only picks from available_models (models with real pre-computed data).
    Returns (model, task_mode) tuple.
    """
    q_lower = question.lower()
    visual = grounding.get("visual_reference") or ""
    ocr = grounding.get("ocr_reference") or ""
    reasoning = (grounding.get("reasoning") or "").lower()

    # Determine task_mode from question content
    person_hints = ["who is", "who are", "person", "people", "speaker",
                    "guest", "host", "interviewer", "interviewee",
                    "anchor", "reporter", "correspondent", "name of"]
    text_hints = ["chyron", "text on screen", "title", "lower third",
                  "graphic", "what does the text say", "what text",
                  "on-screen text", "caption", "banner", "headline"]
    action_hints = ["what is happening", "what happens", "what occurred",
                    "what took place", "describe the scene", "what action",
                    "what event", "what is going on", "doing"]

    if any(h in q_lower for h in text_hints):
        task_mode = "text_focus"
    elif any(h in q_lower for h in person_hints):
        task_mode = "person_identity"
    elif any(h in q_lower for h in action_hints):
        task_mode = "action_event"
    else:
        task_mode = "general_scene"

    # Determine model quality needed
    degraded_hints = ["degraded", "blurry", "unclear", "hard to see",
                      "low quality", "dark", "distant", "small"]
    combined = (visual + " " + reasoning)
    if any(h in combined.lower() for h in degraded_hints):
        model = "qwen-8b"
    elif task_mode in ("text_focus", "person_identity", "action_event"):
        model = "qwen-8b"
    else:
        model = "smolvlm"

    # Constrain to available model/task pairs (those with real pre-computed data)
    if available_pairs:
        if (model, task_mode) in available_pairs:
            return model, task_mode
        model_preference = ["qwen-30b", "qwen-8b", "qwen-small", "smolvlm"]
        task_preference = [task_mode]
        for fallback_mode in ["general_scene", "text_focus", "action_event", "person_identity"]:
            if fallback_mode not in task_preference:
                task_preference.append(fallback_mode)
        for fallback_mode in task_preference:
            for fallback_model in model_preference:
                if (fallback_model, fallback_mode) in available_pairs:
                    return fallback_model, fallback_mode

    if available_models and model not in available_models:
        # Fall back to best available model
        preference = ["qwen-30b", "qwen-8b", "qwen-small", "smolvlm"]
        for fallback in preference:
            if fallback in available_models:
                model = fallback
                break

    return model, task_mode


def _extract_search_keywords(question, grounding):
    """Extract search keywords from question grounding for search_transcript/search_ocr.

    Uses speech_excerpt and ocr_reference from grounding to build a focused query.
    Falls back to extracting content words from the question itself.
    """
    speech = grounding.get("speech_excerpt") or ""
    ocr_ref = grounding.get("ocr_reference") or ""

    # Prefer grounding text -- it contains the actual words in the video
    if speech:
        # Take distinctive words from the speech excerpt
        words = speech.lower().split()
        keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
        if len(keywords) >= 2:
            return " ".join(keywords[:5])
    if ocr_ref:
        words = ocr_ref.lower().split()
        keywords = [w for w in words if w not in STOPWORDS and len(w) > 2]
        if len(keywords) >= 2:
            return " ".join(keywords[:5])

    # Fall back to question content words
    q_words = question.lower().split()
    keywords = [w.strip("?.,!\"'") for w in q_words if w.strip("?.,!\"'") not in STOPWORDS and len(w.strip("?.,!\"'")) > 2]
    return " ".join(keywords[:5]) if keywords else question[:30]


def _time_window(center_ms, duration_ms, window_ms=60000):
    """Return a bounded time window around center_ms."""
    half_window = max(1000, window_ms // 2)
    start_ms = max(0, center_ms - half_window)
    end_limit = duration_ms if duration_ms else center_ms + half_window
    end_ms = min(end_limit, center_ms + half_window)
    if end_ms <= start_ms:
        end_ms = start_ms + window_ms
    return start_ms, end_ms


def _content_words(text):
    words = []
    for word in str(text or "").lower().split():
        clean = word.strip("?.,!\"'():;[]{}")
        if clean and clean not in STOPWORDS and len(clean) > 2:
            words.append(clean)
    return set(words)


def _choose_wrong_center_ms(index_data, source_ms, query, preferred_layers,
                            min_distance_ms=120000):
    """Pick a plausible but wrong timestamp for retry-recovery trajectories.

    Prefer a real search hit far from the gold source time, then fall back to a
    deterministic far-away timestamp. The point is to teach recovery from
    plausible wrong inspections, not random noise.
    """
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)
    query_words = _content_words(query)
    candidates = []

    for layer_name in preferred_layers:
        for item in layers.get(layer_name, {}).get("items", []):
            start_ms = item.get("start_ms", 0)
            if abs(start_ms - source_ms) < min_distance_ms:
                continue
            text_words = _content_words(item.get("text", ""))
            overlap = len(query_words & text_words) if query_words else 0
            if overlap > 0 or not query_words:
                candidates.append((overlap, start_ms))

    if candidates:
        candidates.sort(key=lambda item: (-item[0], abs(item[1] - source_ms)))
        return candidates[0][1], "far_search_hit"

    if duration_ms and duration_ms > min_distance_ms * 2:
        shifted = source_ms + min_distance_ms
        if shifted >= duration_ms:
            shifted = max(0, source_ms - min_distance_ms)
        return min(duration_ms, max(0, shifted)), "far_time_offset"

    if duration_ms:
        endpoint = 0 if source_ms > duration_ms / 2 else duration_ms
        return endpoint, "video_endpoint"

    return max(0, source_ms + min_distance_ms), "fallback_offset"


def _is_unhelpful_tool_output(output):
    """Return True for tool outputs that should not be treated as evidence."""
    text = str(output or "").strip().lower()
    if not text:
        return True
    unhelpful_markers = [
        "error:",
        "no data available",
        "no transcript data available",
        "no ocr data available",
        "no transcript segments matching",
        "no on-screen text matching",
        "no speech detected",
        "no text detected",
        "no readable text",
        "no visible text",
        "no visual description available",
        "no speaker information",
        "caption data available",
    ]
    return any(marker in text for marker in unhelpful_markers)


def _expected_answer_text(question_data):
    """Return the text answer to train the policy to emit after tool use."""
    return (
        question_data.get("answer")
        or question_data.get("mc_correct_text")
        or question_data.get("reference_answer")
        or question_data.get("mc_correct")
        or ""
    )


def _span_timestamp_ms(span):
    """Choose the best single timestamp for a verified evidence span."""
    for key in ("timestamp_ms", "start_ms"):
        value = span.get(key)
        if value is not None:
            return int(value)
    start = span.get("start_ms")
    end = span.get("end_ms")
    if start is not None and end is not None:
        return int((int(start) + int(end)) / 2)
    return 0


def _span_range_ms(span, duration_ms=0, min_window_ms=10000):
    """Return bounded start/end ms for a temporal evidence span."""
    center = _span_timestamp_ms(span)
    start = span.get("start_ms")
    end = span.get("end_ms")
    if start is None or end is None or int(end) <= int(start):
        half = min_window_ms // 2
        start = max(0, center - half)
        end = center + half
    else:
        start = int(start)
        end = int(end)
        if end - start < min_window_ms:
            pad = (min_window_ms - (end - start)) // 2
            start = max(0, start - pad)
            end = end + pad
    if duration_ms:
        end = min(duration_ms, end)
    if end <= start:
        end = start + min_window_ms
    return start, end


def _caption_model_mode_from_layer(layer):
    """Parse caption_<model>_<task_mode> into (model, task_mode)."""
    if not layer or not layer.startswith("caption_"):
        return None, None
    rest = layer[len("caption_"):]
    for task_mode in sorted(CAPTION_TASK_MODES, key=len, reverse=True):
        suffix = f"_{task_mode}"
        if rest.endswith(suffix):
            return rest[:-len(suffix)], task_mode
    return None, None


def _ocr_model_from_layer(layer):
    model, task_mode = _caption_model_mode_from_layer(layer)
    if model and task_mode == "text_focus":
        return model
    return "qwen-small"


def _asr_model_from_layer(layer):
    return "whisper" if layer == "asr_whisper" else "parakeet"


def _evidence_query(span, question=""):
    """Build a compact query from verified span text."""
    text = span.get("source_text") or span.get("matched_text") or question
    words = []
    for word in str(text).lower().split():
        clean = word.strip("?.,!\"'():;[]{}")
        if clean and clean not in STOPWORDS and len(clean) > 2:
            words.append(clean)
    if words:
        return " ".join(words[:6])
    return str(question or text)[:50]


def _span_sort_key(span):
    priority = {"speech": 0, "ocr": 1, "visual": 2}
    return priority.get(span.get("modality"), 9), _span_timestamp_ms(span)


def _primary_evidence_span(spans):
    """Prefer speech for localization, then OCR, then visual evidence."""
    return sorted(spans, key=_span_sort_key)[0] if spans else None


def _construct_evidence_span_trajectory(
    question_data,
    index_data,
    allow_browse=True,
    retry_recovery=False,
    interleaved_recovery=False,
):
    """Construct trajectories directly from V4.1 verified evidence_spans."""
    spans = [
        span for span in question_data.get("evidence_spans", [])
        if span.get("modality") in {"speech", "ocr", "visual"}
    ]
    if not spans:
        return None

    question = question_data["question"]
    answer = _expected_answer_text(question_data)
    if not answer:
        return None

    grounding = question_data.get("grounding", {}) or {}
    video_id = question_data.get("video_id", "")
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)
    available_caption_pairs = _available_caption_pairs(layers)
    available_caption_models = _available_caption_models(layers)

    spans = sorted(spans, key=_span_sort_key)
    primary = _primary_evidence_span(spans)
    source_ms = _span_timestamp_ms(primary)
    messages = [{"role": "user", "content": question}]
    tool_calls_made = []
    strategies = []
    inspected_span_ids = set()

    def span_id(span):
        return (
            span.get("modality"),
            span.get("tool"),
            span.get("layer"),
            span.get("start_ms"),
            span.get("end_ms"),
            span.get("timestamp_ms"),
        )

    def append_tool(name, args, obs, think):
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}],
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append(name)

    def locate_span(span, think_prefix=""):
        modality = span.get("modality")
        query = _evidence_query(span, question)
        if modality == "speech":
            obs = simulate_tool_output("search_transcript", index_data, query=query)
            append_tool(
                "search_transcript",
                {"query": query},
                obs,
                think_prefix or "I need to locate the relevant spoken evidence, so I will search the transcript.",
            )
        elif modality == "ocr":
            obs = simulate_tool_output("search_ocr", index_data, query=query)
            append_tool(
                "search_ocr",
                {"query": query},
                obs,
                think_prefix or "I need to locate the relevant on-screen text, so I will search OCR/text evidence.",
            )
        else:
            start_ms, end_ms = _time_window(_span_timestamp_ms(span), duration_ms, window_ms=120000)
            obs = simulate_tool_output("browse_timeline", index_data, start_ms=start_ms, end_ms=end_ms)
            append_tool(
                "browse_timeline",
                {"start_time": format_time(start_ms), "end_time": format_time(end_ms)},
                obs,
                think_prefix or "I need visual context, so I will browse the surrounding time range first.",
            )

    def inspect_span(span, think_prefix="", override_ms=None, override_range=None, mark_seen=True):
        modality = span.get("modality")
        layer = span.get("layer")
        if modality == "speech":
            if override_range:
                start_ms, end_ms = override_range
            else:
                start_ms, end_ms = _span_range_ms(span, duration_ms)
            model = _asr_model_from_layer(layer)
            obs = simulate_tool_output(
                "run_asr", index_data,
                model=model, start_ms=start_ms, end_ms=end_ms,
            )
            append_tool(
                "run_asr",
                {"start_time": format_time(start_ms), "end_time": format_time(end_ms), "model": model},
                obs,
                think_prefix or "The locator points to this time range. I will inspect the transcript there.",
            )
        elif modality == "ocr":
            timestamp_ms = _span_timestamp_ms(span) if override_ms is None else override_ms
            model = _ocr_model_from_layer(layer)
            obs = simulate_tool_output("extract_text", index_data, timestamp_ms, model=model)
            append_tool(
                "extract_text",
                {"timestamp": format_time(timestamp_ms), "model": model},
                obs,
                think_prefix or "I will read the on-screen text at this timestamp.",
            )
        else:
            timestamp_ms = _span_timestamp_ms(span) if override_ms is None else override_ms
            model, task_mode = _caption_model_mode_from_layer(layer)
            if model not in VLM_MODELS or task_mode not in CAPTION_TASK_MODES:
                model, task_mode = _pick_caption_model_and_mode(
                    question, grounding, available_caption_models, available_caption_pairs
                )
            obs = simulate_tool_output(
                "caption_frame", index_data,
                timestamp_ms=timestamp_ms, model=model, task_mode=task_mode,
            )
            append_tool(
                "caption_frame",
                {"timestamp": format_time(timestamp_ms), "model": model, "task_mode": task_mode},
                obs,
                think_prefix or "I will inspect the frame with the appropriate visual task mode.",
            )
        if mark_seen:
            inspected_span_ids.add(span_id(span))
        return obs

    def inspect_remaining(skip=None, budget=5):
        skip = skip or set()
        for span in spans:
            if len(tool_calls_made) >= budget:
                break
            sid = span_id(span)
            if sid in inspected_span_ids or sid in skip:
                continue
            inspect_span(
                span,
                "I should gather the other verified modality needed to answer the question.",
            )

    retry_metadata = None
    if interleaved_recovery:
        speech_span = next((span for span in spans if span.get("modality") == "speech"), None)
        intervening_span = next((span for span in spans if span is not speech_span), None)
        if not speech_span or not intervening_span:
            return None
        query = _evidence_query(speech_span, question)
        wrong_center_ms, wrong_source = _choose_wrong_center_ms(
            index_data, _span_timestamp_ms(speech_span), query, ["asr", "asr_whisper"]
        )
        wrong_range = _time_window(wrong_center_ms, duration_ms)
        locate_span(
            speech_span,
            "I need to locate the spoken evidence first, so I will search the transcript.",
        )
        inspect_span(
            speech_span,
            "This candidate range may contain the relevant speech, so I will inspect it first.",
            override_range=wrong_range,
            mark_seen=False,
        )
        inspect_span(
            intervening_span,
            "That ASR range did not resolve the question. I will use another modality to identify the correct moment.",
        )
        inspect_span(
            speech_span,
            "The intervening evidence points to a better segment. I will retry ASR on the verified range.",
        )
        inspect_remaining(budget=5)
        strategies.append("evidence_span_interleaved_recovery")
        retry_metadata = {
            "trajectory_variant": "interleaved_recovery",
            "wrong_location_tool": "run_asr",
            "wrong_location_ms": wrong_center_ms,
            "wrong_location_source": wrong_source,
            "correct_location_ms": _span_timestamp_ms(speech_span),
            "same_tool_retry": True,
            "same_tool_retry_consecutive": False,
            "intervening_tools": [intervening_span.get("tool")],
            "wrong_location_attempts": 1,
        }

    elif retry_recovery:
        target = primary
        query = _evidence_query(target, question)
        preferred_layers = {
            "speech": ["asr", "asr_whisper"],
            "ocr": [f"caption_{model}_text_focus" for model in OCR_MODELS],
            "visual": [f"caption_{model}_{mode}" for model in VLM_MODELS for mode in CAPTION_TASK_MODES],
        }.get(target.get("modality"), ["asr", "asr_whisper"])
        wrong_center_ms, wrong_source = _choose_wrong_center_ms(
            index_data, _span_timestamp_ms(target), query, preferred_layers
        )
        locate_span(target)
        if target.get("modality") == "speech":
            wrong_range = _time_window(wrong_center_ms, duration_ms)
            inspect_span(
                target,
                "This candidate range may contain the evidence, so I will inspect it first.",
                override_range=wrong_range,
                mark_seen=False,
            )
        else:
            inspect_span(
                target,
                "This candidate timestamp may contain the evidence, so I will inspect it first.",
                override_ms=wrong_center_ms,
                mark_seen=False,
            )
        inspect_span(
            target,
            "That result did not resolve the question. I will retry the same tool at the verified evidence span.",
        )
        inspect_remaining(budget=5)
        strategies.append(f"evidence_span_retry_{target.get('tool')}")
        retry_metadata = {
            "trajectory_variant": "retry_recovery",
            "wrong_location_tool": target.get("tool"),
            "wrong_location_ms": wrong_center_ms,
            "wrong_location_source": wrong_source,
            "correct_location_ms": _span_timestamp_ms(target),
            "same_tool_retry": True,
            "same_tool_retry_consecutive": True,
            "wrong_location_attempts": 1,
        }

    else:
        locate_span(primary)
        inspect_span(primary)
        inspect_remaining(budget=5)
        strategies.append("evidence_span_standard")

    if not tool_calls_made:
        return None

    reasoning = grounding.get("reasoning", "Based on the verified tool evidence gathered above.")
    messages.append({
        "role": "assistant",
        "content": f"<think>\n{reasoning}\n</think>\n\n{answer}",
    })

    return {
        "messages": messages,
        "tools": get_tool_schemas(allow_browse),
        "tool_calls": tool_calls_made,
        "strategies": strategies,
        "metadata": {
            "question_id": question_data.get("id", ""),
            "video_id": video_id,
            "source_time": format_time(source_ms),
            "evidence_span_layers": [span.get("layer") for span in spans],
            "evidence_span_modalities": [span.get("modality") for span in spans],
            "trajectory_variant": (
                retry_metadata.get("trajectory_variant", "retry_recovery")
                if retry_metadata else "standard"
            ),
            "retry_recovery": retry_metadata,
            "source": "v4_1_evidence_spans",
        },
    }


def _derive_v6_grounding(question_data, index_data):
    """Adapt a v6 free-text benchmark row to the legacy trajectory fields.

    v6 rows carry {answer, source_segment_times} where source_segment_times is
    the (wide) generation window, not verified evidence spans. Locate the best
    ASR segment inside the window by word overlap with the answer (then the
    question), and use it as the speech anchor:
    - source_time: "M:SS" of the best-matching ASR segment
    - grounding.speech_excerpt: that segment's text

    Mutates question_data in place. No-op if legacy fields already present.
    """
    if question_data.get("evidence_spans") or question_data.get("source_time"):
        return
    spans = question_data.get("source_segment_times") or []
    windows = []
    for s in spans:
        try:
            windows.append((int(s.get("start_ms", 0)), int(s.get("end_ms", 0))))
        except (TypeError, ValueError):
            continue
    answer_words = _content_words(_expected_answer_text(question_data))
    question_words = _content_words(question_data.get("question", ""))
    layers = index_data.get("layers", {})

    best_key, best_item = (0, 0), None
    for layer_name in ("asr", "asr_whisper"):
        for item in layers.get(layer_name, {}).get("items", []):
            sm = item.get("start_ms", 0)
            if windows and not any(ws <= sm <= we for ws, we in windows):
                continue
            words = _content_words(item.get("text", ""))
            key = (len(answer_words & words), len(question_words & words))
            if key > best_key:
                best_key, best_item = key, item

    if best_item is not None:
        anchor_ms = int(best_item.get("start_ms", 0))
        excerpt = (best_item.get("text") or "").strip()
    elif windows:
        ws, we = windows[0]
        anchor_ms = (ws + we) // 2
        excerpt = ""
    else:
        return

    question_data["source_time"] = format_time(anchor_ms)
    grounding = question_data.setdefault("grounding", {})
    if excerpt and not grounding.get("speech_excerpt"):
        grounding["speech_excerpt"] = excerpt


def construct_trajectory(question_data, index_data, inject_noise=True, allow_browse=True,
                         retry_recovery=False, interleaved_recovery=False):
    """Construct a trajectory using locate-then-inspect pattern.

    Trajectories follow coarse-to-fine localization:
    1. Locate: search_transcript, search_ocr, browse_timeline, or detect_text_scenes
       to find the relevant time region in the video.
    2. Inspect: run_asr, caption_frame, or extract_text at the discovered timestamps.
    3. Answer: synthesize evidence into a final answer.

    Strategy distribution:
    - search_transcript lead (~50%): keyword search -> run_asr -> optional visual -> answer
    - browse_timeline lead (~20%): browse chunks -> run_asr -> answer
    - direct visual lead (~15%): caption_frame at discovered timestamp -> answer
    - text detection lead (~15%): detect_text_scenes or search_ocr -> extract_text -> answer
    """
    if question_data.get("evidence_spans"):
        return _construct_evidence_span_trajectory(
            question_data,
            index_data,
            allow_browse=allow_browse,
            retry_recovery=retry_recovery,
            interleaved_recovery=interleaved_recovery,
        )

    question = question_data["question"]
    answer = _expected_answer_text(question_data)
    grounding = question_data.get("grounding", {})
    source_time = question_data.get("source_time", "")
    video_id = question_data.get("video_id", "")
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    # Parse source time (used for simulating tool outputs, NOT for direct tool calls)
    source_ms = 0
    m = re.match(r'(\d+):(\d+)', source_time or "")
    has_source_time = bool(m)
    if m:
        source_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

    # Determine which tools produced the evidence
    speech = grounding.get("speech_excerpt")
    visual = grounding.get("visual_reference")
    ocr = grounding.get("ocr_reference")

    messages = [{"role": "user", "content": question}]
    tool_calls_made = []
    strategies = []

    # Determine which models have real pre-computed data for this video
    available_caption_pairs = _available_caption_pairs(layers)
    available_caption_models = _available_caption_models(layers)
    available_asr_models = _available_asr_models(layers)
    q_lower = question.lower()

    # Build a time range around source_ms for inspect tools
    # The model "discovers" this through search results, not by knowing source_ms
    inspect_start_ms = max(0, source_ms - 30000)
    inspect_end_ms = min(duration_ms, source_ms + 30000) if duration_ms else source_ms + 30000
    inspect_start_ts = format_time(inspect_start_ms)
    inspect_end_ts = format_time(inspect_end_ms)
    source_ts = format_time(source_ms)

    # Classify question type
    visual_question = any(h in q_lower for h in [
        "shown on screen", "visible", "wearing", "glasses", "hair",
        "shirt", "holding", "standing", "sitting", "facial",
        "what does", "what kind of", "what type of", "what color",
        "what brand", "sign", "logo", "graphic", "display",
    ])
    text_question = any(h in q_lower for h in [
        "chyron", "text on screen", "title", "lower third",
        "on-screen text", "caption", "banner", "headline",
        "written", "label", "credit",
    ])

    # Extract search keywords from grounding
    search_query = _extract_search_keywords(question, grounding)

    # --- Helper functions (locate-then-inspect) ---

    def add_search_transcript(query, think_prefix=""):
        """Step 1: Locate via transcript search."""
        think = think_prefix or f"Let me search the transcript for relevant segments."
        obs = simulate_tool_output("search_transcript", index_data, query=query)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "search_transcript", "arguments": {"query": query}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("search_transcript")

    def add_search_ocr(query, think_prefix=""):
        """Step 1: Locate via OCR search."""
        think = think_prefix or f"Let me search the on-screen text for relevant items."
        obs = simulate_tool_output("search_ocr", index_data, query=query)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "search_ocr", "arguments": {"query": query}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("search_ocr")

    def add_detect_text_scenes(think_prefix=""):
        """Step 1: Locate text-bearing scenes."""
        think = think_prefix or "Let me find frames that contain on-screen text."
        obs = simulate_tool_output("detect_text_scenes", index_data)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "detect_text_scenes", "arguments": {}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("detect_text_scenes")

    def add_browse_timeline(start_ts, end_ts, s_ms, e_ms, think_prefix=""):
        """Step 1: Locate via timeline browsing."""
        if not allow_browse:
            if speech:
                add_search_transcript(search_query, "I need to locate the relevant section, so I will search the transcript first.")
            elif ocr:
                add_search_ocr(search_query, "I need to locate the relevant on-screen text, so I will search OCR first.")
            else:
                add_caption(source_ms, "I need visual evidence, so I will inspect a candidate frame directly.")
            return
        think = think_prefix or f"Let me browse what happens from {start_ts} to {end_ts}."
        obs = simulate_tool_output("browse_timeline", index_data,
                                   start_ms=s_ms, end_ms=e_ms)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "browse_timeline", "arguments": {"start_time": start_ts, "end_time": end_ts}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("browse_timeline")

    def add_run_asr(start_ts, end_ts, s_ms, e_ms, think_prefix=""):
        """Step 2: Inspect speech in a discovered time range."""
        asr_model = _pick_asr_model(grounding, available_asr_models)
        think = think_prefix or f"The search results point to {start_ts}-{end_ts}. Let me get the full transcript there."
        if asr_model == "whisper":
            think += " I'll use whisper since the audio may be noisy."
        obs = simulate_tool_output("run_asr", index_data,
                                   model=asr_model, start_ms=s_ms, end_ms=e_ms)
        args = {"start_time": start_ts, "end_time": end_ts, "model": asr_model}
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "run_asr", "arguments": args}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("run_asr")

    def add_caption(ts_ms, think_prefix=""):
        """Step 2/3: Inspect a frame visually."""
        caption_model, task_mode = _pick_caption_model_and_mode(
            question,
            grounding,
            available_caption_models,
            available_caption_pairs,
        )
        ts = format_time(ts_ms)
        think = think_prefix or f"Let me look at the frame at {ts}."
        if task_mode == "text_focus":
            think += " I'll use text_focus mode to read on-screen text."
        if caption_model == "qwen-8b":
            think += " Using qwen-8b for detailed output."
        obs = simulate_tool_output("caption_frame", index_data, ts_ms,
                                   model=caption_model, task_mode=task_mode)
        args = {"timestamp": ts, "model": caption_model, "task_mode": task_mode}
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "caption_frame", "arguments": args}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("caption_frame")

    def add_extract_text(ts_ms, think_prefix=""):
        """Step 2/3: Inspect on-screen text at a timestamp."""
        ocr_model = _pick_ocr_model(grounding)
        ts = format_time(ts_ms)
        think = think_prefix or f"Let me read the on-screen text at {ts}."
        if ocr_model == "qwen-8b":
            think += " Could be complex text, using qwen-8b."
        else:
            think += " qwen-small should handle this."
        obs = simulate_tool_output("extract_text", index_data, ts_ms, model=ocr_model)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "extract_text", "arguments": {"timestamp": ts, "model": ocr_model}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("extract_text")

    def add_identify_speakers(start_ts, end_ts, s_ms, e_ms, think_prefix=""):
        """Inspect speakers in a time range."""
        think = think_prefix or f"Let me check who is speaking from {start_ts} to {end_ts}."
        obs = simulate_tool_output("identify_speakers", index_data,
                                   start_ms=s_ms, end_ms=e_ms)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "identify_speakers", "arguments": {"start_time": start_ts, "end_time": end_ts}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("identify_speakers")

    # === Recovery trajectories ===
    #
    # These deliberately inspect a plausible but wrong time range first, then
    # recover. Consecutive retry examples immediately reuse the same tool.
    # Interleaved examples use another modality between same-tool attempts,
    # e.g. run_asr(wrong range) -> OCR/chyron -> run_asr(new range). Both teach
    # the policy not to stop after an unhelpful result and provide a natural
    # efficiency signal: the recovered answer is correct, but it required extra
    # inspection calls.
    retry_metadata = None
    if interleaved_recovery and has_source_time and speech:
        asr_model = _pick_asr_model(grounding, available_asr_models)
        correct_obs = simulate_tool_output(
            "run_asr", index_data, model=asr_model,
            start_ms=inspect_start_ms, end_ms=inspect_end_ms,
        )
        if _is_unhelpful_tool_output(correct_obs):
            return None

        wrong_center_ms, wrong_source = _choose_wrong_center_ms(
            index_data, source_ms, search_query, ["asr", "asr_whisper"]
        )
        wrong_start_ms, wrong_end_ms = _time_window(wrong_center_ms, duration_ms)

        ocr_query = _extract_search_keywords(
            question,
            {"ocr_reference": grounding.get("ocr_reference", ""), "speech_excerpt": ""},
        )
        ocr_model = _pick_ocr_model(grounding)
        search_ocr_obs = simulate_tool_output("search_ocr", index_data, query=ocr_query)
        detect_text_obs = simulate_tool_output("detect_text_scenes", index_data)
        extract_text_obs = simulate_tool_output(
            "extract_text", index_data, source_ms, model=ocr_model
        )
        caption_model, task_mode = _pick_caption_model_and_mode(
            question,
            grounding,
            available_caption_models,
            available_caption_pairs,
        )
        caption_obs = simulate_tool_output(
            "caption_frame", index_data, source_ms,
            model=caption_model, task_mode=task_mode,
        )

        add_search_transcript(
            search_query,
            "I need to locate the relevant spoken segment, so I will search the transcript first.",
        )
        add_run_asr(
            format_time(wrong_start_ms),
            format_time(wrong_end_ms),
            wrong_start_ms,
            wrong_end_ms,
            "This candidate segment may identify the guest or context, so I will inspect its ASR first.",
        )

        if not _is_unhelpful_tool_output(search_ocr_obs) and not _is_unhelpful_tool_output(extract_text_obs):
            add_search_ocr(
                ocr_query,
                "That ASR range did not resolve the identity. I should check on-screen text such as chyrons.",
            )
            add_extract_text(
                source_ms,
                "The OCR result points to a better candidate moment, so I will read the text there.",
            )
            intervening_tools = ["search_ocr", "extract_text"]
            interleave_source = "search_ocr"
        elif not _is_unhelpful_tool_output(detect_text_obs) and not _is_unhelpful_tool_output(extract_text_obs):
            add_detect_text_scenes(
                "That ASR range did not resolve the identity. I should look for text scenes such as chyrons or credits.",
            )
            add_extract_text(
                source_ms,
                "A text scene at this candidate moment can identify who is on screen.",
            )
            intervening_tools = ["detect_text_scenes", "extract_text"]
            interleave_source = "detect_text_scenes"
        elif not _is_unhelpful_tool_output(caption_obs):
            add_caption(
                source_ms,
                "That ASR range did not resolve the question. I should inspect the visual evidence at another candidate moment.",
            )
            intervening_tools = ["caption_frame"]
            interleave_source = "caption_frame"
        else:
            return None

        add_run_asr(
            inspect_start_ts,
            inspect_end_ts,
            inspect_start_ms,
            inspect_end_ms,
            "The intervening visual/text evidence points to a different segment. I should retry ASR on that range.",
        )
        strategies.append("interleaved_recovery_asr")
        retry_metadata = {
            "trajectory_variant": "interleaved_recovery",
            "wrong_location_tool": "run_asr",
            "wrong_location_ms": wrong_center_ms,
            "wrong_location_source": wrong_source,
            "correct_location_ms": source_ms,
            "same_tool_retry": True,
            "same_tool_retry_consecutive": False,
            "intervening_tools": intervening_tools,
            "interleave_source": interleave_source,
            "wrong_location_attempts": 1,
        }

    elif retry_recovery and has_source_time:
        if text_question:
            ocr_model = _pick_ocr_model(grounding)
            correct_obs = simulate_tool_output(
                "extract_text", index_data, source_ms, model=ocr_model
            )
            if _is_unhelpful_tool_output(correct_obs):
                return None
            ocr_layer_names = [n for n in index_data.get("layers", {}) if n.endswith("_text_focus")]
            wrong_ms, wrong_source = _choose_wrong_center_ms(
                index_data, source_ms, search_query, ocr_layer_names + ["scenes", "asr", "asr_whisper"]
            )
            add_search_ocr(search_query, "The question asks about on-screen text. I should search OCR first.")
            add_extract_text(
                wrong_ms,
                "This timestamp could contain the relevant text, so I will read it first.",
            )
            add_extract_text(
                source_ms,
                "That text did not resolve the question. I should retry the same OCR tool at another candidate time.",
            )
            strategies.append("retry_recovery_extract_text")
            retry_metadata = {
                "trajectory_variant": "retry_recovery",
                "wrong_location_tool": "extract_text",
                "wrong_location_ms": wrong_ms,
                "wrong_location_source": wrong_source,
                "correct_location_ms": source_ms,
                "same_tool_retry": True,
                "same_tool_retry_consecutive": True,
                "wrong_location_attempts": 1,
            }

        elif visual_question and source_ms:
            caption_model, task_mode = _pick_caption_model_and_mode(
                question,
                grounding,
                available_caption_models,
                available_caption_pairs,
            )
            correct_obs = simulate_tool_output(
                "caption_frame", index_data, source_ms,
                model=caption_model, task_mode=task_mode,
            )
            if _is_unhelpful_tool_output(correct_obs):
                return None
            wrong_ms, wrong_source = _choose_wrong_center_ms(
                index_data, source_ms, search_query,
                ["caption_smolvlm_general_scene", "caption_qwen-small_general_scene",
                 "caption_qwen-8b_general_scene", "asr", "asr_whisper"],
            )
            if speech:
                add_search_transcript(search_query, "I need to locate where this visual evidence appears.")
            elif ocr:
                add_search_ocr(search_query, "On-screen text may locate the relevant moment.")
            add_caption(
                wrong_ms,
                "This candidate timestamp may show the relevant visual evidence, so I will inspect it.",
            )
            add_caption(
                source_ms,
                "That frame did not answer the question. I should retry the same visual tool at another candidate time.",
            )
            strategies.append("retry_recovery_caption_frame")
            retry_metadata = {
                "trajectory_variant": "retry_recovery",
                "wrong_location_tool": "caption_frame",
                "wrong_location_ms": wrong_ms,
                "wrong_location_source": wrong_source,
                "correct_location_ms": source_ms,
                "same_tool_retry": True,
                "same_tool_retry_consecutive": True,
                "wrong_location_attempts": 1,
            }

        elif speech:
            asr_model = _pick_asr_model(grounding, available_asr_models)
            correct_obs = simulate_tool_output(
                "run_asr", index_data, model=asr_model,
                start_ms=inspect_start_ms, end_ms=inspect_end_ms,
            )
            if _is_unhelpful_tool_output(correct_obs):
                return None
            wrong_center_ms, wrong_source = _choose_wrong_center_ms(
                index_data, source_ms, search_query, ["asr", "asr_whisper"]
            )
            wrong_start_ms, wrong_end_ms = _time_window(wrong_center_ms, duration_ms)
            add_search_transcript(search_query, "I need to locate where this topic is discussed.")
            add_run_asr(
                format_time(wrong_start_ms),
                format_time(wrong_end_ms),
                wrong_start_ms,
                wrong_end_ms,
                "This search hit could be relevant, so I will inspect that transcript range first.",
            )
            add_run_asr(
                inspect_start_ts,
                inspect_end_ts,
                inspect_start_ms,
                inspect_end_ms,
                "That range did not contain the needed evidence. I should retry the same ASR tool on another candidate range.",
            )
            strategies.append("retry_recovery_run_asr")
            retry_metadata = {
                "trajectory_variant": "retry_recovery",
                "wrong_location_tool": "run_asr",
                "wrong_location_ms": wrong_center_ms,
                "wrong_location_source": wrong_source,
                "correct_location_ms": source_ms,
                "same_tool_retry": True,
                "same_tool_retry_consecutive": True,
                "wrong_location_attempts": 1,
            }

        if retry_metadata is not None:
            inject_noise = False

    # === Strategy selection (locate-then-inspect) ===

    roll = random.random()

    if retry_metadata is not None:
        pass
    elif text_question:
        # Text detection lead (~15% of overall, but 100% when text_question=True)
        if roll < 0.5 and ocr:
            # search_ocr -> extract_text
            add_search_ocr(search_query, "The question asks about on-screen text. Let me search OCR data.")
            add_extract_text(source_ms, "Found relevant text. Let me read the full content.")
            strategies.append("search_ocr_lead")
        elif roll < 0.8:
            # detect_text_scenes -> extract_text
            think = "The question asks about on-screen text. Let me find text scenes first."
            obs = simulate_tool_output("detect_text_scenes", index_data)
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>",
                "tool_calls": [{"type": "function", "function": {"name": "detect_text_scenes", "arguments": {}}}]
            })
            messages.append({"role": "tool", "content": obs})
            tool_calls_made.append("detect_text_scenes")
            add_extract_text(source_ms, "Found text scenes. Let me read the text at the relevant timestamp.")
            strategies.append("text_detect_lead")
        else:
            # search_transcript -> extract_text (cross-modal: find time via speech, read text)
            add_search_transcript(search_query, "Let me find when this topic comes up in the transcript.")
            add_extract_text(source_ms, "The transcript points to this time. Let me check for on-screen text.")
            strategies.append("search_transcript_to_text")
        # Optionally add ASR context
        if speech and random.random() < 0.3:
            add_run_asr(inspect_start_ts, inspect_end_ts, inspect_start_ms, inspect_end_ms,
                        "Let me also get the speech context around this point.")

    elif visual_question and source_ms:
        # Direct visual lead (~15%)
        if roll < 0.6 and speech:
            # search_transcript -> caption_frame
            add_search_transcript(search_query, "Let me find when this topic comes up.")
            add_caption(source_ms, "The search results point to this time. Let me look at the frame.")
            strategies.append("search_to_visual")
        else:
            # browse_timeline -> caption_frame
            add_browse_timeline(inspect_start_ts, inspect_end_ts, inspect_start_ms, inspect_end_ms,
                                "Let me browse the video to find the relevant section.")
            add_caption(source_ms, "This section looks relevant. Let me examine the frame.")
            strategies.append("browse_to_visual" if allow_browse else "search_fallback_to_visual")

    elif speech:
        # search_transcript lead (~50%) -- the highest-value pattern
        if roll < 0.7:
            # search_transcript -> run_asr -> optional visual
            add_search_transcript(search_query, "I need to find where this topic is discussed.")
            add_run_asr(inspect_start_ts, inspect_end_ts, inspect_start_ms, inspect_end_ms,
                        "Found relevant segments. Let me get the full transcript in this range.")
            strategies.append("search_transcript_lead")
            # Sometimes follow up with visual inspection
            if (ocr or visual) and source_ms and random.random() < 0.4:
                if ocr:
                    add_extract_text(source_ms, "Let me also check for on-screen text at this point.")
                else:
                    add_caption(source_ms, "Let me also look at what's on screen.")
        elif roll < 0.9:
            # browse_timeline lead (~20%)
            # Pick a chunk of the video to browse
            browse_start = max(0, source_ms - 150000)  # ~2.5 min before
            browse_end = min(duration_ms, source_ms + 150000) if duration_ms else source_ms + 150000
            browse_start_ts = format_time(browse_start)
            browse_end_ts = format_time(browse_end)
            add_browse_timeline(browse_start_ts, browse_end_ts, browse_start, browse_end,
                                "Let me browse a section of the video to orient myself.")
            add_run_asr(inspect_start_ts, inspect_end_ts, inspect_start_ms, inspect_end_ms,
                        "This section looks relevant. Let me get the detailed transcript.")
            strategies.append("browse_timeline_lead" if allow_browse else "search_fallback_to_asr")
        else:
            # search_transcript -> identify_speakers
            add_search_transcript(search_query, "Let me search for this topic in the transcript.")
            add_identify_speakers(inspect_start_ts, inspect_end_ts, inspect_start_ms, inspect_end_ms,
                                  "Let me check who is speaking in this section.")
            strategies.append("search_to_speakers")

    else:
        # No clear signal -- browse then inspect
        if source_ms and duration_ms:
            browse_start = max(0, source_ms - 150000)
            browse_end = min(duration_ms, source_ms + 150000)
            add_browse_timeline(format_time(browse_start), format_time(browse_end),
                                browse_start, browse_end,
                                "Let me explore the video to find relevant content.")
            add_caption(source_ms, "Let me look at a frame in this section.")
            strategies.append("explore_browse" if allow_browse else "explore_search_fallback")
        elif source_ms:
            add_caption(source_ms, "Let me start by looking at the video.")
            strategies.append("explore_visual")
        else:
            # No source_ms at all -- search with question keywords
            add_search_transcript(search_query, "Let me search the transcript.")
            strategies.append("explore_search")

    # Noise: sometimes inject a failed tool call that the agent recovers from
    if inject_noise and random.random() < 0.15 and len(tool_calls_made) >= 1:
        wrong_ms = random.randint(0, duration_ms) if duration_ms > 0 else 0
        wrong_ts = format_time(wrong_ms)
        think = f"Let me also check for text at {wrong_ts}."
        obs = simulate_tool_output("extract_text", index_data, wrong_ms, model="qwen-small")
        if "No text" in obs or "No readable" in obs:
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>",
                "tool_calls": [{"type": "function", "function": {"name": "extract_text", "arguments": {"timestamp": wrong_ts, "model": "qwen-small"}}}]
            })
            messages.append({"role": "tool", "content": obs})
            tool_calls_made.append("extract_text_failed")
            strategies.append("noise_recovery")

    # Final answer
    if not tool_calls_made:
        return None

    reasoning = grounding.get("reasoning", "Based on the tool outputs gathered above.")
    messages.append({
        "role": "assistant",
        "content": f"<think>\n{reasoning}\n</think>\n\n{answer}"
    })

    return {
        "messages": messages,
        "tools": get_tool_schemas(allow_browse),
        "tool_calls": tool_calls_made,
        "strategies": strategies,
        "metadata": {
            "question_id": question_data.get("id", ""),
            "video_id": video_id,
            "source_time": source_time,
            "trajectory_variant": (
                retry_metadata.get("trajectory_variant", "retry_recovery")
                if retry_metadata else "standard"
            ),
            "retry_recovery": retry_metadata,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--targeted-evidence-tools", action="store_true",
                        help="Deprecated compatibility flag. All tools are exposed in current trajectory generation.")
    parser.add_argument("--no-browse", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--trajectories-per-question", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions for smoke tests.")
    parser.add_argument("--retry-trajectories-per-question", type=int, default=0,
                        help="Additional trajectories per question with a wrong-location same-tool retry.")
    parser.add_argument("--interleaved-retry-trajectories-per-question", type=int, default=0,
                        help="Additional trajectories with wrong ASR, intervening visual/text evidence, then ASR retry.")
    args = parser.parse_args()
    if args.targeted_evidence_tools or args.no_browse:
        print("NOTE: --targeted-evidence-tools/--no-browse is ignored; all tools are exposed.")

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Constructing tool execution trajectories for {len(questions)} questions...")
    stats = defaultdict(int)
    tool_counts = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            # v6 rows: retrieval_set answers are segment lists, not text --
            # they cannot supervise an answer-emitting policy trajectory.
            fmt = (q.get("format") or "").lower().replace("_", "")
            if fmt == "retrievalset":
                stats["skipped_retrieval_set"] += 1
                continue

            idx = indexes.get(q.get("video_id", ""))
            if not idx:
                stats["no_index"] += 1
                continue

            # v6 rows have no evidence_spans/grounding/source_time; derive a
            # speech anchor from the source_segment_times window.
            if q.get("source_segment_times"):
                _derive_v6_grounding(q, idx)

            for t in range(args.trajectories_per_question):
                result = construct_trajectory(
                    q,
                    idx,
                    inject_noise=not args.no_noise,
                    allow_browse=True,
                )
                if result is None:
                    stats["no_trajectory"] += 1
                    continue

                import hashlib
                vid_hash = hashlib.md5(q["video_id"].encode()).hexdigest()[:8]
                result["metadata"]["trajectory_id"] = f"traj-{vid_hash}-{i:04d}-{t}"

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                stats["total"] += 1

                for tc in result["tool_calls"]:
                    tool_counts[tc] += 1

            for t in range(args.retry_trajectories_per_question):
                result = construct_trajectory(
                    q,
                    idx,
                    inject_noise=False,
                    allow_browse=True,
                    retry_recovery=True,
                )
                if result is None:
                    stats["no_trajectory"] += 1
                    continue
                if result["metadata"].get("trajectory_variant") != "retry_recovery":
                    stats["no_retry_trajectory"] += 1
                    continue

                import hashlib
                vid_hash = hashlib.md5(q["video_id"].encode()).hexdigest()[:8]
                result["metadata"]["trajectory_id"] = f"traj-{vid_hash}-{i:04d}-retry-{t}"

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                stats["total"] += 1

                for tc in result["tool_calls"]:
                    tool_counts[tc] += 1

            for t in range(args.interleaved_retry_trajectories_per_question):
                result = construct_trajectory(
                    q,
                    idx,
                    inject_noise=False,
                    allow_browse=True,
                    interleaved_recovery=True,
                )
                if result is None:
                    stats["no_trajectory"] += 1
                    continue
                if result["metadata"].get("trajectory_variant") != "interleaved_recovery":
                    stats["no_interleaved_trajectory"] += 1
                    continue

                import hashlib
                vid_hash = hashlib.md5(q["video_id"].encode()).hexdigest()[:8]
                result["metadata"]["trajectory_id"] = f"traj-{vid_hash}-{i:04d}-interleaved-{t}"

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()
                stats["total"] += 1

                for tc in result["tool_calls"]:
                    tool_counts[tc] += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(questions)}] trajectories: {stats['total']}")

    print(f"\nDone.")
    print(f"Total: {stats['total']}, No index: {stats['no_index']}, No trajectory: {stats['no_trajectory']}, "
          f"Skipped retrieval_set: {stats['skipped_retrieval_set']}")
    print(f"Tool usage: {dict(sorted(tool_counts.items(), key=lambda x: -x[1]))}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
