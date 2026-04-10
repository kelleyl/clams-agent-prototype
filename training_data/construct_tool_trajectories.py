#!/usr/bin/env python3
"""Construct trajectories reflecting actual CLAMS tool execution decisions.

Instead of "search the index for keywords", trajectories show the agent
deciding which CLAMS tools to run on the video and processing their outputs.
The index is just a cache of tool outputs -- the agent's job is to decide
what tools to run, not how to search a pre-built index.

Provenance chain: each index item tracks which CLAMS app produced it.
We reverse-engineer the tool decisions that would produce the evidence
needed to answer each question.

Tool palette (matching actual CLAMS apps):
- run_asr(video): Run speech recognition (Parakeet TDT). Returns timestamped transcript.
- detect_text_scenes(video): Run SWT to find frames with text (chyrons, slates, credits).
- extract_text(video, timestamp): Run OCR on a specific frame to read on-screen text.
- caption_frame(video, timestamp): Run VLM captioner on a frame for visual description.
- identify_speakers(video): Run diarization + LLM to identify who speaks when.
- extract_entities(text): Run NER on text to find people, orgs, places.

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

# Tool descriptions for the system prompt
TOOL_DESCRIPTIONS = {
    "run_asr": {
        "description": "Run speech recognition on the video. Returns timestamped transcript of everything said.",
        "cost": "low",
        "accuracy": "3/5",
        "parameters": {},  # processes whole video
    },
    "detect_text_scenes": {
        "description": "Detect frames containing text (chyrons, slates, credits, graphics) using scene classification. Fast but may miss some text.",
        "cost": "low",
        "accuracy": "2/5",
        "parameters": {},
    },
    "extract_text": {
        "description": "Run OCR on a specific frame to read on-screen text. Use after detecting text scenes, or on a specific timestamp.",
        "cost": "medium",
        "accuracy": "4/5",
        "parameters": {"timestamp": "time in the video (e.g., '15:47')"},
    },
    "caption_frame": {
        "description": "Run VLM captioner on a frame for detailed visual description. More expensive than text detection but captures everything visible.",
        "cost": "high",
        "accuracy": "4/5",
        "parameters": {"timestamp": "time in the video (e.g., '15:47')"},
    },
    "identify_speakers": {
        "description": "Run speaker diarization to identify who speaks when. Returns speaker names and their speech segments.",
        "cost": "medium",
        "accuracy": "3/5",
        "parameters": {},
    },
    "extract_entities": {
        "description": "Run named entity recognition on text to find people, organizations, places. Use on ASR or OCR output.",
        "cost": "very low",
        "accuracy": "3/5",
        "parameters": {"text": "the text to analyze"},
    },
}

# Tool schemas for Qwen3.5 native format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Run speech recognition on the video. Returns timestamped transcript.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '0:00'). Optional, defaults to full video."},
                    "end_time": {"type": "string", "description": "End time. Optional."},
                },
                "required": []
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
            "description": "Run OCR on a specific frame to read on-screen text. Use after finding a text scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"}
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "caption_frame",
            "description": "Run VLM captioner on a frame for detailed visual description. More expensive than text detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"}
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_speakers",
            "description": "Run speaker diarization to identify who speaks when.",
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
            "name": "extract_entities",
            "description": "Run NER on text to find people, organizations, places.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Which text to analyze: 'asr' or 'ocr'"}
                },
                "required": ["source"]
            }
        }
    },
]


def format_time(ms):
    """Format milliseconds as MM:SS."""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def get_layer_items_at(layers, layer_name, start_ms, end_ms):
    """Get items from a layer overlapping a time range."""
    items = layers.get(layer_name, {}).get("items", [])
    return [i for i in items if i.get("start_ms", 0) < end_ms and i.get("end_ms", 0) > start_ms]


def simulate_tool_output(tool_name, index_data, timestamp_ms=None, context=None,
                         layer_override=None, model=None, task_mode=None):
    """Simulate what a CLAMS tool would return, using actual index data.

    Args:
        tool_name: Name of the tool to simulate.
        index_data: Video index data dict with "layers" and "duration_ms".
        timestamp_ms: Optional timestamp in milliseconds for time-specific tools.
        context: Optional context dict (e.g., {"start_ms": ..., "end_ms": ...}).
        layer_override: Optional explicit layer name override (e.g., "asr_whisper").
        model: Optional model name controlling output detail level.
            - run_asr: "whisper" uses asr_whisper layer; "parakeet"/None uses default.
            - extract_text: "qwen-8b" truncates lines to 120 chars; "qwen-30b"/None returns full.
            - caption_frame: "smolvlm" returns brief (first sentence, max 80 chars);
              "qwen-8b" returns moderate (max 200 chars); "qwen-30b"/None returns full.
        task_mode: Optional task mode for caption_frame controlling what to extract.
            - "text_focus": pull from OCR layer instead of visual captions.
            - "person_identity": merge visual caption + OCR (name/title on screen).
            - "action_event" / "general_scene" / None: use visual captions (default).

    Default behavior (model=None, task_mode=None) is identical to the original
    implementation for backwards compatibility.
    """
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    if tool_name == "run_asr":
        # model="whisper" selects the whisper ASR layer; explicit layer_override
        # takes priority over model-based selection.
        if layer_override:
            asr_layer = layer_override
        elif model == "whisper":
            asr_layer = "asr_whisper"
        else:
            asr_layer = "asr"
        if timestamp_ms:
            start = max(0, timestamp_ms - 30000)
            end = timestamp_ms + 30000
        else:
            start, end = 0, duration_ms
        items = get_layer_items_at(layers, asr_layer, start, end)
        if not items:
            return "No speech detected in this range."
        lines = []
        for item in items[:5]:
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
        items = get_layer_items_at(layers, "ocr", timestamp_ms - 5000, timestamp_ms + 5000)
        if not items:
            return "No text detected at this timestamp."
        texts = []
        for item in items[:3]:
            label = item.get("scene_label", "")
            text = item.get("text", "")
            if text and not text.startswith("The "):
                texts.append(f"[{label}] {text[:200]}")
        output = "\n".join(texts) if texts else "No readable text at this timestamp."
        # Model-quality scaling for OCR
        if output and not output.startswith("No "):
            if model == "qwen-small":
                lines = output.split("\n")
                output = lines[0][:100] if lines else output
            elif model == "qwen-8b":
                lines = output.split("\n")
                output = "\n".join(line[:120] for line in lines)
        return output

    elif tool_name == "caption_frame":
        if timestamp_ms is None:
            return "Error: timestamp required"

        # Try pre-computed caption variant layer first (from generate_caption_variants.py)
        variant_layer = f"caption_{model}_{task_mode}" if model and task_mode else None
        if variant_layer and variant_layer in layers:
            items = get_layer_items_at(layers, variant_layer,
                                       timestamp_ms - 3000, timestamp_ms + 3000)
            if items:
                return items[0].get("text", "")

        # Fall back to synthetic assembly from canonical layers
        if task_mode == "text_focus":
            ocr_output = _caption_ocr_at(layers, timestamp_ms, model)
            if ocr_output:
                return ocr_output
            caption = _caption_visual_at(layers, timestamp_ms, duration_ms, model)
            return caption or "No visible text at this timestamp."

        elif task_mode == "person_identity":
            caption = _caption_visual_at(layers, timestamp_ms, duration_ms, model)
            ocr_output = _caption_ocr_at(layers, timestamp_ms, model)
            parts = []
            if caption:
                parts.append(caption)
            if ocr_output:
                parts.append("Name/title on screen: " + ocr_output)
            return "\n".join(parts) if parts else "No people clearly visible."

        else:
            # "general_scene", "action_event", or None (default)
            caption = _caption_visual_at(layers, timestamp_ms, duration_ms, model)
            return caption or "No visual description available."

    elif tool_name == "identify_speakers":
        if timestamp_ms:
            items = get_layer_items_at(layers, "speakers", timestamp_ms - 15000, timestamp_ms + 15000)
        else:
            items = layers.get("speakers", {}).get("items", [])[:5]
        if not items:
            return "No speaker information available."
        seen = set()
        lines = []
        for item in items:
            name = item.get("speaker_name", "Unknown")
            if name not in seen:
                seen.add(name)
                t = format_time(item["start_ms"])
                lines.append(f"[{t}] {name}: {item.get('text', '')[:100]}")
        return "\n".join(lines)

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


def _caption_visual_at(layers, timestamp_ms, duration_ms=0, model=None):
    """Get visual caption text at a timestamp, scaled by model quality.

    Args:
        layers: Index layers dict.
        timestamp_ms: Target timestamp in milliseconds.
        duration_ms: Video duration (unused, kept for interface consistency).
        model: VLM model name controlling output detail.
            - "smolvlm": first sentence only, max 80 chars.
            - "qwen-small": brief, max 100 chars (Qwen3.5-0.8B).
            - "qwen-8b": moderate detail, max 200 chars (Qwen3.5-9B).
            - "qwen-30b": full output (Qwen3.5-27B).
            - None: original behavior (first sentence, backwards compatible).

    Returns:
        Caption text string, or empty string if nothing found.
    """
    items = get_layer_items_at(layers, "visual_captions", timestamp_ms - 3000, timestamp_ms + 3000)
    if not items:
        return ""
    text = items[0].get("text", "")
    if not text:
        return ""
    if model == "smolvlm":
        return text.split(". ")[0] + "." if ". " in text else text[:80]
    elif model == "qwen-small":
        return text.split(". ")[0] + "." if ". " in text else text[:100]
    elif model == "qwen-8b":
        return text[:200]
    elif model == "qwen-30b":
        return text
    # model=None: original behavior (first sentence or first 200 chars)
    first_sentence = text.split(". ")[0] + "." if ". " in text else text[:200]
    return first_sentence


def _caption_ocr_at(layers, timestamp_ms, model=None):
    """Get OCR text at a timestamp, scaled by model quality.

    Args:
        layers: Index layers dict.
        timestamp_ms: Target timestamp in milliseconds.
        model: VLM model name controlling output detail.
            - "smolvlm": most prominent text only (first line).
            - "qwen-small": first line, max 100 chars (Qwen3.5-0.8B).
            - "qwen-8b": first 2 lines, truncated to 150 chars each.
            - "qwen-30b" or None: full output.

    Returns:
        OCR text string, or empty string if nothing found.
    """
    items = get_layer_items_at(layers, "ocr", timestamp_ms - 5000, timestamp_ms + 5000)
    if not items:
        return ""
    texts = []
    for item in items[:3]:
        label = item.get("scene_label", "")
        text = item.get("text", "")
        if text and not text.startswith("The "):
            texts.append(f"[{label}] {text[:200]}")
    output = "\n".join(texts) if texts else ""
    if not output:
        return ""
    if model == "smolvlm":
        lines = output.split("\n")
        return lines[0][:80] if lines else ""
    elif model == "qwen-small":
        lines = output.split("\n")
        return lines[0][:100] if lines else ""
    elif model == "qwen-8b":
        lines = output.split("\n")
        return "\n".join(line[:150] for line in lines[:2])
    return output


def construct_trajectory(question_data, index_data, inject_noise=True):
    """Construct a trajectory showing actual CLAMS tool execution decisions."""
    question = question_data["question"]
    answer = question_data.get("answer", "")
    grounding = question_data.get("grounding", {})
    source_time = question_data.get("source_time", "")
    video_id = question_data.get("video_id", "")
    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    # Parse source time
    source_ms = 0
    m = re.match(r'(\d+):(\d+)', source_time or "")
    if m:
        source_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

    # Determine which tools produced the evidence
    speech = grounding.get("speech_excerpt")
    visual = grounding.get("visual_reference")
    ocr = grounding.get("ocr_reference")

    messages = [{"role": "user", "content": question}]
    tool_calls_made = []
    strategies = []

    # Strategy: always start with ASR if speech evidence exists
    if speech:
        think = "I need to understand what's being discussed in this video. Let me run speech recognition."
        obs = simulate_tool_output("run_asr", index_data, source_ms)

        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "run_asr", "arguments": {}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("run_asr")
        strategies.append("asr")

    # If OCR evidence: detect text scenes, then extract text
    if ocr:
        if inject_noise and random.random() < 0.3:
            # Noise: try SWT first, sometimes it doesn't find the right scene
            think = "I see there might be on-screen text. Let me detect text scenes first."
            obs = simulate_tool_output("detect_text_scenes", index_data)
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>",
                "tool_calls": [{"type": "function", "function": {"name": "detect_text_scenes", "arguments": {}}}]
            })
            messages.append({"role": "tool", "content": obs})
            tool_calls_made.append("detect_text_scenes")

            # Then extract text at a detected scene near source time
            if source_ms:
                ts = format_time(source_ms)
                think = f"Text scene detected near {ts}. Let me read what it says."
                obs = simulate_tool_output("extract_text", index_data, source_ms)
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\n{think}\n</think>",
                    "tool_calls": [{"type": "function", "function": {"name": "extract_text", "arguments": {"timestamp": ts}}}]
                })
                messages.append({"role": "tool", "content": obs})
                tool_calls_made.append("extract_text")
                strategies.append("swt+ocr")
        else:
            # Direct OCR at timestamp
            if source_ms:
                ts = format_time(source_ms)
                think = "Based on the speech content, there might be identifying text on screen at this point. Let me extract text from that frame."
                obs = simulate_tool_output("extract_text", index_data, source_ms)
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\n{think}\n</think>",
                    "tool_calls": [{"type": "function", "function": {"name": "extract_text", "arguments": {"timestamp": ts}}}]
                })
                messages.append({"role": "tool", "content": obs})
                tool_calls_made.append("extract_text")
                strategies.append("ocr_direct")

    # If visual evidence but no OCR
    elif visual:
        if source_ms:
            ts = format_time(source_ms)
            think = "I need to see what's visually shown at this point in the video."
            obs = simulate_tool_output("caption_frame", index_data, source_ms)
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>",
                "tool_calls": [{"type": "function", "function": {"name": "caption_frame", "arguments": {"timestamp": ts}}}]
            })
            messages.append({"role": "tool", "content": obs})
            tool_calls_made.append("caption_frame")
            strategies.append("visual")

    # Sometimes add speaker identification
    if inject_noise and random.random() < 0.2 and speech:
        think = "Let me also check who is speaking at this point."
        obs = simulate_tool_output("identify_speakers", index_data, source_ms)
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{think}\n</think>",
            "tool_calls": [{"type": "function", "function": {"name": "identify_speakers", "arguments": {}}}]
        })
        messages.append({"role": "tool", "content": obs})
        tool_calls_made.append("identify_speakers")
        strategies.append("speakers")

    # Noise: sometimes inject a failed tool call that the agent recovers from
    if inject_noise and random.random() < 0.15 and len(tool_calls_made) >= 1:
        # Try to extract text at wrong timestamp
        wrong_ms = random.randint(0, duration_ms) if duration_ms > 0 else 0
        wrong_ts = format_time(wrong_ms)
        think = f"Let me also check for text at {wrong_ts}."
        obs = simulate_tool_output("extract_text", index_data, wrong_ms)

        if "No text" in obs or "No readable" in obs:
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>",
                "tool_calls": [{"type": "function", "function": {"name": "extract_text", "arguments": {"timestamp": wrong_ts}}}]
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
        "tools": TOOL_SCHEMAS,
        "tool_calls": tool_calls_made,
        "strategies": strategies,
        "metadata": {
            "question_id": question_data.get("id", ""),
            "video_id": video_id,
            "source_time": source_time,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--trajectories-per-question", type=int, default=1)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Constructing tool execution trajectories for {len(questions)} questions...")
    stats = defaultdict(int)
    tool_counts = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            idx = indexes.get(q.get("video_id", ""))
            if not idx:
                stats["no_index"] += 1
                continue

            for t in range(args.trajectories_per_question):
                result = construct_trajectory(q, idx, inject_noise=not args.no_noise)
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

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(questions)}] trajectories: {stats['total']}")

    print(f"\nDone.")
    print(f"Total: {stats['total']}, No index: {stats['no_index']}, No trajectory: {stats['no_trajectory']}")
    print(f"Tool usage: {dict(sorted(tool_counts.items(), key=lambda x: -x[1]))}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
