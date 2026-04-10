#!/usr/bin/env python3
"""Generate tool-orchestration trajectories for CLAMS-Agent training.

Each trajectory shows an agent deciding which CLAMS tools to run on a
RAW VIDEO to answer a question — no pre-built index. The agent must:
1. Analyze the question to determine what information is needed
2. Select appropriate CLAMS tools (ASR, OCR, shot detection, captioning)
3. Choose the right parameters (e.g., swt_transcription vs transnet_shots)
4. Process tool outputs and synthesize an answer

Training examples use SIMULATED tool outputs reconstructed from existing
MMIF pipeline outputs and video indexes.

Usage:
    python training_data/generate_orchestration_trajectories.py \
        --questions qa-data/raw/llm_comprehension_qa.jsonl \
        --mmif-dir data/combined_outputs \
        --index-dir data/video_indexes \
        --output training_data/output/orchestration_trajectories.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mmif_simulator import MMIFOutputSimulator


# Modality → tool sequence mapping
MODALITY_TOOLS = {
    "speech": [
        {"name": "parakeet-wrapper", "parameters": {"modelSize": "0.6b"},
         "reason": "transcribe speech to get what was said"},
    ],
    "ocr": [
        {"name": "swt-detection", "parameters": {},
         "reason": "find text scenes (chyrons, slates, credits)"},
        {"name": "qwen3vl-captioner",
         "parameters": {"config": "config/swt_transcription.yaml", "batchSize": "64"},
         "reason": "read on-screen text from detected text scenes"},
    ],
    "visual": [
        {"name": "transnet-wrapper", "parameters": {},
         "reason": "detect shot boundaries for visual segmentation"},
        {"name": "qwen3vl-captioner",
         "parameters": {"config": "config/transnet_shots.yaml", "batchSize": "64"},
         "reason": "describe visual content of each shot"},
    ],
    "entities": [
        {"name": "spacy-wrapper", "parameters": {},
         "reason": "extract named entities (people, organizations, locations)"},
    ],
}

# Tool dependency graph (tool → must run after)
TOOL_DEPENDENCIES = {
    "qwen3vl-captioner:swt_transcription": ["swt-detection"],
    "qwen3vl-captioner:transnet_shots": ["transnet-wrapper"],
    "spacy-wrapper": ["parakeet-wrapper", "qwen3vl-captioner"],
}

SYSTEM_PROMPT = """You are a CLAMS video analysis agent. Given a question about a video, you must decide which processing tools to run and in what order to find the answer.

Available tools:
- swt-detection: Find text scenes (slates, chyrons, credits) in the video
- parakeet-wrapper: Transcribe speech (ASR) with word-level timestamps
- transnet-wrapper: Detect shot boundaries for visual segmentation
- qwen3vl-captioner: Vision-language model for captioning or OCR
  - config=swt_transcription: Read on-screen text (requires swt-detection first)
  - config=transnet_shots: Describe visual content (requires transnet-wrapper first)
- spacy-wrapper: Named entity recognition on text
- ffmpeg:get_video_info: Get video metadata (duration, resolution)
- ffmpeg:extract_frame: Extract a frame at a specific timestamp

Strategy:
1. Start with ffmpeg:get_video_info to understand the video
2. Run ASR (parakeet-wrapper) if you need speech content
3. Run swt-detection if you need on-screen text (chyrons, titles)
4. Run transnet-wrapper if you need visual descriptions (b-roll, graphics)
5. Run qwen3vl-captioner with the appropriate config after its prerequisite
6. Run spacy-wrapper if you need entity extraction

Be EFFICIENT: only run tools that are needed for this specific question.
Provide your answer with evidence citations from the tool outputs."""


def modalities_to_tool_plan(modalities: list) -> list:
    """Convert modalities_required to an ordered tool plan."""
    tools_needed = []
    tool_names_seen = set()

    # Always start with video info
    tools_needed.append({
        "name": "ffmpeg:get_video_info", "parameters": {},
        "reason": "get video duration and properties",
    })
    tool_names_seen.add("ffmpeg:get_video_info")

    # Add tools for each modality
    for mod in modalities:
        mod_lower = mod.lower()
        if mod_lower in MODALITY_TOOLS:
            for tool in MODALITY_TOOLS[mod_lower]:
                tool_key = tool["name"]
                if "qwen3vl" in tool_key:
                    config = tool["parameters"].get("config", "")
                    tool_key = f"{tool['name']}:{config.split('/')[-1].replace('.yaml','')}"
                if tool_key not in tool_names_seen:
                    tools_needed.append(tool)
                    tool_names_seen.add(tool_key)
        elif mod_lower in ("on_screen_text",):
            # Map alternate names
            for tool in MODALITY_TOOLS["ocr"]:
                tool_key = tool["name"]
                if tool_key not in tool_names_seen:
                    tools_needed.append(tool)
                    tool_names_seen.add(tool_key)

    # Topological sort by dependencies
    return _sort_by_dependencies(tools_needed)


def _sort_by_dependencies(tools: list) -> list:
    """Sort tools respecting dependency order."""
    # Simple: swt-detection before qwen(swt), transnet before qwen(transnet),
    # text-producers before spacy
    priority = {
        "ffmpeg:get_video_info": 0,
        "swt-detection": 1,
        "transnet-wrapper": 1,
        "parakeet-wrapper": 1,
        "whisper-wrapper": 1,
        "qwen3vl-captioner": 2,
        "spacy-wrapper": 3,
    }
    return sorted(tools, key=lambda t: priority.get(t["name"], 5))


def generate_reasoning(question: str, modalities: list, grounding: dict) -> str:
    """Generate the initial reasoning about which tools to use."""
    parts = [f"This question asks about a video and requires information from: {', '.join(modalities)}."]

    if "speech" in modalities:
        parts.append("I need ASR (parakeet-wrapper) to get what was said.")
    if "ocr" in modalities or "on_screen_text" in modalities:
        parts.append("I need to read on-screen text, so I'll run swt-detection to find text scenes, then qwen3vl-captioner with swt_transcription config to OCR them.")
    if "visual" in modalities:
        parts.append("I need visual descriptions, so I'll run transnet-wrapper for shot boundaries, then qwen3vl-captioner with transnet_shots config for captions.")
    if "entities" in modalities:
        parts.append("I need named entity extraction via spacy-wrapper.")

    return " ".join(parts)


def generate_step_reasoning(tool: dict, step_num: int, prev_results: list) -> str:
    """Generate reasoning for a specific tool call step."""
    name = tool["name"]
    reason = tool.get("reason", "")

    if name == "ffmpeg:get_video_info":
        return "Let me start by getting the video metadata to understand what I'm working with."
    elif name == "parakeet-wrapper":
        return f"Now I'll run ASR to {reason}."
    elif name == "swt-detection":
        return f"I'll run scene-with-text detection to {reason}."
    elif name == "transnet-wrapper":
        return f"Running shot detection to {reason}."
    elif name == "qwen3vl-captioner":
        config = tool.get("parameters", {}).get("config", "")
        if "swt" in config:
            return "Now that I have the text scene locations, I'll run the VLM to read the on-screen text."
        else:
            return "With shot boundaries identified, I'll caption the visual content of each shot."
    elif name == "spacy-wrapper":
        return "Running named entity recognition on the extracted text."
    return f"Running {name} to {reason}."


def generate_answer_reasoning(grounding: dict, answer: str) -> str:
    """Generate the final reasoning that synthesizes evidence into an answer."""
    parts = []
    speech = grounding.get("speech_excerpt") or ""
    visual = grounding.get("visual_reference") or ""
    ocr = grounding.get("ocr_reference") or ""
    reasoning = grounding.get("reasoning") or ""

    if speech:
        parts.append(f'The ASR transcript shows: "{speech[:150]}"')
    if ocr:
        parts.append(f'The on-screen text reads: "{ocr[:150]}"')
    if visual:
        parts.append(f'The visual description shows: "{visual[:150]}"')
    if reasoning:
        parts.append(f"Combining this evidence: {reasoning[:200]}")

    if not parts:
        parts.append("Based on the tool outputs, I can answer this question.")

    return " ".join(parts)


def build_trajectory(question_data: dict, simulator: MMIFOutputSimulator) -> dict:
    """Build a complete orchestration trajectory for one question."""
    video_id = question_data.get("video_id", "")
    question = question_data["question"]
    answer = question_data.get("answer", "")
    if isinstance(answer, dict):
        answer = json.dumps(answer)
    modalities = question_data.get("modalities_required", ["speech", "visual"])
    grounding = question_data.get("grounding", {})
    source_times = question_data.get("source_segment_times", [])

    # Get time range for focused tool outputs
    time_range = None
    if source_times:
        start = min(t["start_ms"] for t in source_times)
        end = max(t["end_ms"] for t in source_times)
        # Expand slightly for context
        time_range = (max(0, start - 30000), end + 30000)

    # Determine tool plan
    tool_plan = modalities_to_tool_plan(modalities)

    # Build conversation
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Video: {video_id}.mp4\nQuestion: {question}"},
    ]

    # Initial reasoning
    initial_think = generate_reasoning(question, modalities, grounding)
    first_tool = tool_plan[0]
    conversation.append({
        "role": "assistant",
        "content": (
            f"<think>{initial_think} Let me start by getting the video info.</think>\n"
            f'<tool_call>{{"name": "{first_tool["name"]}", '
            f'"arguments": {json.dumps(first_tool.get("parameters", {}))}}}</tool_call>'
        ),
    })

    # Simulate first tool output
    output = simulator.simulate_tool(video_id, first_tool["name"],
                                      first_tool.get("parameters"), time_range)
    conversation.append({"role": "tool", "name": first_tool["name"], "content": output})

    # Subsequent tools
    for i, tool in enumerate(tool_plan[1:], 2):
        think = generate_step_reasoning(tool, i, tool_plan[:i-1])
        conversation.append({
            "role": "assistant",
            "content": (
                f"<think>{think}</think>\n"
                f'<tool_call>{{"name": "{tool["name"]}", '
                f'"arguments": {json.dumps(tool.get("parameters", {}))}}}</tool_call>'
            ),
        })
        output = simulator.simulate_tool(video_id, tool["name"],
                                          tool.get("parameters"), time_range)
        conversation.append({"role": "tool", "name": tool["name"], "content": output})

    # Final answer with reasoning
    answer_think = generate_answer_reasoning(grounding, answer)
    conversation.append({
        "role": "assistant",
        "content": f"<think>{answer_think}</think>\n<answer>{answer}</answer>",
    })

    # Metadata
    ground_truth_tools = [t["name"] for t in tool_plan]
    optimal_count = len([t for t in tool_plan if t["name"] != "ffmpeg:get_video_info"])

    return {
        "id": f"orch-{question_data.get('id', '')}",
        "video_id": video_id,
        "question": question,
        "answer": answer,
        "modalities_required": modalities,
        "conversation": conversation,
        "ground_truth_tools": ground_truth_tools,
        "tool_count": len(ground_truth_tools),
        "optimal_tool_count": optimal_count,
        "has_time_range_focus": time_range is not None,
        "reasoning_type": question_data.get("reasoning_type", ""),
        "difficulty": question_data.get("difficulty", ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate tool-orchestration trajectories")
    parser.add_argument("--questions", type=Path,
                        default=Path("qa-data/raw/llm_comprehension_qa.jsonl"))
    parser.add_argument("--mmif-dir", type=Path,
                        default=Path("data/combined_outputs"))
    parser.add_argument("--index-dir", type=Path,
                        default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/output/orchestration_trajectories.jsonl"))
    parser.add_argument("--max-questions", type=int, default=None)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    simulator = MMIFOutputSimulator(args.mmif_dir, args.index_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating orchestration trajectories for {len(questions)} questions...")

    tool_usage = defaultdict(int)
    results = []

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")

            trajectory = build_trajectory(q, simulator)
            results.append(trajectory)

            for tool in trajectory["ground_truth_tools"]:
                tool_usage[tool] += 1

            fout.write(json.dumps(trajectory, ensure_ascii=False) + "\n")

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(questions)}] generated", flush=True)

    # Summary
    print(f"\nGenerated {len(results)} orchestration trajectories")
    print(f"Output: {args.output}")

    print(f"\nTool usage across all trajectories:")
    for tool, count in sorted(tool_usage.items(), key=lambda x: -x[1]):
        print(f"  {tool:30s} {count:5d} ({100*count/len(results):.0f}%)")

    avg_tools = sum(t["tool_count"] for t in results) / max(1, len(results))
    avg_turns = sum(len(t["conversation"]) for t in results) / max(1, len(results))
    print(f"\nAvg tools per trajectory: {avg_tools:.1f}")
    print(f"Avg conversation turns: {avg_turns:.1f}")


if __name__ == "__main__":
    main()
