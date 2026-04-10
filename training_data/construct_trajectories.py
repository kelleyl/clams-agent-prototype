#!/usr/bin/env python3
"""Construct expert tool-use trajectories from Gemini grounding data.

Instead of asking an LLM to figure out how to search for evidence,
we use Gemini's grounding (speech excerpts, visual references, OCR text)
to deterministically construct the optimal tool-use trajectory:

1. Map each grounding excerpt to its actual index item (by substring match)
2. Determine which tools would find each piece of evidence
3. Construct a ReAct-style trajectory showing the ideal search pattern

This produces *expert* trajectories that demonstrate optimal tool use,
rather than recording a baseline model's (often poor) search behavior.

Usage:
    python training_data/construct_trajectories.py \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/trajectories_v3_expert.jsonl
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_trajectories_v2 import execute_tool, format_timestamp


def find_index_item(layers, layer_name, excerpt, max_results=1):
    """Find the index item that contains a given excerpt."""
    if not excerpt:
        return []
    excerpt_lower = excerpt.lower()[:80]  # match on first 80 chars
    items = layers.get(layer_name, {}).get("items", [])
    matches = []
    for item in items:
        text = (item.get("text", "") or "").lower()
        if excerpt_lower in text or text in excerpt_lower:
            matches.append(item)
        elif len(excerpt_lower) > 20:
            # Fuzzy: check if most words overlap
            excerpt_words = set(excerpt_lower.split())
            text_words = set(text.split())
            if len(excerpt_words) > 3 and len(excerpt_words & text_words) >= len(excerpt_words) * 0.6:
                matches.append(item)
    return matches[:max_results]


def parse_source_time(source_time):
    """Parse source_time string like '15:43' to milliseconds."""
    m = re.match(r'(\d+):(\d+)', source_time or "")
    if m:
        return (int(m.group(1)) * 60 + int(m.group(2))) * 1000
    return 0


def construct_trajectory(question_data, index_data):
    """Construct an expert trajectory from Gemini's grounding data.

    Strategy:
    1. If speech_excerpt exists: search_asr with keywords from the excerpt
    2. If ocr_reference exists: search_ocr with keywords
    3. If visual_reference exists and we have a timestamp: get_visual at that time
    4. Combine evidence into answer

    Each tool call uses real execution against the index.
    """
    question = question_data["question"]
    answer = question_data.get("answer", "")
    video_id = question_data.get("video_id", "")
    grounding = question_data.get("grounding", {})
    source_time = question_data.get("source_time", "")
    source_ms = parse_source_time(source_time)

    layers = index_data.get("layers", {})
    duration_ms = index_data.get("duration_ms", 0)

    conversation = [{"role": "user", "content": question}]
    tool_calls = []
    observations = []

    # Step 1: Search ASR if we have a speech excerpt
    speech = grounding.get("speech_excerpt")
    asr_time_ms = None
    if speech:
        # Extract 2-3 key content words for the search query
        speech_words = speech.lower().split()
        stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
                 "to", "of", "and", "or", "for", "with", "that", "this", "it",
                 "he", "she", "they", "we", "i", "you", "not", "but", "have",
                 "has", "had", "be", "been", "do", "does", "did", "will", "would",
                 "can", "could", "should", "may", "might", "so", "very", "just",
                 "about", "than", "then", "there", "here", "now", "also", "as"}
        content_words = [w for w in speech_words if w not in stops and len(w) > 2][:4]
        query = " ".join(content_words)

        if query:
            tool_call = f'search_asr(video_id="{video_id}", query="{query}")'
            observation = execute_tool(tool_call, index_data)
            tool_calls.append(tool_call)
            observations.append(observation)

            # Find the timestamp from the observation
            time_match = re.search(r'\[(\d+):(\d+)', observation)
            if time_match:
                asr_time_ms = (int(time_match.group(1)) * 60 + int(time_match.group(2))) * 1000

            think = f"I need to find information about this topic. Let me search the speech transcripts."
            conversation.append({"role": "assistant", "content": f"<think>{think}</think>\n<tool>{tool_call}</tool>"})
            conversation.append({"role": "tool", "content": observation})

    # Step 2: Search OCR if we have on-screen text evidence
    ocr = grounding.get("ocr_reference")
    if ocr:
        ocr_words = [w for w in ocr.lower().split() if len(w) > 2][:3]
        query = " ".join(ocr_words)

        if query:
            tool_call = f'search_ocr(video_id="{video_id}", query="{query}")'
            observation = execute_tool(tool_call, index_data)
            tool_calls.append(tool_call)
            observations.append(observation)

            think = "Let me check the on-screen text for additional evidence."
            if asr_time_ms:
                think = f"I found relevant speech content. Now let me check on-screen text around that time for identification."
            conversation.append({"role": "assistant", "content": f"<think>{think}</think>\n<tool>{tool_call}</tool>"})
            conversation.append({"role": "tool", "content": observation})

    # Step 3: Get visual at the relevant time
    visual = grounding.get("visual_reference")
    if visual and not ocr:
        # Use the time from ASR hit or source_time
        time_ms = asr_time_ms or source_ms
        if time_ms > 0:
            start = max(0, time_ms - 5000)
            end = time_ms + 15000
            tool_call = f'get_visual(video_id="{video_id}", start_ms={start}, end_ms={end})'
            observation = execute_tool(tool_call, index_data)
            tool_calls.append(tool_call)
            observations.append(observation)

            think = "Let me check what was shown visually at this time."
            if asr_time_ms:
                think = f"The speech at {asr_time_ms//60000}:{(asr_time_ms//1000)%60:02d} discusses this topic. Let me see what was shown on screen."
            conversation.append({"role": "assistant", "content": f"<think>{think}</think>\n<tool>{tool_call}</tool>"})
            conversation.append({"role": "tool", "content": observation})

    # Step 4: If we still have no tool calls, try a broader search
    if not tool_calls:
        # Use answer terms as search query
        answer_words = [w for w in answer.lower().split() if len(w) > 2][:3]
        if answer_words:
            query = " ".join(answer_words)
            tool_call = f'search_asr(video_id="{video_id}", query="{query}")'
            observation = execute_tool(tool_call, index_data)
            tool_calls.append(tool_call)
            observations.append(observation)

            conversation.append({"role": "assistant", "content": f"<think>Let me search for relevant content.</think>\n<tool>{tool_call}</tool>"})
            conversation.append({"role": "tool", "content": observation})

    # Final answer
    reasoning = grounding.get("reasoning", "Based on the evidence gathered from multiple sources.")
    final_think = f"Based on the evidence I found: {reasoning}"
    conversation.append({"role": "assistant", "content": f"<think>{final_think}</think>\n<answer>{answer}</answer>"})

    # Verification
    has_results = any("No " not in obs[:10] and "Error" not in obs[:10] for obs in observations) if observations else False

    verification = {
        "method": "expert_scripted",
        "tool_calls": [{"call": tc, "has_results": "No " not in obs[:10]} for tc, obs in zip(tool_calls, observations)],
        "has_answer": True,
        "has_tools": len(tool_calls) > 0,
        "tools_found_results": has_results,
        "all_verified": len(tool_calls) > 0 and has_results,
    }

    return conversation, verification, tool_calls


def main():
    parser = argparse.ArgumentParser(description="Construct expert trajectories from Gemini grounding")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, default=Path("training_data/output/trajectories_v3_expert.jsonl"))
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Constructing expert trajectories for {len(questions)} questions...")

    stats = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                continue

            conversation, verification, tools = construct_trajectory(q, idx)

            result = {
                "id": f"traj-expert-{i:04d}",
                "question_id": q.get("id", ""),
                "video_id": video_id,
                "question": q["question"],
                "expected_answer": q.get("answer", ""),
                "format": q.get("format", "free_text"),
                "conversation": conversation,
                "verification": verification,
                "num_turns": len([c for c in conversation if c["role"] == "assistant"]),
                "tools_used": [tc.split("(")[0] for tc in tools],
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            if verification["all_verified"]:
                stats["verified"] += 1
            else:
                stats["unverified"] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(questions)}] verified: {stats['verified']}, "
                      f"unverified: {stats['unverified']}", flush=True)

    print(f"\nDone.")
    print(f"Verified: {stats['verified']}, Unverified: {stats['unverified']}, No index: {stats['no_index']}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
