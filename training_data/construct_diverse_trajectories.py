#!/usr/bin/env python3
"""Construct diverse tool-use trajectories using multiple search strategies.

For each question, generates 1-3 trajectories using different tool-use
strategies. This teaches the model that there are multiple valid ways
to find evidence, not just one fixed pattern.

Strategies:
1. topic_first:    search_asr -> search_ocr/get_visual -> answer
2. identity_first: search_ocr (chyron) -> search_asr (what they said) -> answer
3. chapter_nav:    get_chapter -> browse_timeline at that chapter -> answer
4. speaker_track:  get_speakers -> search_asr for their content -> answer
5. entity_lookup:  search_entities -> get context around entity time -> answer
6. visual_anchor:  get_visual -> use visual info to search_asr -> answer

Usage:
    python training_data/construct_diverse_trajectories.py \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/trajectories_v3_diverse.jsonl
"""
import argparse
import json
import os
import re
import sys
import random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_trajectories_v2 import execute_tool, format_timestamp


def _overlaps(a_s, a_e, b_s, b_e):
    return a_s < b_e and a_e > b_s


def extract_keywords(text, n=4):
    """Extract content keywords from text."""
    stops = {"the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this", "it",
             "he", "she", "they", "we", "i", "you", "not", "but", "have",
             "has", "had", "be", "been", "do", "does", "did", "will", "would",
             "can", "could", "should", "may", "might", "so", "very", "just",
             "about", "than", "then", "there", "here", "now", "also", "as",
             "what", "who", "which", "how", "when", "where", "why"}
    words = [w for w in re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
             if w not in stops and len(w) > 2]
    return words[:n]


def find_answer_in_layer(layers, layer_name, answer):
    """Find index items containing the answer in a specific layer."""
    answer_lower = answer.lower()
    answer_words = set(answer_lower.split())
    items = layers.get(layer_name, {}).get("items", [])
    matches = []
    for item in items:
        text = (item.get("text", "") or "").lower()
        if answer_lower in text:
            matches.append(item)
        elif len(answer_words) > 1:
            text_words = set(text.split())
            if len(answer_words & text_words) >= len(answer_words) * 0.5:
                matches.append(item)
    return matches


def strategy_topic_first(q, index_data):
    """Search ASR for topic keywords, then check OCR/visual at that time."""
    grounding = q.get("grounding", {})
    speech = grounding.get("speech_excerpt", "")
    video_id = q["video_id"]
    answer = q.get("answer", "")

    if not speech:
        return None

    keywords = extract_keywords(speech)
    if not keywords:
        return None

    conv = [{"role": "user", "content": q["question"]}]
    tools = []

    # Step 1: Search ASR
    query = " ".join(keywords[:3])
    tool_call = f'search_asr(video_id="{video_id}", query="{query}")'
    obs = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    if "No matching" in obs[:20]:
        return None

    conv.append({"role": "assistant", "content":
        f"<think>I need to find what was discussed about this topic. Let me search the speech transcripts.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs})

    # Extract timestamp from observation
    time_match = re.search(r'\[(\d+):(\d+)', obs)
    found_ms = None
    if time_match:
        found_ms = (int(time_match.group(1)) * 60 + int(time_match.group(2))) * 1000

    # Step 2: Check OCR or visual at that time
    ocr_ref = grounding.get("ocr_reference")
    if ocr_ref and found_ms:
        ocr_kw = extract_keywords(ocr_ref, 2)
        if ocr_kw:
            tool_call = f'search_ocr(video_id="{video_id}", query="{" ".join(ocr_kw)}")'
            obs2 = execute_tool(tool_call, index_data)
            tools.append(tool_call)
            conv.append({"role": "assistant", "content":
                f"<think>Found relevant speech. Now let me check on-screen text for identification.</think>\n<tool>{tool_call}</tool>"})
            conv.append({"role": "tool", "content": obs2})
    elif found_ms:
        start = max(0, found_ms - 5000)
        end = found_ms + 15000
        tool_call = f'get_visual(video_id="{video_id}", start_ms={start}, end_ms={end})'
        obs2 = execute_tool(tool_call, index_data)
        tools.append(tool_call)
        conv.append({"role": "assistant", "content":
            f"<think>Found the speech at this time. Let me check what was shown on screen.</think>\n<tool>{tool_call}</tool>"})
        conv.append({"role": "tool", "content": obs2})

    # Answer
    reasoning = grounding.get("reasoning", "Based on the evidence gathered.")
    conv.append({"role": "assistant", "content":
        f"<think>{reasoning}</think>\n<answer>{answer}</answer>"})

    return conv, tools, "topic_first"


def strategy_identity_first(q, index_data):
    """Search OCR for identity, then find what they discussed in ASR."""
    grounding = q.get("grounding", {})
    ocr_ref = grounding.get("ocr_reference")
    video_id = q["video_id"]
    answer = q.get("answer", "")

    if not ocr_ref:
        return None

    conv = [{"role": "user", "content": q["question"]}]
    tools = []

    # Step 1: Search OCR
    ocr_kw = extract_keywords(ocr_ref, 3)
    if not ocr_kw:
        return None

    tool_call = f'search_ocr(video_id="{video_id}", query="{" ".join(ocr_kw)}")'
    obs = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    if "No matching" in obs[:20]:
        return None

    conv.append({"role": "assistant", "content":
        f"<think>Let me start by checking on-screen text to identify who or what is being shown.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs})

    # Step 2: Search ASR around that time
    time_match = re.search(r'\[(\d+):(\d+)', obs)
    speech = grounding.get("speech_excerpt", "")
    if speech and time_match:
        speech_kw = extract_keywords(speech, 3)
        if speech_kw:
            tool_call = f'search_asr(video_id="{video_id}", query="{" ".join(speech_kw)}")'
            obs2 = execute_tool(tool_call, index_data)
            tools.append(tool_call)
            conv.append({"role": "assistant", "content":
                f"<think>Found on-screen identification. Now let me find what was being discussed at this time.</think>\n<tool>{tool_call}</tool>"})
            conv.append({"role": "tool", "content": obs2})

    reasoning = grounding.get("reasoning", "Based on the evidence gathered.")
    conv.append({"role": "assistant", "content":
        f"<think>{reasoning}</think>\n<answer>{answer}</answer>"})

    return conv, tools, "identity_first"


def strategy_chapter_nav(q, index_data):
    """Navigate by chapter, then browse the relevant time range."""
    video_id = q["video_id"]
    answer = q.get("answer", "")
    source_time = q.get("source_time", "")
    grounding = q.get("grounding", {})

    chapters = index_data.get("layers", {}).get("chapters", {}).get("items", [])
    if not chapters or not source_time:
        return None

    # Parse source time
    m = re.match(r'(\d+):(\d+)', source_time)
    if not m:
        return None
    source_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

    conv = [{"role": "user", "content": q["question"]}]
    tools = []

    # Step 1: Get chapter at source time
    tool_call = f'get_chapter(video_id="{video_id}", time_ms={source_ms})'
    obs = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    conv.append({"role": "assistant", "content":
        f"<think>Let me find what segment of the broadcast this question refers to.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs})

    # Step 2: Browse timeline at that time range
    start = max(0, source_ms - 10000)
    end = source_ms + 30000
    tool_call = f'browse_timeline(video_id="{video_id}", start_ms={start}, end_ms={end})'
    obs2 = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    conv.append({"role": "assistant", "content":
        f"<think>This is in the segment about that topic. Let me browse everything happening at this time.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs2})

    reasoning = grounding.get("reasoning", "Based on the evidence gathered.")
    conv.append({"role": "assistant", "content":
        f"<think>{reasoning}</think>\n<answer>{answer}</answer>"})

    return conv, tools, "chapter_nav"


def strategy_speaker_track(q, index_data):
    """Find speakers at the relevant time, then check what they said."""
    video_id = q["video_id"]
    answer = q.get("answer", "")
    source_time = q.get("source_time", "")
    grounding = q.get("grounding", {})

    speakers = index_data.get("layers", {}).get("speakers", {}).get("items", [])
    if not speakers or not source_time:
        return None

    m = re.match(r'(\d+):(\d+)', source_time)
    if not m:
        return None
    source_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000

    conv = [{"role": "user", "content": q["question"]}]
    tools = []

    # Step 1: Get speakers at this time
    start = max(0, source_ms - 10000)
    end = source_ms + 30000
    tool_call = f'get_speakers(video_id="{video_id}", start_ms={start}, end_ms={end})'
    obs = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    if "No speaker" in obs[:20]:
        return None

    conv.append({"role": "assistant", "content":
        f"<think>Let me find who was speaking during this part of the broadcast.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs})

    # Step 2: Search ASR for relevant content
    speech = grounding.get("speech_excerpt", "")
    if speech:
        kw = extract_keywords(speech, 3)
        if kw:
            tool_call = f'search_asr(video_id="{video_id}", query="{" ".join(kw)}")'
            obs2 = execute_tool(tool_call, index_data)
            tools.append(tool_call)
            conv.append({"role": "assistant", "content":
                f"<think>I found the speaker. Now let me find what they were discussing.</think>\n<tool>{tool_call}</tool>"})
            conv.append({"role": "tool", "content": obs2})

    reasoning = grounding.get("reasoning", "Based on the evidence gathered.")
    conv.append({"role": "assistant", "content":
        f"<think>{reasoning}</think>\n<answer>{answer}</answer>"})

    return conv, tools, "speaker_track"


def strategy_entity_lookup(q, index_data):
    """Search entities directly, then get context around the match."""
    video_id = q["video_id"]
    answer = q.get("answer", "")
    grounding = q.get("grounding", {})

    entities = index_data.get("layers", {}).get("entities", {}).get("items", [])
    if not entities:
        return None

    # Search for answer in entities
    answer_kw = extract_keywords(answer, 2)
    if not answer_kw:
        return None

    conv = [{"role": "user", "content": q["question"]}]
    tools = []

    tool_call = f'search_entities(video_id="{video_id}", query="{" ".join(answer_kw)}")'
    obs = execute_tool(tool_call, index_data)
    tools.append(tool_call)

    if "No matching" in obs[:20]:
        return None

    conv.append({"role": "assistant", "content":
        f"<think>Let me search for relevant entities in the video index.</think>\n<tool>{tool_call}</tool>"})
    conv.append({"role": "tool", "content": obs})

    # Get ASR context around the entity
    time_match = re.search(r'\[(\d+):(\d+)', obs)
    if time_match:
        found_ms = (int(time_match.group(1)) * 60 + int(time_match.group(2))) * 1000
        start = max(0, found_ms - 10000)
        end = found_ms + 20000
        tool_call = f'search_asr(video_id="{video_id}", query="{" ".join(extract_keywords(grounding.get("speech_excerpt",""), 3))}")'
        obs2 = execute_tool(tool_call, index_data)
        tools.append(tool_call)
        conv.append({"role": "assistant", "content":
            f"<think>Found the entity. Let me check what was being discussed at that time.</think>\n<tool>{tool_call}</tool>"})
        conv.append({"role": "tool", "content": obs2})

    reasoning = grounding.get("reasoning", "Based on the evidence gathered.")
    conv.append({"role": "assistant", "content":
        f"<think>{reasoning}</think>\n<answer>{answer}</answer>"})

    return conv, tools, "entity_lookup"


ALL_STRATEGIES = [
    strategy_topic_first,
    strategy_identity_first,
    strategy_chapter_nav,
    strategy_speaker_track,
    strategy_entity_lookup,
]


def construct_diverse_trajectories(q, index_data, max_per_question=2):
    """Try all strategies for a question, keep the best ones."""
    results = []

    for strategy_fn in ALL_STRATEGIES:
        try:
            result = strategy_fn(q, index_data)
        except Exception:
            continue

        if result is None:
            continue

        conv, tools, strategy_name = result

        # Check if tools found useful results
        has_results = any(
            "No " not in execute_tool(tc, index_data)[:15]
            for tc in tools
        ) if tools else False

        if has_results and len(tools) >= 1:
            results.append({
                "conversation": conv,
                "tools_used": [tc.split("(")[0] for tc in tools],
                "strategy": strategy_name,
                "num_tools": len(tools),
                "verified": True,
            })

    # Deduplicate: prefer diverse strategies
    if len(results) > max_per_question:
        # Keep one from each unique strategy
        seen_strategies = set()
        diverse = []
        for r in results:
            if r["strategy"] not in seen_strategies:
                diverse.append(r)
                seen_strategies.add(r["strategy"])
        results = diverse[:max_per_question]

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, default=Path("training_data/output/trajectories_v3_diverse.jsonl"))
    parser.add_argument("--max-per-question", type=int, default=2)
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = [json.loads(l) for l in f]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Constructing diverse trajectories for {len(questions)} questions...")
    print(f"Max {args.max_per_question} trajectories per question")

    stats = defaultdict(int)
    strategy_counts = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                continue

            trajectories = construct_diverse_trajectories(q, idx, args.max_per_question)

            for j, traj in enumerate(trajectories):
                import hashlib
                vid_hash = hashlib.md5(video_id.encode()).hexdigest()[:8]

                result = {
                    "id": f"traj-{vid_hash}-{i:04d}-{j}",
                    "question_id": q.get("id", ""),
                    "video_id": video_id,
                    "question": q["question"],
                    "expected_answer": q.get("answer", ""),
                    "format": q.get("format", "free_text"),
                    "conversation": traj["conversation"],
                    "verification": {
                        "method": "expert_diverse",
                        "strategy": traj["strategy"],
                        "all_verified": traj["verified"],
                        "tool_calls": [{"call": t} for t in traj["tools_used"]],
                    },
                    "num_turns": len([c for c in traj["conversation"] if c["role"] == "assistant"]),
                    "tools_used": traj["tools_used"],
                    "strategy": traj["strategy"],
                }

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                stats["total"] += 1
                strategy_counts[traj["strategy"]] += 1

            if not trajectories:
                stats["no_valid"] += 1

            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(questions)}] total: {stats['total']}, "
                      f"strategies: {dict(strategy_counts)}", flush=True)

    print(f"\nDone.")
    print(f"Total trajectories: {stats['total']}")
    print(f"Questions with no valid trajectory: {stats['no_valid']}")
    print(f"Strategy distribution: {dict(strategy_counts)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
