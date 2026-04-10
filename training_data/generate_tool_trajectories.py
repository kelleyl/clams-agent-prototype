#!/usr/bin/env python3
"""Generate tool-use trajectories from benchmark QA pairs.

For each benchmark question, scripts the ideal tool-use sequence
an agent would follow to find the answer in the video index.

Follows Agent-R1 / WebAgent-R1 format:
  user → assistant(<think> + <tool>) → tool(observation) → assistant(<think> + <answer>)

Available tools:
  - search_index(query, video_id): Keyword search over segments
  - browse_timeline(video_id, start_ms, end_ms): Get full segment details
  - get_entity_info(entity_name): Get grounded entity description
  - list_speakers(video_id): Get speaker names and segments

Usage:
    python training_data/generate_tool_trajectories.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/tool_trajectories.jsonl
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def search_index(question, index_data, top_k=5):
    """Keyword search over segments (mirrors eval/run_index_qa.py)."""
    segments = index_data.get("segments", [])
    question_words = set(question.lower().split())
    stops = {"the", "a", "an", "is", "was", "what", "how", "did", "does",
             "who", "where", "when", "why", "which", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this",
             "based", "does", "according", "between", "during"}
    question_words -= stops

    scored = []
    for seg in segments:
        text = " ".join([
            seg.get("asr_transcript", ""),
            seg.get("ocr_text", "") or "",
            seg.get("visual_caption", "") or "",
            " ".join(e["text"] for e in seg.get("named_entities", [])),
        ]).lower()
        text_words = set(text.split())
        overlap = question_words & text_words
        if overlap:
            scored.append((len(overlap), seg))

    scored.sort(key=lambda x: -x[0])
    return [seg for _, seg in scored[:top_k]]


def format_search_observation(segments, index_data, max_segments=3):
    """Format search results as a tool observation."""
    speaker_names = index_data.get("speaker_names", {})
    parts = []
    for seg in segments[:max_segments]:
        lines = []
        start_s = seg.get("start_ms", 0) / 1000
        end_s = seg.get("end_ms", 0) / 1000
        lines.append(f"Segment ({start_s:.0f}s-{end_s:.0f}s):")

        speaker = speaker_names.get(seg.get("primary_speaker", ""), "")
        if speaker:
            lines.append(f"  Speaker: {speaker}")
        if seg.get("asr_transcript"):
            lines.append(f"  Speech: {seg['asr_transcript'][:150]}")
        if seg.get("ocr_text") and not (seg["ocr_text"] or "").startswith("The image"):
            lines.append(f"  On-screen text: {(seg['ocr_text'] or '')[:100]}")
        if seg.get("visual_caption"):
            lines.append(f"  Visual: {seg['visual_caption'][:100]}")
        ents = seg.get("named_entities", [])
        if ents:
            lines.append(f"  Entities: {', '.join(e['text'] for e in ents[:8])}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def extract_search_query(question, grounding):
    """Extract key terms from question for the search query."""
    # Use grounding to identify key entities/terms
    speech = grounding.get("speech_excerpt", "") or ""
    ocr = grounding.get("ocr_reference", "") or ""

    # Extract quoted names or key nouns from the question
    words = question.split()
    # Find proper nouns (capitalized words not at sentence start)
    key_terms = []
    for i, w in enumerate(words):
        clean = w.strip("'\"?,.")
        if i > 0 and clean and clean[0].isupper() and len(clean) > 2:
            key_terms.append(clean)

    if key_terms:
        return " ".join(key_terms[:4])

    # Fallback: use first few content words from the question
    stops = {"what", "how", "did", "does", "who", "where", "when", "why",
             "which", "the", "a", "an", "is", "was", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this",
             "based", "according", "between", "during", "role", "does"}
    content_words = [w.strip("'\"?,.") for w in words if w.lower().strip("'\"?,.") not in stops]
    return " ".join(content_words[:5])


def build_trajectory(question_data, index_data):
    """Build a multi-turn tool-use trajectory."""
    question = question_data["question"]
    video_id = question_data.get("video_id", "")
    grounding = question_data.get("grounding", {})
    fmt = question_data.get("format", "free_text")

    answer = question_data.get("mc_correct_text",
              question_data.get("reference_answer",
              question_data.get("answer", "")))

    speech_ex = grounding.get("speech_excerpt") or ""
    visual_ref = grounding.get("visual_reference") or ""
    ocr_ref = grounding.get("ocr_reference") or ""
    reasoning = grounding.get("reasoning") or ""

    # Step 1: Search the index
    search_query = extract_search_query(question, grounding)
    relevant_segs = search_index(question, index_data)
    search_obs = format_search_observation(relevant_segs, index_data)

    # Build conversation
    conversation = [
        {"role": "user", "content": question},
    ]

    # Turn 1: Think + search
    think1 = (
        f"I need to search the video index for segments relevant to this question. "
        f"Let me search for key terms."
    )
    conversation.append({
        "role": "assistant",
        "content": f"<think>{think1}</think>\n"
                   f"<tool>search_index(query=\"{search_query}\", video_id=\"{video_id}\")</tool>"
    })

    # Search observation
    if search_obs:
        conversation.append({
            "role": "tool",
            "content": f"Found {len(relevant_segs)} matching segments:\n\n{search_obs}"
        })
    else:
        conversation.append({
            "role": "tool",
            "content": "No matching segments found."
        })

    tools_used = ["search_index"]

    # Turn 2: Browse specific segment if we have source times
    source_times = question_data.get("source_segment_times", [])
    if source_times and len(source_times) > 0:
        st = source_times[0]
        start_ms = st.get("start_ms", 0)
        end_ms = st.get("end_ms", 0)

        think2 = "Let me look at the specific segment in more detail."
        conversation.append({
            "role": "assistant",
            "content": f"<think>{think2}</think>\n"
                       f"<tool>browse_timeline(video_id=\"{video_id}\", "
                       f"start_ms={start_ms}, end_ms={end_ms})</tool>"
        })

        # Find the matching segment
        browse_segs = [s for s in index_data.get("segments", [])
                       if s.get("start_ms", 0) >= start_ms - 1000
                       and s.get("end_ms", 0) <= end_ms + 1000]
        if browse_segs:
            browse_obs = format_search_observation(browse_segs[:2], index_data)
        else:
            browse_obs = format_search_observation(relevant_segs[:1], index_data)

        conversation.append({
            "role": "tool",
            "content": browse_obs
        })
        tools_used.append("browse_timeline")

    # Turn 3: Final reasoning + answer
    think_final_parts = []
    if speech_ex:
        think_final_parts.append(f"The speech transcript mentions: \"{speech_ex[:150]}\"")
    if ocr_ref:
        think_final_parts.append(f"The on-screen text shows: \"{ocr_ref[:150]}\"")
    if visual_ref:
        think_final_parts.append(f"The visual shows: \"{visual_ref[:150]}\"")
    if reasoning:
        think_final_parts.append(f"Combining this evidence: {reasoning}")

    think_final = " ".join(think_final_parts) if think_final_parts else \
        "Based on the evidence from the search results, I can answer."

    conversation.append({
        "role": "assistant",
        "content": f"<think>{think_final}</think>\n<answer>{answer}</answer>"
    })

    return conversation, tools_used


def main():
    parser = argparse.ArgumentParser(
        description="Generate tool-use trajectories from benchmark")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path,
                        default=Path("training_data/output/tool_trajectories.jsonl"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    # Pre-load indexes
    indexes = {}
    for idx_file in args.index_dir.glob("*.json"):
        indexes[idx_file.stem] = json.load(open(idx_file))

    print(f"Generating tool-use trajectories for {len(questions)} questions...")

    results = []
    tool_counts = defaultdict(int)

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            idx_data = indexes.get(video_id)
            if not idx_data:
                continue

            conversation, tools_used = build_trajectory(q, idx_data)

            result = {
                "id": f"traj-{q.get('id', f'q-{i:04d}')}",
                "question_id": q.get("id", f"q-{i:04d}"),
                "video_id": video_id,
                "question": q["question"],
                "format": q.get("format", "free_text"),
                "conversation": conversation,
                "tools_used": tools_used,
                "num_turns": len([c for c in conversation if c["role"] == "assistant"]),
            }

            results.append(result)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            for tool in tools_used:
                tool_counts[tool] += 1

    print(f"Generated {len(results)} trajectories")
    print(f"Output: {args.output}")
    print(f"\nTool usage:")
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")
    print(f"\nAvg turns per trajectory: "
          f"{sum(r['num_turns'] for r in results) / max(1, len(results)):.1f}")


if __name__ == "__main__":
    main()
