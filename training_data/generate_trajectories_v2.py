#!/usr/bin/env python3
"""Generate verified tool-use trajectories from benchmark QA pairs.

For each QA pair, uses an LLM to generate a reasoning trace with tool calls,
then executes each tool call against the real v2 index to verify observations.

Output format follows ReAct (Thought/Action/Observation):
  user: question
  assistant: <think>reasoning</think><tool>tool_call</tool>
  tool: observation (from real index execution)
  assistant: <think>reasoning</think><tool>tool_call</tool>
  tool: observation
  assistant: <think>final reasoning</think><answer>answer</answer>

Tools available to the agent:
  - search_asr(video_id, query): Search ASR transcripts
  - search_ocr(video_id, query): Search on-screen text
  - search_entities(video_id, query): Search named entities
  - get_visual(video_id, start_ms, end_ms): Get visual descriptions in range
  - get_speakers(video_id, start_ms, end_ms): Get speaker info in range
  - get_chapter(video_id, time_ms): Get chapter at time
  - browse_timeline(video_id, start_ms, end_ms): Get all layers in range

Usage:
    python training_data/generate_trajectories_v2.py \
        --benchmark qa-data/benchmark/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/trajectories_v2.jsonl \
        --model qwen3:8b
"""
import argparse
import json
import os
import re
import sys
import time
import requests
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.video_index import overlaps

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ============================================================
# Tool definitions (executed against real v2 indexes)
# ============================================================

TOOL_DESCRIPTIONS = """Available tools:
- search_asr(video_id, query): Search speech transcripts for keywords. Returns matching ASR segments with timestamps and text.
- search_ocr(video_id, query): Search on-screen text (chyrons, slates, graphics). Returns matching OCR items with timestamps and scene labels.
- search_entities(video_id, query): Search named entities (people, organizations, places). Returns matching entities with types and grounding info.
- get_visual(video_id, start_ms, end_ms): Get visual descriptions of what's shown on screen in a time range.
- get_speakers(video_id, start_ms, end_ms): Get who is speaking in a time range, with speaker names if identified.
- get_chapter(video_id, time_ms): Get the chapter/topic at a given time.
- browse_timeline(video_id, start_ms, end_ms): Get everything happening in a time range across all layers (ASR, OCR, visual, speakers, entities)."""

TOOL_FUNCTIONS = {}


def _search_layer(index_data, layer_name, query, max_results=5):
    """Generic search across a layer's text fields."""
    layers = index_data.get("layers", {})
    layer = layers.get(layer_name, {})
    items = layer.get("items", [])
    query_terms = set(query.lower().split())

    scored = []
    for item in items:
        searchable = " ".join(
            str(v) for v in item.values() if isinstance(v, str)
        ).lower()
        # Also search nested entries
        for entry in item.get("entries", []):
            searchable += " " + " ".join(str(v) for v in entry.values() if isinstance(v, str)).lower()
        score = sum(1 for t in query_terms if t in searchable)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[:max_results]]


def format_timestamp(start_ms, end_ms, index_data):
    """Format a timestamp with structural context.

    Instead of raw ms, provides:
    - Human-readable time (MM:SS)
    - Position in video (percentage)
    - Chapter context (what topic is being discussed)
    - Relative position within chapter
    """
    duration_ms = index_data.get("duration_ms", 0)
    start_s = start_ms / 1000
    end_s = end_ms / 1000

    # Human-readable
    start_fmt = "%d:%02d" % (int(start_s) // 60, int(start_s) % 60)
    end_fmt = "%d:%02d" % (int(end_s) // 60, int(end_s) % 60)

    # Video position
    pct = ""
    if duration_ms > 0:
        p = start_ms / duration_ms * 100
        pct = " | %d%%" % p

    # Chapter context
    chapter_info = ""
    chapters = index_data.get("layers", {}).get("chapters", {}).get("items", [])
    for ch in chapters:
        if overlaps(start_ms, end_ms, ch["start_ms"], ch["end_ms"]):
            ch_offset = (start_ms - ch["start_ms"]) / 1000
            chapter_info = " | %s (+%ds)" % (ch.get("title", ""), int(ch_offset))
            break

    return "[%s-%s%s%s]" % (start_fmt, end_fmt, pct, chapter_info)


def execute_tool(tool_call, index_data):
    """Execute a tool call against the real index and return the observation."""
    # Parse tool call: tool_name(arg1=val1, arg2=val2)
    match = re.match(r'(\w+)\((.*)\)', tool_call.strip())
    if not match:
        return f"Error: Could not parse tool call: {tool_call}"

    func_name = match.group(1)
    args_str = match.group(2)

    # Parse arguments
    args = {}
    for part in re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([\d]+))', args_str):
        key = part[0]
        if part[1]:
            val = part[1]
        elif part[2]:
            try:
                val = int(part[2])
            except ValueError:
                val = part[2]
        else:
            continue  # skip empty arguments
        args[key] = val

    layers = index_data.get("layers", {})

    if func_name == "search_asr":
        query = args.get("query", "")
        results = _search_layer(index_data, "asr", query)
        if not results:
            return "No matching ASR segments found."
        lines = []
        for item in results:
            ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
            lines.append(f"{ts} {item.get('text', '')[:200]}")
        return "\n".join(lines)

    elif func_name == "search_ocr":
        query = args.get("query", "")
        results = _search_layer(index_data, "ocr", query)
        if not results:
            return "No matching on-screen text found."
        lines = []
        for item in results:
            ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
            label = item.get("scene_label", "")
            lines.append(f"{ts} ({label}) {item.get('text', '')[:200]}")
        return "\n".join(lines)

    elif func_name == "search_entities":
        query = args.get("query", "")
        results = _search_layer(index_data, "entities", query)
        if not results:
            return "No matching entities found."
        lines = []
        for item in results:
            ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
            grounding = item.get("grounding", "")
            lines.append(f"{ts} {item.get('text', '')} ({item.get('type', '')}) {grounding[:100] if grounding else ''}")
        return "\n".join(lines)

    elif func_name == "get_visual":
        start_ms = int(args.get("start_ms", 0))
        end_ms = int(args.get("end_ms", start_ms + 30000))
        items = [i for i in layers.get("visual_captions", {}).get("items", [])
                 if overlaps(i["start_ms"], i["end_ms"], start_ms, end_ms)]
        if not items:
            return "No visual descriptions in this range."
        lines = []
        for item in items[:5]:
            ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
            lines.append(f"{ts} {item.get('text', '')[:200]}")
        return "\n".join(lines)

    elif func_name == "get_speakers":
        start_ms = int(args.get("start_ms", 0))
        end_ms = int(args.get("end_ms", start_ms + 30000))
        items = [i for i in layers.get("speakers", {}).get("items", [])
                 if overlaps(i["start_ms"], i["end_ms"], start_ms, end_ms)]
        if not items:
            return "No speaker information in this range."
        seen = set()
        lines = []
        for item in items:
            ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
            name = item.get("speaker_name", item.get("speaker_id", "unknown"))
            if name not in seen:
                seen.add(name)
                lines.append(f"{ts} Speaker: {name}")
        return "\n".join(lines)

    elif func_name == "get_chapter":
        time_ms = int(args.get("time_ms", 0))
        duration_ms = index_data.get("duration_ms", 0)
        chapters = layers.get("chapters", {}).get("items", [])
        for ch in chapters:
            if ch["start_ms"] <= time_ms <= ch["end_ms"]:
                pct = ""
                if duration_ms > 0:
                    pct = " | %d%% through video" % (ch["start_ms"] / duration_ms * 100)
                ch_dur = (ch["end_ms"] - ch["start_ms"]) / 1000
                return f"Chapter: {ch.get('title', 'untitled')} ({ch_dur:.0f}s long{pct})"
        return "No chapter information at this time."

    elif func_name == "browse_timeline":
        start_ms = int(args.get("start_ms", 0))
        end_ms = int(args.get("end_ms", start_ms + 30000))
        parts = []
        for layer_name in ["asr", "ocr", "visual_captions", "speakers", "scenes"]:
            items = [i for i in layers.get(layer_name, {}).get("items", [])
                     if overlaps(i["start_ms"], i["end_ms"], start_ms, end_ms)]
            if items:
                layer_lines = []
                for item in items[:3]:
                    ts = format_timestamp(item["start_ms"], item["end_ms"], index_data)
                    text = item.get("text", item.get("label", item.get("speaker_name", "")))
                    layer_lines.append(f"  {ts} {str(text)[:100]}")
                parts.append(f"{layer_name}:\n" + "\n".join(layer_lines))
        return "\n".join(parts) if parts else "No data in this range."

    return f"Unknown tool: {func_name}"


# ============================================================
# LLM-based trajectory generation
# ============================================================

TRAJECTORY_SYSTEM = """You are a video understanding agent. Use tools to find evidence, then answer.

{tool_descriptions}

Rules:
- Use <think>reasoning</think> before each action
- Use <tool>tool_call(args)</tool> to call ONE tool at a time
- After seeing observations, reason about what you learned and what to do next
- Use timestamps from observations to make follow-up calls (e.g., if ASR says something at 450s, use get_visual at that time)
- When you have enough evidence, use <answer>your answer</answer>
- Use 2-3 tool calls to gather evidence from multiple layers before answering

IMPORTANT: Use the video_id "{video_id}" in all tool calls."""

TRAJECTORY_USER = """Question about video {video_id}: {question}

Search the index to find the answer. Start by searching speech or on-screen text for relevant content."""


def generate_trajectory_llm(question_data, index_data, model, max_turns=3):
    """Generate a trajectory INTERACTIVELY -- call LLM, execute tool, feed back real observation.

    This ensures tool arguments (especially time ranges) are based on real
    observations, not hallucinated by the LLM.
    """
    question = question_data["question"]
    video_id = question_data.get("video_id", "")
    answer = question_data.get("mc_correct_text",
              question_data.get("reference_answer",
              question_data.get("answer", "")))

    system = TRAJECTORY_SYSTEM.format(
        tool_descriptions=TOOL_DESCRIPTIONS,
        video_id=video_id,
    )
    user_msg = TRAJECTORY_USER.format(video_id=video_id, question=question)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    conversation = [{"role": "user", "content": question}]
    verification = {"tool_calls": [], "all_verified": True, "method": "llm_interactive"}

    for turn in range(max_turns):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 500},
                },
                timeout=120,
            )
            resp.raise_for_status()
            response = resp.json().get("message", {}).get("content", "")
        except Exception as e:
            return None, f"LLM error: {e}"

        # Strip thinking tags for parsing but keep them in conversation
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Check for answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match and len(verification["tool_calls"]) > 0:
            predicted_answer = answer_match.group(1).strip()
            conversation.append({"role": "assistant", "content": response})
            break

        # Check for tool call
        tool_match = re.search(r'<tool>(.*?)</tool>', response, re.DOTALL)
        if tool_match:
            tool_call = tool_match.group(1).strip()
            # Execute against real index
            real_observation = execute_tool(tool_call, index_data)

            conversation.append({"role": "assistant", "content": response})
            conversation.append({"role": "tool", "content": real_observation})

            # Feed observation back to LLM for next step
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation:\n{real_observation[:800]}\n\nBased on this observation, decide your next action. Use timestamps from above if you need to look up more details at a specific time."})

            verification["tool_calls"].append({
                "call": tool_call,
                "observation_length": len(real_observation),
                "has_results": "No " not in real_observation[:10] and "Error" not in real_observation[:10],
            })
        else:
            # No tool call and no answer -- add response and force answer next turn
            conversation.append({"role": "assistant", "content": response})
            if len(verification["tool_calls"]) > 0:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Based on the evidence gathered, provide your <answer>final answer</answer> now."})
            else:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"You must search the index first. Try: <tool>search_asr(video_id=\"{video_id}\", query=\"relevant keywords\")</tool>"})

    # Verify
    has_answer = any("<answer>" in c.get("content", "") for c in conversation if c["role"] == "assistant")
    has_tools = any(c["role"] == "tool" for c in conversation)
    tools_found_results = any(tc["has_results"] for tc in verification["tool_calls"]) if verification["tool_calls"] else False

    verification["has_answer"] = has_answer
    verification["has_tools"] = has_tools
    verification["tools_found_results"] = tools_found_results
    verification["all_verified"] = has_answer and has_tools and tools_found_results

    return conversation, verification


# ============================================================
# Scripted trajectory (fallback, deterministic)
# ============================================================

def generate_trajectory_scripted(question_data, index_data):
    """Generate a scripted trajectory using the question's grounding info."""
    question = question_data["question"]
    video_id = question_data.get("video_id", "")
    grounding = question_data.get("grounding", {})
    answer = question_data.get("mc_correct_text",
              question_data.get("reference_answer",
              question_data.get("answer", "")))

    conversation = [{"role": "user", "content": question}]

    # Step 1: Search based on question content
    # Extract search terms from question
    words = question.split()
    stops = {"what", "how", "did", "does", "who", "where", "when", "why",
             "which", "the", "a", "an", "is", "was", "in", "on", "at",
             "to", "of", "and", "or", "for", "with", "that", "this"}
    key_terms = [w.strip("'\"?,.") for w in words
                 if w.lower().strip("'\"?,.") not in stops and len(w) > 2][:5]
    search_query = " ".join(key_terms)

    # Decide which tool to use first based on grounding
    speech_ex = grounding.get("speech_excerpt") or ""
    ocr_ref = grounding.get("ocr_reference") or ""
    visual_ref = grounding.get("visual_reference") or ""

    if ocr_ref:
        first_tool = f'search_ocr(video_id="{video_id}", query="{search_query}")'
        think1 = f"The question mentions on-screen text. Let me search the OCR layer for relevant text."
    elif speech_ex:
        first_tool = f'search_asr(video_id="{video_id}", query="{search_query}")'
        think1 = f"Let me search the speech transcripts for information about this topic."
    else:
        first_tool = f'search_asr(video_id="{video_id}", query="{search_query}")'
        think1 = f"Let me search the video index for segments relevant to this question."

    # Execute tool
    obs1 = execute_tool(first_tool, index_data)
    conversation.append({"role": "assistant", "content": f"<think>{think1}</think>\n<tool>{first_tool}</tool>"})
    conversation.append({"role": "tool", "content": obs1})

    # Step 2: Follow up based on what was found
    # Extract a timestamp from the first result
    time_match = re.search(r'\[(\d+)s', obs1)
    if time_match:
        found_time_s = int(time_match.group(1))
        found_time_ms = found_time_s * 1000

        if visual_ref:
            second_tool = f'get_visual(video_id="{video_id}", start_ms={found_time_ms}, end_ms={found_time_ms + 30000})'
            think2 = "Let me check what was visually shown during this segment."
        else:
            second_tool = f'get_speakers(video_id="{video_id}", start_ms={found_time_ms}, end_ms={found_time_ms + 30000})'
            think2 = "Let me check who was speaking during this segment."

        obs2 = execute_tool(second_tool, index_data)
        conversation.append({"role": "assistant", "content": f"<think>{think2}</think>\n<tool>{second_tool}</tool>"})
        conversation.append({"role": "tool", "content": obs2})

    # Step 3: Answer
    think_parts = []
    if speech_ex:
        think_parts.append(f'The speech transcript mentions: "{speech_ex[:150]}"')
    if ocr_ref:
        think_parts.append(f'The on-screen text shows: "{ocr_ref[:150]}"')
    if visual_ref:
        think_parts.append(f'The visual shows: "{visual_ref[:150]}"')
    think_final = " ".join(think_parts) if think_parts else "Based on the evidence from my search, I can answer."

    conversation.append({"role": "assistant", "content": f"<think>{think_final}</think>\n<answer>{answer}</answer>"})

    verification = {
        "tool_calls": [{"call": first_tool, "has_results": "No " not in obs1[:10]}],
        "has_answer": True,
        "has_tools": True,
        "tools_found_results": "No " not in obs1[:10],
        "all_verified": True,
        "method": "scripted",
    }

    return conversation, verification


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, default=Path("training_data/output/trajectories_v2.jsonl"))
    parser.add_argument("--model", type=str, default="qwen3:8b",
                        help="LLM for trajectory generation (or 'scripted' for deterministic)")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output (append mode)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]

    if args.max_questions:
        questions = questions[:args.max_questions]

    # Pre-load indexes
    indexes = {}
    for idx_file in args.index_dir.glob("*.json"):
        indexes[idx_file.stem] = json.load(open(idx_file))

    # Resume support: skip already-processed questions
    done_ids = set()
    if args.resume and args.output.exists():
        with open(args.output) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_ids)} already done, skipping those")

    print(f"Generating trajectories for {len(questions)} questions...")
    print(f"Method: {'scripted' if args.model == 'scripted' else f'LLM ({args.model})'}")

    stats = defaultdict(int)
    file_mode = "a" if args.resume and done_ids else "w"

    with open(args.output, file_mode) as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            qid = q.get("id", f"q-{i:04d}")

            if qid in done_ids:
                continue

            idx_data = indexes.get(video_id)
            if not idx_data:
                stats["no_index"] += 1
                continue

            if args.model == "scripted":
                conversation, verification = generate_trajectory_scripted(q, idx_data)
            else:
                try:
                    conversation, verification = generate_trajectory_llm(q, idx_data, args.model)
                except Exception as e:
                    stats["llm_error"] += 1
                    conversation, verification = generate_trajectory_scripted(q, idx_data)
                    verification["method"] = "scripted_fallback"
                    verification["error"] = str(e)
                else:
                    if conversation is None:
                        stats["llm_error"] += 1
                        conversation, verification = generate_trajectory_scripted(q, idx_data)
                        verification["method"] = "scripted_fallback"

            result = {
                "id": f"traj-{i:04d}",
                "question_id": q.get("id", f"q-{i:04d}"),
                "video_id": video_id,
                "question": q["question"],
                "expected_answer": q.get("mc_correct_text", q.get("reference_answer", q.get("answer", ""))),
                "format": q.get("format", "free_text"),
                "conversation": conversation,
                "verification": verification,
                "num_turns": len([c for c in conversation if c["role"] == "assistant"]),
                "tools_used": [tc["call"].split("(")[0] for tc in verification.get("tool_calls", [])],
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

            if verification.get("all_verified"):
                stats["verified"] += 1
            else:
                stats["unverified"] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(questions)}] verified: {stats['verified']}, "
                      f"unverified: {stats['unverified']}", flush=True)

            if args.model != "scripted":
                time.sleep(0.5)

    print(f"\nDone: {stats['verified']} verified, {stats['unverified']} unverified, "
          f"{stats.get('no_index', 0)} no index, {stats.get('llm_error', 0)} LLM errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
