#!/usr/bin/env python3
"""Evaluate a LangGraph-based QA agent on the benchmark.

Uses structured tool calling via LangGraph instead of free-text ReAct parsing.
The agent has access to video index search tools and iteratively gathers
evidence before answering.

Usage:
    python eval/run_langgraph_eval.py \
        --benchmark qa-data/benchmark/v2/benchmark.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v2_langgraph_qwen3_8b.jsonl \
        --model qwen3:8b
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# ============================================================
# Video index tools (as LangChain tools with proper schemas)
# ============================================================

# Global index data -- set per question
_current_index = None
_current_video_id = None


def _search_layer(layer_name, query, max_results=5):
    """Generic search across a layer's text fields."""
    if not _current_index:
        return "No index loaded"
    layers = _current_index.get("layers", {})
    items = layers.get(layer_name, {}).get("items", [])
    query_terms = set(query.lower().split())

    scored = []
    for item in items:
        searchable = " ".join(
            str(v) for v in item.values() if isinstance(v, str)
        ).lower()
        score = sum(1 for t in query_terms if t in searchable)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    results = [item for _, item in scored[:max_results]]

    if not results:
        return f"No matching {layer_name} segments found."

    lines = []
    for item in results:
        start_s = item.get("start_ms", 0) / 1000
        end_s = item.get("end_ms", 0) / 1000
        text = item.get("text", "")[:200]
        label = item.get("scene_label", "")
        extra = f" ({label})" if label else ""
        lines.append(f"[{start_s:.0f}s-{end_s:.0f}s]{extra} {text}")
    return "\n".join(lines)


def _overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and a_end > b_start


def _get_layer_range(layer_name, start_ms, end_ms):
    """Get all items from a layer overlapping a time range."""
    if not _current_index:
        return "No index loaded"
    layers = _current_index.get("layers", {})
    items = layers.get(layer_name, {}).get("items", [])
    results = [item for item in items
               if _overlaps(item.get("start_ms", 0), item.get("end_ms", 0), start_ms, end_ms)]
    if not results:
        return f"No {layer_name} content in this time range."
    lines = []
    for item in results:
        text = item.get("text", "")[:200]
        name = item.get("speaker_name", "")
        prefix = f"{name}: " if name else ""
        lines.append(f"[{item['start_ms']/1000:.0f}s-{item['end_ms']/1000:.0f}s] {prefix}{text}")
    return "\n".join(lines)


@tool
def search_speech(query: str) -> str:
    """Search speech transcripts (ASR) for keywords. Use this to find what was said about a topic."""
    return _search_layer("asr", query)


@tool
def search_onscreen_text(query: str) -> str:
    """Search on-screen text like chyrons, titles, and graphics (OCR). Use this to find names, titles, and labels shown on screen."""
    return _search_layer("ocr", query)


@tool
def search_entities(query: str) -> str:
    """Search named entities (people, organizations, places). Returns entity type and source layer."""
    if not _current_index:
        return "No index loaded"
    layers = _current_index.get("layers", {})
    items = layers.get("entities", {}).get("items", [])
    query_terms = set(query.lower().split())

    scored = []
    for item in items:
        text = (item.get("text", "") or "").lower()
        if any(t in text for t in query_terms):
            scored.append(item)

    if not scored:
        return "No matching entities found."

    lines = []
    for item in scored[:8]:
        grounding = item.get("grounding", "")
        g_str = f" | {grounding[:60]}" if grounding else ""
        lines.append(f"[{item['start_ms']/1000:.0f}s-{item['end_ms']/1000:.0f}s] "
                     f"{item.get('text','')} ({item.get('type','')}) "
                     f"src={item.get('source_layer','')}{g_str}")
    return "\n".join(lines)


@tool
def get_speakers(start_seconds: int, end_seconds: int) -> str:
    """Get who is speaking in a time range. Returns speaker names and what they said."""
    return _get_layer_range("speakers", start_seconds * 1000, end_seconds * 1000)


@tool
def get_visual_description(start_seconds: int, end_seconds: int) -> str:
    """Get visual descriptions of what's shown on screen in a time range."""
    return _get_layer_range("visual_captions", start_seconds * 1000, end_seconds * 1000)


@tool
def get_chapter_info(time_seconds: int) -> str:
    """Get the chapter/topic at a given time in the video."""
    if not _current_index:
        return "No index loaded"
    chapters = _current_index.get("layers", {}).get("chapters", {}).get("items", [])
    t_ms = time_seconds * 1000
    for ch in chapters:
        if ch["start_ms"] <= t_ms <= ch["end_ms"]:
            return f"Chapter: {ch.get('title', 'Unknown')} [{ch['start_ms']/1000:.0f}s-{ch['end_ms']/1000:.0f}s]"
    return "No chapter information at this time."


# All tools
TOOLS = [search_speech, search_onscreen_text, search_entities,
         get_speakers, get_visual_description, get_chapter_info]


# ============================================================
# LangGraph agent
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


SYSTEM_PROMPT = """You are a video understanding agent answering questions about broadcast videos using a structured index.

You have tools to search different layers of the video index:
- search_speech: Find what was SAID (speech transcripts)
- search_onscreen_text: Find what was SHOWN as text (chyrons, titles, graphics)
- search_entities: Find named people, organizations, places
- get_speakers: Find who is speaking at a specific time
- get_visual_description: Find what's visually shown at a specific time
- get_chapter_info: Find the topic/chapter at a specific time

STRATEGY: Use 2-3 tools to gather evidence from multiple layers before answering.
1. First search speech OR on-screen text to find the relevant time range
2. Then use other tools at that time range to get complementary information
3. Combine evidence from multiple sources to answer

Always gather evidence before answering. One search is usually not enough."""


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Decide whether to call tools or end."""
    messages = state["messages"]
    last = messages[-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_agent(model_name: str):
    """Build a LangGraph agent with video index tools."""
    llm = ChatOllama(model=model_name, temperature=0.3)
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ============================================================
# Evaluation loop
# ============================================================

def run_agent_on_question(agent, question, video_id, index_data, mc_options=None, max_steps=8, model_name="qwen3:8b"):
    """Run the LangGraph agent on a single question."""
    global _current_index, _current_video_id
    _current_index = index_data
    _current_video_id = video_id

    # Don't show MC options yet -- force the model to search first
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Video: {video_id}\n\nQuestion: {question}\n\nSearch the index to find the answer. Do NOT guess."),
    ]

    tool_calls_made = []
    try:
        result = agent.invoke(
            {"messages": messages},
            config={"recursion_limit": max_steps},
        )

        final_messages = result["messages"]

        # Count tool calls
        for msg in final_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append(tc["name"])

        # Extract the agent's free-text answer (evidence summary)
        evidence_answer = ""
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                evidence_answer = msg.content
                evidence_answer = re.sub(r'<think>.*?</think>', '', evidence_answer, flags=re.DOTALL).strip()
                break

        # For MC: now show the options and ask to pick based on gathered evidence
        if mc_options and evidence_answer:
            opts = "\n".join(f"  {k}. {v}" for k, v in mc_options.items())
            llm = ChatOllama(model=model_name, temperature=0.1)
            mc_messages = [
                SystemMessage(content="Based on the evidence you gathered, select the best answer option."),
                HumanMessage(content=f"Question: {question}\n\nEvidence found:\n{evidence_answer}\n\nOptions:\n{opts}\n\nSelect one option letter (A/B/C/D)."),
            ]
            mc_response = llm.invoke(mc_messages)
            answer = mc_response.content
            answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        else:
            answer = evidence_answer

        return {
            "answer": answer,
            "tool_calls": tool_calls_made,
            "num_tool_calls": len(tool_calls_made),
            "turns": len([m for m in final_messages if isinstance(m, AIMessage)]),
        }

    except Exception as e:
        return {
            "answer": f"Error: {e}",
            "tool_calls": tool_calls_made,
            "num_tool_calls": len(tool_calls_made),
            "turns": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="LangGraph agent evaluation")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="qwen3:8b")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max agent steps (tool calls + responses)")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    agent = build_agent(args.model)

    print(f"LangGraph Agent Evaluation")
    print(f"Model: {args.model}")
    print(f"Questions: {len(questions)}")
    print(f"Max steps: {args.max_steps}")

    stats = {"total": 0, "answered": 0, "no_index": 0, "errors": 0,
             "total_tools": 0, "total_turns": 0}

    with open(args.output, "w") as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            question_text = q["question"]
            fmt = q.get("format", "free_text")

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                continue

            result = run_agent_on_question(
                agent, question_text, video_id, idx,
                mc_options=mc_options, max_steps=args.max_steps,
                model_name=args.model
            )

            if "error" in result:
                stats["errors"] += 1

            # Extract MC letter
            predicted_answer = result["answer"]
            if fmt == "multiple_choice" and isinstance(predicted_answer, str):
                p = predicted_answer.strip().upper()
                if p and p[0] in "ABCD" and (len(p) == 1 or p[1] in ".):, "):
                    predicted_answer = p[0]
                else:
                    m = re.search(r'(?:answer|option|choice)\s*(?:is\s*)?([A-D])\b', p, re.IGNORECASE)
                    if m:
                        predicted_answer = m.group(1).upper()
                    else:
                        m = re.search(r'\b([A-D])\b', p)
                        if m:
                            predicted_answer = m.group(1).upper()

            stats["total"] += 1
            stats["answered"] += 1
            stats["total_tools"] += result["num_tool_calls"]
            stats["total_turns"] += result["turns"]

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": question_text,
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted_answer,
                "tool_calls": result["tool_calls"],
                "num_tool_calls": result["num_tool_calls"],
                "turns": result["turns"],
                "reasoning_type": q.get("reasoning_type", ""),
            }

            fout.write(json.dumps(output) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                n = max(1, stats["answered"])
                print(f"  [{i+1}/{len(questions)}] "
                      f"avg tools: {stats['total_tools']/n:.1f}, "
                      f"avg turns: {stats['total_turns']/n:.1f}, "
                      f"errors: {stats['errors']}", flush=True)

            time.sleep(0.3)

    n = max(1, stats["answered"])
    print(f"\nDone.")
    print(f"Total: {stats['total']}, Errors: {stats['errors']}")
    print(f"Avg tools: {stats['total_tools']/n:.1f}")
    print(f"Avg turns: {stats['total_turns']/n:.1f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
