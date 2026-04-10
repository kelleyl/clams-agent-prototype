"""
LangGraph Studio entry point for the CLAMS Agent.

Iterative tool-use agent that processes video/image content by selecting
and running CLAMS apps and FFmpeg operations in a reasoning loop.
The agent reviews output after each tool use and decides the next action,
similar to DVD (Deep Video Discovery) and VideoAgent patterns.

Run with: langgraph dev
"""

from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from pathlib import Path
import operator
import json
import logging
import time

from utils.clams_tools import CLAMSToolbox
from utils.config import ConfigManager
from utils.ffmpeg_tools import FFmpegTools
from utils.evaluation_rag import get_evaluation_rag, EvaluationRAG
from utils.kg_tools import get_kg, kg_search, kg_neighbors, kg_path, kg_query
from utils.video_index import get_video_index, VideoIndex

logger = logging.getLogger(__name__)

# ============================================================================
# Singletons
# ============================================================================

_toolbox = None
_ffmpeg_tools = None
_evaluation_rag = None
_video_index = None


def get_toolbox() -> CLAMSToolbox:
    """Get or create the CLAMS toolbox singleton."""
    global _toolbox
    if _toolbox is None:
        _toolbox = CLAMSToolbox()
    return _toolbox


def get_ffmpeg_tools() -> Optional[FFmpegTools]:
    """Get or create the FFmpeg tools singleton."""
    global _ffmpeg_tools
    if _ffmpeg_tools is None:
        try:
            _ffmpeg_tools = FFmpegTools()
            logger.info("FFmpeg tools initialized successfully")
        except Exception as e:
            logger.warning(f"FFmpeg tools not available: {e}")
            _ffmpeg_tools = None
    return _ffmpeg_tools


def get_eval_rag() -> Optional[EvaluationRAG]:
    """Get or create the Evaluation RAG singleton."""
    global _evaluation_rag
    if _evaluation_rag is None:
        try:
            _evaluation_rag = get_evaluation_rag()
            _evaluation_rag.initialize()
            logger.info("Evaluation RAG initialized successfully")
        except Exception as e:
            logger.warning(f"Evaluation RAG not available: {e}")
            _evaluation_rag = None
    return _evaluation_rag


def get_video_index_singleton() -> Optional[VideoIndex]:
    """Get or create the VideoIndex singleton."""
    global _video_index
    if _video_index is None:
        try:
            _video_index = get_video_index()
            logger.info("Video index initialized successfully")
        except Exception as e:
            logger.warning(f"Video index not available: {e}")
            _video_index = None
    return _video_index


def get_llm():
    """Get the configured LLM."""
    config_manager = ConfigManager()
    llm_config = config_manager.get_config().llm

    if llm_config.provider == "ollama":
        return ChatOllama(
            model=llm_config.model_name,
            base_url=llm_config.base_url,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=llm_config.model_name,
            streaming=True,
            temperature=llm_config.temperature
        )


# ============================================================================
# State Definition
# ============================================================================

class PipelineState(TypedDict):
    """State for the iterative CLAMS tool-use agent."""
    messages: Annotated[list[BaseMessage], operator.add]

    # Input
    user_request: str
    video_path: str
    image_paths: list[str]
    input_type: str                     # "video", "image", or "mixed"
    video_metadata: dict
    mode: str                           # "pipeline" or "qa"

    # Approval
    execution_approved: bool

    # Iterative execution loop
    current_mmif: dict
    iteration: int
    max_depth: int
    tool_history: list[dict]
    next_action: dict

    # Execution tracking
    execution_results: list[dict]
    execution_errors: list[str]
    execution_complete: bool

    # Output
    execution_summary: str
    output_mmif_path: str


# ============================================================================
# Helper Functions
# ============================================================================

def _get_kg_stats_summary() -> str:
    """Get a one-line summary of the KG for prompts."""
    try:
        kg = get_kg()
        s = kg.stats()
        return (
            f"{s.get('entities', 0)} entities (persons, orgs, locations), "
            f"{s.get('relations', 0)} relations across {s.get('videos', 0)} videos"
        )
    except Exception:
        return "entities and relations"


def summarize_mmif_state(mmif: dict) -> str:
    """Create an LLM-readable summary of the current MMIF state."""
    if not mmif:
        return "No MMIF document yet."

    lines = ["Documents:"]
    for doc in mmif.get("documents", []):
        # Extract type name: "http://mmif.clams.ai/vocabulary/VideoDocument/v1" → "VideoDocument"
        type_uri = doc.get("@type", "")
        parts = type_uri.rstrip("/").split("/")
        doc_type = parts[-2] if len(parts) >= 2 else parts[-1] if parts else "unknown"
        location = doc.get("properties", {}).get("location", "unknown")
        # Shorten file paths for readability
        if "file://" in location:
            location = location.replace("file://", "")
            location = "..." + location[-60:] if len(location) > 60 else location
        lines.append(f"  - {doc_type}: {location}")

    views = mmif.get("views", [])
    if views:
        lines.append(f"\nViews ({len(views)}):")
        for view in views:
            app = view.get("metadata", {}).get("app", "unknown")
            # Shorten app URIs
            if "/" in app:
                app = app.split("/")[-2] if app.endswith("/") else app.split("/")[-1]
            annotations = view.get("annotations", [])
            # Count by type
            type_counts = {}
            for ann in annotations:
                atype_uri = ann.get("@type", "unknown")
                atype_parts = atype_uri.rstrip("/").split("/")
                ann_type = atype_parts[-2] if len(atype_parts) >= 2 else atype_parts[-1]
                type_counts[ann_type] = type_counts.get(ann_type, 0) + 1
            counts_str = ", ".join(f"{t}: {c}" for t, c in type_counts.items())
            lines.append(f"  - {app}: {len(annotations)} annotations ({counts_str})")
    else:
        lines.append("\nNo views/annotations yet.")

    return "\n".join(lines)


def format_available_tools(gpu_available: bool) -> str:
    """Format available CLAMS tools and FFmpeg operations for the LLM prompt."""
    from utils.gpu_classification import requires_gpu

    toolbox = get_toolbox()
    lines = ["CLAMS Apps:"]
    for name, info in sorted(toolbox.app_metadata.items()):
        gpu_needed = requires_gpu(name)
        if gpu_needed and not gpu_available:
            continue
        label = "(GPU)" if gpu_needed else "(CPU)"
        metadata = info.get("metadata", {})
        desc = metadata.get("description", "No description")
        if len(desc) > 100:
            desc = desc[:100] + "..."

        # Show input/output types
        inputs = [inp.get("@type", "").split("/")[-1]
                  for inp in metadata.get("input", []) if isinstance(inp, dict)]
        outputs = [out.get("@type", "").split("/")[-1]
                   for out in metadata.get("output", []) if isinstance(out, dict)]
        io_str = ""
        if inputs or outputs:
            io_str = f" [{' + '.join(inputs[:3])} → {' + '.join(outputs[:3])}]"

        lines.append(f"  - {name}: {desc} {label}{io_str}")

    lines.extend([
        "",
        "FFmpeg Operations:",
        "  - extract_frames: Extract video frames at specified FPS (produces image files)",
        "  - extract_audio: Extract audio track from video",
        "  - trim_video: Trim video to a time segment (start/end in seconds)",
        "  - generate_thumbnail_sprite: Create composite grid image of thumbnails",
        "  - get_video_info: Get video metadata (duration, resolution, codec)",
    ])

    # Video Index Operations (only show if index has videos)
    index = get_video_index_singleton()
    if index:
        indexed_videos = index.get_indexed_videos()
        if indexed_videos:
            lines.extend([
                "",
                f"Video Index Operations ({len(indexed_videos)} video(s) indexed):",
                "  - list_videos: List all indexed video IDs (no parameters needed)",
                "  - search: Semantic search across all indexed segments (parameters: query, top_k)",
                "  - browse_timeline: Browse segments by time range (parameters: video_id, start_ms, end_ms)",
                "  - filter_segments: Filter by scene label or text presence (parameters: video_id, scene_label, has_text)",
                "  - get_video_summary: Get overview stats for an indexed video (parameters: video_id)",
                "  NOTE: For operations requiring video_id, use EXACT IDs from this list:",
                f"  Indexed videos: {', '.join(indexed_videos[:10])}{'...' if len(indexed_videos) > 10 else ''}",
            ])

    # Web / Knowledge Operations (always available)
    lines.extend([
        "",
        "Web & Knowledge Operations:",
        "  - enrich_entities: Ground all named entities in a video via Wikidata + Wikipedia (parameters: video_id)",
        "  - wikipedia_lookup: Look up a single entity on Wikipedia (parameters: entity)",
        "  - wikidata_lookup: Get structured facts from Wikidata — positions, roles, relationships (parameters: entity)",
        "  - pubmed_search: Search PubMed for biomedical/health literature (parameters: query, max_results)",
        "  - web_search: Search the web via DuckDuckGo (parameters: query)",
    ])

    # Knowledge Graph Operations
    try:
        kg = get_kg()
        kg_stats = kg.stats()
        entity_count = kg_stats.get("entities", 0)
        relation_count = kg_stats.get("relations", 0)
        lines.extend([
            "",
            f"Knowledge Graph Operations ({entity_count} entities, {relation_count} relations across {kg_stats.get('videos', 0)} videos):",
            "  - kg_search: Search entities by name (parameters: query, entity_type)",
            "  - kg_neighbors: Explore entity relations and facts (parameters: entity_id, hops, predicate_filter)",
            "  - kg_path: Find shortest path between two entities (parameters: entity1, entity2)",
            "  - kg_query: Find entities by relation type (parameters: predicate, value, subject_type)",
            f"  Available predicates: occupation, position_held, political_party, educated_at, employer, "
            f"affiliated_with, has_role, mentioned_with, quoted_by, headquarters_location",
        ])
    except Exception:
        pass  # KG not available

    return "\n".join(lines)


def format_tool_history(tool_history: list[dict]) -> str:
    """Format tool history for the LLM prompt.

    For index queries, includes the full result detail so the LLM can reason
    over the retrieved content (not just the summary count).
    """
    if not tool_history:
        return "No tools used yet."

    lines = []
    for i, entry in enumerate(tool_history, 1):
        tool = entry.get("tool", "unknown")
        result = entry.get("result_summary", "")
        time_s = entry.get("execution_time", 0)
        status = "OK" if entry.get("success", False) else "FAILED"
        lines.append(f"  {i}. {tool} [{status}, {time_s:.1f}s] → {result}")

        # Include full detail for index queries so agent can reason over content
        detail = entry.get("detail", "")
        if detail:
            lines.append(f"     Results:\n{detail}")

    return "\n".join(lines)


# ============================================================================
# Graph Nodes
# ============================================================================

def understand_request(state: PipelineState) -> PipelineState:
    """Analyze the user's request and prepare for execution."""
    # Extract user_request from messages if not provided directly
    # (LangGraph Studio sends messages, not user_request)
    user_request = state.get("user_request", "")
    if not user_request:
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_request = msg.content
                break
            elif isinstance(msg, dict) and msg.get("role") == "human":
                user_request = msg.get("content", "")
                break
    state["user_request"] = user_request

    video_path = state.get("video_path", "")
    image_paths = state.get("image_paths", [])

    # Detect input type
    if image_paths and not video_path:
        input_type = "image"
    elif video_path:
        ext = Path(video_path).suffix.lower() if video_path else ""
        if ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
            input_type = "image"
            image_paths = [video_path]
            video_path = ""
        else:
            input_type = "video"
    else:
        input_type = "video"

    # Get video metadata if we have a video
    video_metadata = {}
    if video_path and input_type == "video":
        ffmpeg = get_ffmpeg_tools()
        if ffmpeg:
            metadata = ffmpeg.get_video_metadata(video_path)
            if "error" not in metadata:
                video_metadata = metadata

    # Get GPU availability
    config = ConfigManager().get_config()
    gpu_available = config.execution.gpu_available if hasattr(config, 'execution') else False

    # Build task understanding with LLM
    llm = get_llm()
    available_tools = format_available_tools(gpu_available)

    video_info = ""
    if video_metadata:
        video_info = (
            f"\nVideo: {video_metadata.get('filename', 'unknown')}, "
            f"duration: {video_metadata.get('duration_formatted', 'unknown')} "
            f"({video_metadata.get('duration_seconds', 0):.1f}s), "
            f"resolution: {video_metadata.get('video', {}).get('width', 0)}x"
            f"{video_metadata.get('video', {}).get('height', 0)}"
        )
    elif image_paths:
        video_info = f"\nImages: {len(image_paths)} image(s) provided"

    # Detect QA mode early so we can use the right prompt
    user_req = state.get("user_request", "").strip()
    is_question = (
        user_req.endswith("?")
        or user_req.lower().startswith(("who ", "what ", "where ", "when ", "which ", "how ", "why ",
                                         "is ", "are ", "was ", "were ", "did ", "do ", "does ",
                                         "can ", "could ", "name ", "find ", "list ",
                                         "tell me", "describe ", "explain ", "show me",
                                         "what about ", "look up "))
    )
    mode = "qa" if is_question else "pipeline"

    if mode == "qa":
        prompt = f"""You are a knowledge graph QA agent for broadcast news video archives.
The user is asking a question. Briefly describe how you will answer it using the Knowledge Graph.

Question: {user_req}

The Knowledge Graph contains {_get_kg_stats_summary()} from PBS NewsHour broadcasts.
Available KG operations: kg_search (find entities by name), kg_neighbors (explore entity relations),
kg_query (find entities by relation, e.g. educated_at=Harvard), kg_path (find connections between entities).

Describe your plan in 1-2 sentences."""
    else:
        prompt = f"""You are a CLAMS video/image analysis agent. Analyze this request and describe what processing is needed.

User Request: {user_req}

Input Type: {input_type}{video_info}

Available tools:
{available_tools}

Provide a brief (2-3 sentence) description of what tools you plan to use and why.
Focus on the analysis goal and the tool sequence that would achieve it."""

    response = llm.invoke([HumanMessage(content=prompt)])

    state["input_type"] = input_type
    state["video_path"] = video_path
    state["image_paths"] = image_paths
    state["video_metadata"] = video_metadata
    state["mode"] = mode
    state["messages"] = [AIMessage(content=f"Task Analysis:\n{response.content}")]

    return state


def await_approval(state: PipelineState) -> PipelineState:
    """Pause for user approval before executing."""
    # Get the task analysis from the last message
    last_msg = state["messages"][-1].content if state["messages"] else "No analysis available."

    user_response = interrupt({
        "type": "approval_request",
        "task_analysis": last_msg,
        "input_type": state.get("input_type", "unknown"),
        "video_path": state.get("video_path", ""),
        "image_count": len(state.get("image_paths", [])),
        "message": "Review the task analysis above. Reply 'approve' to start execution, or 'reject' to cancel."
    })

    approved = str(user_response).lower().strip() in (
        "approve", "yes", "y", "ok", "run", "go", "start"
    )
    state["execution_approved"] = approved
    state["messages"] = [AIMessage(
        content=f"Execution {'approved' if approved else 'rejected'} by user."
    )]

    return state


def agent_step(state: PipelineState) -> PipelineState:
    """Core reasoning node: decide what tool to use next."""
    llm = get_llm()
    config = ConfigManager().get_config()
    gpu_available = config.execution.gpu_available if hasattr(config, 'execution') else False

    iteration = state.get("iteration", 0)
    max_depth = state.get("max_depth", 10)

    # Build the prompt
    mmif_summary = summarize_mmif_state(state.get("current_mmif", {}))
    available_tools = format_available_tools(gpu_available)
    history = format_tool_history(state.get("tool_history", []))

    input_info = ""
    if state.get("video_path"):
        input_info = f"Video: {state['video_path']}"
    if state.get("image_paths"):
        count = len(state["image_paths"])
        input_info += f"{', ' if input_info else ''}Images: {count} file(s)"

    # Check for recent failures to prevent retrying the same tool
    failed_tools_warning = ""
    tool_history_list = state.get("tool_history", [])
    failed_tools = set()
    for entry in tool_history_list:
        if not entry.get("success", False):
            failed_tools.add(entry.get("tool", ""))
    if failed_tools:
        failed_tools_warning = (
            f"\n## Failed Tools (DO NOT RETRY)\n"
            f"The following tools have already failed and should NOT be used again:\n"
            f"{', '.join(failed_tools)}\n"
            f"Use a different approach or skip this step.\n"
        )

    # Build conversation context from messages for follow-up questions
    conversation_context = ""
    all_messages = state.get("messages", [])
    if len(all_messages) > 2:  # More than just the initial analysis
        recent = []
        for msg in all_messages[-6:]:  # Last few messages
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            content = msg.content if hasattr(msg, 'content') else str(msg)
            if len(content) > 200:
                content = content[:200] + "..."
            recent.append(f"  {role}: {content}")
        if recent:
            conversation_context = "\n## Recent Conversation\n" + "\n".join(recent) + "\n"

    prompt = f"""You are a CLAMS video/image analysis agent. Decide what tool to use next.

## User Request
{state.get('user_request', '')}
{conversation_context}
## Current State
Input: {input_info}
Iteration: {iteration + 1}/{max_depth}

## Current MMIF Summary
{mmif_summary}

## Tool History
{history}

## Available Tools
{available_tools}
{failed_tools_warning}
## Mode: {state.get('mode', 'pipeline')}

## Patterns
{'''### QA Mode — Use Knowledge Graph tools ONLY
IMPORTANT: Use ONLY use_kg operations. Do NOT use use_index, use_tool, or use_ffmpeg.

Step-by-step examples:

Q: "What political party is Putin in?"
1. {{"action": "use_kg", "operation": "kg_neighbors", "parameters": {{"entity_id": "Putin"}}, "reasoning": "Look up Putin's relations"}}
   → Result shows political_party relation
2. {{"action": "done", "answer": "Putin is not affiliated with a major party (independent)", "reasoning": "Found answer"}}

Q: "Who from Harvard appeared in the news?"
1. {{"action": "use_kg", "operation": "kg_query", "parameters": {{"predicate": "educated_at", "value": "Harvard"}}, "reasoning": "Find people educated at Harvard"}}
   → Result: list of people
2. {{"action": "done", "answer": "People from Harvard include: ...", "reasoning": "Found list"}}

Q: "What is the connection between Lincoln and Bush?"
1. {{"action": "use_kg", "operation": "kg_path", "parameters": {{"entity1": "Lincoln", "entity2": "Bush"}}, "reasoning": "Find path"}}

CRITICAL: If tool history already has the data you need, output done with the answer. Do NOT repeat calls.''' if state.get('mode') == 'qa' else '''### Pipeline Mode
- **Full video analysis**: swt-detection → whisper (ASR) → text-slicer → OCR → index → extract_entities → enrich_entities
- **NER on indexed content**: use_index with extract_entities operation
- **Semantic search**: After indexing + enrichment, use_index search
- text-slicer requires both TextDocument (from whisper/OCR) and TimeFrame (from swt-detection) inputs
- After the pipeline completes, the video is auto-indexed. Use extract_entities then enrich_entities for full entity grounding.'''}
- Entity enrichment uses layered lookup: Wikidata (structured facts like positions/roles) → Wikipedia (prose) → web search (fallback).
- **IMPORTANT**: If a tool fails, do NOT retry it. Try a different approach or skip it.

## Instructions
Based on the user's request and what has been processed so far, decide what to do next.

Respond with ONLY a JSON object (no other text):
- To run a CLAMS app: {{"action": "use_tool", "tool": "app-name", "parameters": {{}}, "reasoning": "brief explanation"}}
- To run FFmpeg: {{"action": "use_ffmpeg", "operation": "extract_frames", "parameters": {{"fps": 1}}, "reasoning": "brief explanation"}}
- To query the video index: {{"action": "use_index", "operation": "search", "parameters": {{"query": "..."}}, "reasoning": "brief explanation"}}
  (operations: list_videos, search, browse_timeline, filter_segments, get_video_summary, extract_entities)
  IMPORTANT: Use list_videos first if you need video IDs. Use exact video IDs from the available tools list.
- To enrich the index with entity descriptions: {{"action": "use_web", "operation": "enrich_entities", "parameters": {{"video_id": "..."}}, "reasoning": "brief explanation"}}
- To look up an entity on Wikipedia: {{"action": "use_web", "operation": "wikipedia_lookup", "parameters": {{"entity": "..."}}, "reasoning": "brief explanation"}}
- To look up structured facts from Wikidata: {{"action": "use_web", "operation": "wikidata_lookup", "parameters": {{"entity": "..."}}, "reasoning": "brief explanation"}}
- To search PubMed for biomedical literature: {{"action": "use_web", "operation": "pubmed_search", "parameters": {{"query": "..."}}, "reasoning": "brief explanation"}}
- To search the web: {{"action": "use_web", "operation": "web_search", "parameters": {{"query": "..."}}, "reasoning": "brief explanation"}}
- To search the knowledge graph by name: {{"action": "use_kg", "operation": "kg_search", "parameters": {{"query": "entity name"}}, "reasoning": "..."}}
- To find entities by relation (e.g. "who studied at Harvard"): {{"action": "use_kg", "operation": "kg_query", "parameters": {{"predicate": "educated_at", "value": "Harvard"}}, "reasoning": "..."}}
  Common predicates: occupation, position_held, political_party, educated_at, employer, affiliated_with, has_role, mentioned_with, headquarters_location
- To explore an entity's relations: {{"action": "use_kg", "operation": "kg_neighbors", "parameters": {{"entity_id": "..."}}, "reasoning": "..."}}
- To find path between two entities: {{"action": "use_kg", "operation": "kg_path", "parameters": {{"entity1": "...", "entity2": "..."}}, "reasoning": "..."}}
- If the task is complete: {{"action": "done", "reasoning": "brief explanation of what was accomplished", "answer": "the answer if this is a question"}}

Choose based on what has already been done and what still needs to be done to fulfill the user's request.
IMPORTANT: If the tool history already contains the answer to the question, output "done" with the answer immediately. Do NOT repeat a tool call that already succeeded.
If no more processing is needed, output "done"."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the LLM's decision
    try:
        content = response.content.strip()
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        action = json.loads(content.strip())
    except (json.JSONDecodeError, IndexError):
        logger.warning(f"Could not parse agent action: {response.content}")
        action = {"action": "done", "reasoning": "Could not determine next action."}

    # Dedup guard: if this exact action was already executed successfully, force "done"
    action_key = json.dumps({"a": action.get("action"), "o": action.get("operation"),
                             "p": action.get("parameters")}, sort_keys=True)
    for prev in tool_history_list:
        prev_key = json.dumps({"a": prev.get("tool", "").split(":")[-1] if ":" in prev.get("tool", "") else prev.get("tool"),
                               "o": prev.get("tool", "").split(":")[0] if ":" in prev.get("tool", "") else "",
                               "p": prev.get("parameters")}, sort_keys=True)
        # Simpler: check if same operation and params were already tried
        if (prev.get("success") and
            action.get("operation", "") == prev.get("tool", "").replace("kg:", "").replace("web:", "") and
            action.get("parameters") == prev.get("parameters")):
            logger.info(f"Dedup guard: action already executed, forcing done")
            # Synthesize answer from previous result
            answer = prev.get("result_summary", "")
            detail = prev.get("detail", "")
            if detail:
                answer = f"{answer}\n\nResults:\n{detail[:500]}"
            action = {"action": "done", "reasoning": "Results already obtained.",
                      "answer": answer}
            break

    state["next_action"] = action
    state["messages"] = [AIMessage(
        content=f"Agent decision (step {iteration + 1}): {action.get('reasoning', '')}"
    )]

    return state


def execute_tool_node(state: PipelineState) -> PipelineState:
    """Execute the tool selected by agent_step."""
    from utils.clams_executor import get_clams_executor, UnifiedCLAMSExecutor

    action = state.get("next_action", {})
    action_type = action.get("action", "done")
    iteration = state.get("iteration", 0)
    tool_history = list(state.get("tool_history", []))
    execution_results = list(state.get("execution_results", []))
    execution_errors = list(state.get("execution_errors", []))
    current_mmif = state.get("current_mmif", {})

    result_summary = ""
    success = False
    exec_time = 0.0
    index_detail = ""

    if action_type == "use_tool":
        # Execute a CLAMS app
        tool_name = action.get("tool", "")
        parameters = action.get("parameters", {})

        # Redirect spacy-wrapper to the built-in extract_entities operation
        if "spacy" in tool_name.lower():
            vid = ""
            if state.get("video_path"):
                vid = Path(state["video_path"]).stem
            logger.info(f"Redirecting spacy-wrapper to extract_entities for {vid}")
            try:
                from utils.video_index import extract_entities_from_transcripts
                t_start = time.time()
                count = extract_entities_from_transcripts(vid)
                exec_time = time.time() - t_start
                success = True
                result_summary = f"Extracted {count} entities from transcripts (used built-in spaCy instead of CLAMS app)"
                index_detail = f"Ran spaCy NER on indexed ASR transcripts. Found {count} named entities."
            except Exception as e:
                exec_time = 0
                result_summary = f"Entity extraction failed: {str(e)}"
                index_detail = ""
                logger.error(f"extract_entities failed: {e}", exc_info=True)

            tool_history.append({
                "tool": "index:extract_entities",
                "parameters": {"video_id": vid},
                "result_summary": result_summary,
                "success": success,
                "execution_time": exec_time,
                "detail": index_detail,
            })
            # Skip normal CLAMS execution
            state["current_mmif"] = current_mmif
            state["tool_history"] = tool_history
            state["execution_results"] = execution_results
            state["execution_errors"] = execution_errors
            state["iteration"] = iteration + 1
            msg_content = f"Tool result: {result_summary}"
            if index_detail:
                msg_content += f"\n\n{index_detail}"
            state["messages"] = [AIMessage(content=msg_content)]
            return state

        # Initialize MMIF if needed
        if not current_mmif:
            video_path = state.get("video_path", "")
            image_paths = state.get("image_paths", [])
            if video_path and image_paths:
                current_mmif = UnifiedCLAMSExecutor.create_initial_mmif_combined(
                    video_path=video_path, image_paths=image_paths
                )
            elif image_paths:
                current_mmif = UnifiedCLAMSExecutor.create_initial_mmif_for_images(image_paths)
            elif video_path:
                current_mmif = UnifiedCLAMSExecutor.create_initial_mmif(video_path)
            else:
                execution_errors.append("No video or image input provided")
                state["execution_errors"] = execution_errors
                state["next_action"] = {"action": "done", "reasoning": "No input files."}
                state["messages"] = [AIMessage(content="Error: No input files available.")]
                return state

        try:
            executor = get_clams_executor()
            t_start = time.time()
            result = executor.execute_app(
                app_name=tool_name,
                input_mmif=current_mmif,
                parameters=parameters
            )
            exec_time = time.time() - t_start

            execution_results.append(result.to_dict())

            if result.success and result.output_mmif:
                current_mmif = result.output_mmif
                success = True
                ann_count = result.annotations_created
                result_summary = f"{ann_count} annotations created"
            else:
                error_msg = f"{tool_name} failed: {result.error}"
                execution_errors.append(error_msg)
                result_summary = f"FAILED: {result.error}"

        except Exception as e:
            exec_time = time.time() - t_start if 't_start' in dir() else 0
            error_msg = f"{tool_name} error: {str(e)}"
            execution_errors.append(error_msg)
            result_summary = f"ERROR: {str(e)}"
            logger.error(error_msg, exc_info=True)

        tool_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "result_summary": result_summary,
            "success": success,
            "execution_time": exec_time,
        })

    elif action_type == "use_index":
        # Query the video index
        operation = action.get("operation", "search")
        parameters = action.get("parameters", {})
        index = get_video_index_singleton()
        index_detail = ""  # Full results for the LLM to reason over

        if index is None:
            execution_errors.append("Video index not available")
            result_summary = "FAILED: Video index not available"
        else:
            try:
                t_start = time.time()

                if operation == "search":
                    query = parameters.get("query", "")
                    top_k = parameters.get("top_k", 10)
                    vid = parameters.get("video_id")
                    results = index.search(query, top_k=top_k, video_id=vid)
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"Found {len(results)} segments"
                    if results:
                        detail_lines = []
                        for r in results:
                            meta = r.get("metadata", {})
                            dist = r.get("distance")
                            dist_str = f" (score={dist:.3f})" if dist is not None else ""
                            detail_lines.append(
                                f"- [{meta.get('video_id', '?')} "
                                f"{meta.get('start_ms', 0)}-{meta.get('end_ms', 0)}ms "
                                f"{meta.get('scene_label', '')}]{dist_str}\n"
                                f"  {r.get('summary', '')}"
                            )
                        index_detail = "\n".join(detail_lines)

                elif operation == "browse_timeline":
                    vid = parameters.get("video_id", "")
                    start = parameters.get("start_ms", 0)
                    end = parameters.get("end_ms", -1)
                    results = index.browse_timeline(vid, start, end)
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"Found {len(results)} segments in timeline"
                    detail_lines = []
                    for seg in results:
                        detail_lines.append(
                            f"- [{seg['start_ms']}-{seg['end_ms']}ms] "
                            f"{seg.get('scene_label', '')}: "
                            f"{seg.get('summary', '')[:300]}"
                        )
                    index_detail = "\n".join(detail_lines)

                elif operation == "filter_segments":
                    vid = parameters.get("video_id", "")
                    label = parameters.get("scene_label")
                    has_text = parameters.get("has_text")
                    results = index.filter_segments(vid, scene_label=label, has_text=has_text)
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"Found {len(results)} matching segments"
                    detail_lines = []
                    for seg in results:
                        detail_lines.append(
                            f"- [{seg['start_ms']}-{seg['end_ms']}ms] "
                            f"{seg.get('scene_label', '')}: "
                            f"{seg.get('summary', '')[:300]}"
                        )
                    index_detail = "\n".join(detail_lines)

                elif operation == "extract_entities":
                    from utils.video_index import extract_entities_from_transcripts
                    vid = parameters.get("video_id", "")
                    if not vid and state.get("video_path"):
                        vid = Path(state["video_path"]).stem
                    count = extract_entities_from_transcripts(vid)
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"Extracted {count} entities from transcripts for {vid}"
                    index_detail = f"Ran spaCy NER on indexed ASR transcripts. Found {count} named entities (PERSON, ORG, GPE, etc.)."

                elif operation == "list_videos":
                    videos = index.get_indexed_videos()
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"{len(videos)} indexed videos"
                    index_detail = "\n".join(f"- {v}" for v in videos)

                elif operation == "get_video_summary":
                    vid = parameters.get("video_id", "")
                    summary = index.get_video_summary(vid)
                    exec_time = time.time() - t_start
                    success = "error" not in summary
                    if success:
                        result_summary = (
                            f"{summary['total_segments']} segments, "
                            f"{summary['total_ocr_chars']} OCR chars, "
                            f"{summary['total_entities']} entities"
                        )
                        index_detail = (
                            f"Video: {summary['video_id']}\n"
                            f"Path: {summary['video_path']}\n"
                            f"Duration: {summary['duration_ms']}ms\n"
                            f"Total segments: {summary['total_segments']}\n"
                            f"Labels: {json.dumps(summary['label_counts'])}\n"
                            f"OCR text: {summary['total_ocr_chars']} chars\n"
                            f"ASR text: {summary['total_asr_chars']} chars\n"
                            f"Entities: {summary['total_entities']} "
                            f"({json.dumps(summary['entity_type_counts'])})"
                        )
                    else:
                        result_summary = f"FAILED: {summary.get('error', 'unknown error')}"

                else:
                    exec_time = time.time() - t_start
                    result_summary = f"Unknown index operation: {operation}"

            except Exception as e:
                exec_time = time.time() - t_start if 't_start' in dir() else 0
                error_msg = f"Index {operation} error: {str(e)}"
                execution_errors.append(error_msg)
                result_summary = f"ERROR: {str(e)}"
                logger.error(error_msg, exc_info=True)

        tool_history.append({
            "tool": f"index:{operation}",
            "parameters": parameters,
            "result_summary": result_summary,
            "success": success,
            "execution_time": exec_time,
            "detail": index_detail,
        })

    elif action_type == "use_web":
        # Web search, Wikipedia/Wikidata lookup, PubMed, or entity enrichment
        from utils.video_index import (
            wikipedia_lookup, web_search_entity, wikidata_lookup, pubmed_search,
        )
        operation = action.get("operation", "")
        parameters = action.get("parameters", {})

        try:
            t_start = time.time()

            if operation == "enrich_entities":
                vid = parameters.get("video_id", "")
                index = get_video_index_singleton()
                if index is None:
                    execution_errors.append("Video index not available")
                    result_summary = "FAILED: Video index not available"
                else:
                    grounded = index.enrich_entities(vid)
                    exec_time = time.time() - t_start
                    found = sum(1 for v in grounded.values() if v)
                    success = True
                    result_summary = f"Grounded {found}/{len(grounded)} entities for {vid}"
                    detail_lines = []
                    for ent, desc in grounded.items():
                        if desc:
                            detail_lines.append(f"- {ent}: {desc[:150]}")
                        else:
                            detail_lines.append(f"- {ent}: (no description found)")
                    index_detail = "\n".join(detail_lines)

            elif operation == "wikipedia_lookup":
                entity = parameters.get("entity", "")
                desc = wikipedia_lookup(entity)
                exec_time = time.time() - t_start
                success = True
                if desc:
                    result_summary = f"Found Wikipedia article for '{entity}'"
                    index_detail = f"{entity}: {desc}"
                else:
                    result_summary = f"No Wikipedia article found for '{entity}'"

            elif operation == "wikidata_lookup":
                entity = parameters.get("entity", "")
                desc = wikidata_lookup(entity)
                exec_time = time.time() - t_start
                success = True
                if desc:
                    result_summary = f"Found Wikidata entry for '{entity}'"
                    index_detail = f"{entity}: {desc}"
                else:
                    result_summary = f"No Wikidata entry found for '{entity}'"

            elif operation == "pubmed_search":
                query = parameters.get("query", "")
                max_results = parameters.get("max_results", 5)
                result_text = pubmed_search(query, max_results=max_results)
                exec_time = time.time() - t_start
                success = True
                if result_text:
                    result_summary = f"Found PubMed results for '{query}'"
                    index_detail = result_text
                else:
                    result_summary = f"No PubMed results for '{query}'"

            elif operation == "web_search":
                query = parameters.get("query", "")
                try:
                    from duckduckgo_search import DDGS
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=5))
                    exec_time = time.time() - t_start
                    success = True
                    result_summary = f"Found {len(results)} web results"
                    detail_lines = []
                    for r in results:
                        detail_lines.append(
                            f"- {r.get('title', '')}: {r.get('body', '')[:200]}"
                        )
                    index_detail = "\n".join(detail_lines)
                except ImportError:
                    exec_time = time.time() - t_start
                    result_summary = "FAILED: duckduckgo-search not installed"
                    execution_errors.append(result_summary)

            else:
                exec_time = time.time() - t_start
                result_summary = f"Unknown web operation: {operation}"

        except Exception as e:
            exec_time = time.time() - t_start if 't_start' in dir() else 0
            error_msg = f"Web {operation} error: {str(e)}"
            execution_errors.append(error_msg)
            result_summary = f"ERROR: {str(e)}"
            logger.error(error_msg, exc_info=True)

        tool_history.append({
            "tool": f"web:{operation}",
            "parameters": parameters,
            "result_summary": result_summary,
            "success": success,
            "execution_time": exec_time,
            "detail": index_detail,
        })

    elif action_type == "use_kg":
        # Execute a Knowledge Graph operation
        operation = action.get("operation", "")
        parameters = action.get("parameters", {})

        # Auto-detect operation from parameters if not specified
        if not operation:
            if "entity1" in parameters and "entity2" in parameters:
                operation = "kg_path"
            elif "predicate" in parameters:
                operation = "kg_query"
            elif "entity_id" in parameters:
                operation = "kg_neighbors"
            elif "query" in parameters:
                operation = "kg_search"
        t_start = time.time()

        try:
            if operation == "kg_search":
                result = kg_search(
                    parameters.get("query", ""),
                    entity_type=parameters.get("entity_type"),
                )
            elif operation == "kg_neighbors":
                result = kg_neighbors(
                    parameters.get("entity_id", ""),
                    hops=parameters.get("hops", 1),
                    predicate_filter=parameters.get("predicate_filter"),
                )
            elif operation == "kg_path":
                result = kg_path(
                    parameters.get("entity1", ""),
                    parameters.get("entity2", ""),
                )
            elif operation == "kg_query":
                result = kg_query(
                    parameters.get("predicate", ""),
                    value=parameters.get("value"),
                    subject_type=parameters.get("subject_type"),
                )
            else:
                result = {"success": False, "result_summary": f"Unknown KG operation: {operation}", "detail": ""}

            exec_time = time.time() - t_start
            success = result.get("success", False)
            result_summary = result.get("result_summary", "")
            index_detail = result.get("detail", "")

        except Exception as e:
            exec_time = time.time() - t_start
            error_msg = f"KG {operation} error: {str(e)}"
            execution_errors.append(error_msg)
            result_summary = f"ERROR: {str(e)}"
            logger.error(error_msg, exc_info=True)

        tool_history.append({
            "tool": f"kg:{operation}",
            "parameters": parameters,
            "result_summary": result_summary,
            "success": success,
            "execution_time": exec_time,
            "detail": index_detail,
        })

    elif action_type == "use_ffmpeg":
        # Execute an FFmpeg operation
        operation = action.get("operation", "")
        parameters = action.get("parameters", {})
        ffmpeg = get_ffmpeg_tools()

        if ffmpeg is None:
            execution_errors.append("FFmpeg not available")
            result_summary = "FAILED: FFmpeg not available"
        else:
            video_path = state.get("video_path", "")
            image_paths = list(state.get("image_paths", []))

            try:
                t_start = time.time()

                if operation == "extract_frames":
                    fps = parameters.get("fps", 1.0)
                    max_frames = parameters.get("max_frames", 100)
                    result = ffmpeg.extract_frames(video_path, fps=fps, max_frames=max_frames)
                    exec_time = time.time() - t_start
                    if result.get("success"):
                        frames = result.get("frames", [])
                        image_paths.extend(frames)
                        state["image_paths"] = image_paths
                        success = True
                        result_summary = f"Extracted {len(frames)} frames"
                        # Add frames as ImageDocument entries in MMIF
                        if not current_mmif:
                            current_mmif = UnifiedCLAMSExecutor.create_initial_mmif_combined(
                                video_path=video_path, image_paths=frames
                            )
                        else:
                            # Add new ImageDocuments to existing MMIF
                            doc_id = len(current_mmif.get("documents", [])) + 1
                            for frame_path in frames:
                                abs_path = Path(frame_path).absolute()
                                current_mmif.setdefault("documents", []).append({
                                    "@type": "http://mmif.clams.ai/vocabulary/ImageDocument/v1",
                                    "properties": {
                                        "id": f"d{doc_id}",
                                        "mime": "image/jpeg",
                                        "location": f"file://{abs_path}"
                                    }
                                })
                                doc_id += 1
                    else:
                        result_summary = f"FAILED: {result.get('error', 'unknown error')}"

                elif operation == "extract_audio":
                    fmt = parameters.get("format", "wav")
                    result = ffmpeg.extract_audio(video_path, format=fmt)
                    exec_time = time.time() - t_start
                    if result.get("success"):
                        success = True
                        result_summary = f"Audio extracted to {result.get('output_path', 'unknown')}"
                    else:
                        result_summary = f"FAILED: {result.get('error', 'unknown error')}"

                elif operation == "trim_video":
                    start = parameters.get("start", 0)
                    end = parameters.get("end", 60)
                    result = ffmpeg.trim_video(video_path, start_time=start, end_time=end)
                    exec_time = time.time() - t_start
                    if result.get("success"):
                        state["video_path"] = result["output_path"]
                        success = True
                        result_summary = f"Video trimmed to {start}-{end}s"
                    else:
                        result_summary = f"FAILED: {result.get('error', 'unknown error')}"

                elif operation == "generate_thumbnail_sprite":
                    result = ffmpeg.generate_thumbnail_sprite(video_path)
                    exec_time = time.time() - t_start
                    if result.get("success"):
                        sprite_path = result.get("sprite_path", "")
                        if sprite_path:
                            image_paths.append(sprite_path)
                            state["image_paths"] = image_paths
                        success = True
                        result_summary = f"Sprite generated: {sprite_path}"
                    else:
                        result_summary = f"FAILED: {result.get('error', 'unknown error')}"

                elif operation == "get_video_info":
                    result = ffmpeg.get_video_metadata(video_path)
                    exec_time = time.time() - t_start
                    if "error" not in result:
                        state["video_metadata"] = result
                        success = True
                        dur = result.get("duration_formatted", "unknown")
                        res = f"{result.get('video', {}).get('width', 0)}x{result.get('video', {}).get('height', 0)}"
                        result_summary = f"Video info: {dur}, {res}"
                    else:
                        result_summary = f"FAILED: {result.get('error', 'unknown error')}"
                else:
                    result_summary = f"Unknown FFmpeg operation: {operation}"

            except Exception as e:
                exec_time = time.time() - t_start if 't_start' in dir() else 0
                error_msg = f"FFmpeg {operation} error: {str(e)}"
                execution_errors.append(error_msg)
                result_summary = f"ERROR: {str(e)}"
                logger.error(error_msg, exc_info=True)

        tool_history.append({
            "tool": f"ffmpeg:{operation}",
            "parameters": parameters,
            "result_summary": result_summary,
            "success": success,
            "execution_time": exec_time,
        })

    # Update state
    state["current_mmif"] = current_mmif
    state["tool_history"] = tool_history
    state["execution_results"] = execution_results
    state["execution_errors"] = execution_errors
    state["iteration"] = iteration + 1

    # For index/web/kg queries, include the full results so the LLM can reason over them
    msg_content = f"Tool result: {result_summary}"
    if action_type in ("use_index", "use_web", "use_kg") and index_detail:
        msg_content += f"\n\n{index_detail}"
    state["messages"] = [AIMessage(content=msg_content)]

    return state


def summarize_results(state: PipelineState) -> PipelineState:
    """Summarize execution results and save final MMIF."""
    tool_history = state.get("tool_history", [])
    errors = state.get("execution_errors", [])
    current_mmif = state.get("current_mmif", {})

    lines = ["# Execution Results", ""]

    if not tool_history and not errors:
        if not state.get("execution_approved", False):
            lines.append("Execution was not approved.")
        else:
            lines.append("No tools were executed.")
        state["execution_summary"] = "\n".join(lines)
        state["execution_complete"] = True
        state["messages"] = [AIMessage(content="\n".join(lines))]
        return state

    # Tool execution summary
    if tool_history:
        lines.append("## Tools Used")
        total_time = 0.0
        for entry in tool_history:
            status = "OK" if entry.get("success") else "FAILED"
            exec_time = entry.get("execution_time", 0)
            total_time += exec_time
            lines.append(
                f"- **{entry['tool']}** [{status}, {exec_time:.1f}s]: "
                f"{entry.get('result_summary', '')}"
            )
        lines.extend(["", f"**Total execution time:** {total_time:.1f}s", ""])

    # Errors
    if errors:
        lines.append("## Warnings/Errors")
        for err in errors:
            lines.append(f"- {err}")
        lines.append("")

    # For QA mode, include the answer from the final agent action
    if state.get("mode") == "qa":
        last_action = state.get("next_action", {})
        answer = last_action.get("answer", last_action.get("reasoning", ""))
        if answer:
            lines.append("## Answer")
            lines.append(answer)
            lines.append("")

    # Save final MMIF (skip for QA mode — no MMIF produced)
    if state.get("mode") != "qa" and current_mmif and current_mmif.get("views"):
        output_dir = Path("data/pipeline_outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"output_{timestamp}.mmif")

        with open(output_path, "w") as f:
            json.dump(current_mmif, f, indent=2)

        state["output_mmif_path"] = output_path
        lines.append("## Output")
        lines.append(f"MMIF saved to: `{output_path}`")

        # Auto-index the video and enrich entities
        video_path = state.get("video_path", "")
        if video_path:
            try:
                index = get_video_index_singleton()
                if index:
                    vid_id = Path(video_path).stem
                    index_doc = index.build_from_mmif(vid_id, video_path, current_mmif, source_mmif_path=output_path)
                    lines.append(f"Video indexed as: `{vid_id}`")

                    # Auto-enrich named entities via Wikipedia
                    seg_list = index_doc.get("segments", [])
                    has_entities = any(s.get("named_entities") for s in seg_list)
                    if has_entities:
                        try:
                            grounded = index.enrich_entities(vid_id)
                            found = sum(1 for v in grounded.values() if v)
                            lines.append(
                                f"Entity enrichment: grounded {found}/{len(grounded)} "
                                f"entities via Wikipedia"
                            )
                        except Exception as e:
                            logger.warning(f"Entity enrichment failed: {e}")
                            lines.append(f"Entity enrichment failed: {e}")
            except Exception as e:
                logger.warning(f"Auto-indexing failed: {e}")
                lines.append(f"Auto-indexing failed: {e}")

        for view in current_mmif.get("views", []):
            app = view.get("metadata", {}).get("app", "unknown")
            if "/" in app:
                app = app.split("/")[-2] if app.endswith("/") else app.split("/")[-1]
            ann_count = len(view.get("annotations", []))
            lines.append(f"  - {app}: {ann_count} annotations")

    summary = "\n".join(lines)
    state["execution_summary"] = summary
    state["execution_complete"] = True
    state["messages"] = [AIMessage(content=summary)]

    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_approval(state: PipelineState) -> str:
    """Route based on user approval."""
    if state.get("execution_approved"):
        return "agent_step"
    return "end"


def route_after_agent_step(state: PipelineState) -> str:
    """Route based on agent's decision."""
    action = state.get("next_action", {})
    action_type = action.get("action", "done")
    iteration = state.get("iteration", 0)
    max_depth = state.get("max_depth", 10)

    # Check termination conditions
    if action_type == "done":
        return "summarize_results"
    if iteration >= max_depth:
        logger.info(f"Max depth {max_depth} reached, stopping.")
        return "summarize_results"

    # Continue the loop
    if action_type in ("use_tool", "use_ffmpeg", "use_index", "use_web", "use_kg"):
        return "execute_tool"

    # Unknown action type — stop
    return "summarize_results"


# ============================================================================
# Graph Construction
# ============================================================================

def create_pipeline_graph():
    """Create the iterative CLAMS agent graph."""
    # IMPORTANT: Initialize video index BEFORE any LLM calls.
    # ChromaDB's embedding model conflicts with Ollama if loaded after.
    try:
        get_video_index_singleton()
    except Exception:
        pass  # OK if not available

    # Other singletons are lazily initialized on first use inside graph nodes.

    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("understand_request", understand_request)
    workflow.add_node("await_approval", await_approval)
    workflow.add_node("agent_step", agent_step)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("summarize_results", summarize_results)

    # Entry point
    workflow.set_entry_point("understand_request")

    # Edges — QA mode skips approval, pipeline mode requires it
    def route_after_understanding(state: PipelineState) -> str:
        if state.get("mode") == "qa":
            return "agent_step"
        return "await_approval"

    workflow.add_conditional_edges("understand_request", route_after_understanding, {
        "agent_step": "agent_step",
        "await_approval": "await_approval",
    })

    workflow.add_conditional_edges("await_approval", route_after_approval, {
        "agent_step": "agent_step",
        "end": END,
    })

    workflow.add_conditional_edges("agent_step", route_after_agent_step, {
        "execute_tool": "execute_tool",
        "summarize_results": "summarize_results",
    })

    # execute_tool loops back to agent_step
    workflow.add_edge("execute_tool", "agent_step")

    workflow.add_edge("summarize_results", END)

    # When running under LangGraph API (langgraph dev / deploy), the platform
    # provides its own checkpointer — passing MemorySaver here causes a
    # ValueError.  We only add MemorySaver when running standalone (e.g. tests,
    # TUI, direct Python invocation) so that interrupt() still works.
    import os
    running_in_langgraph_api = (
        os.environ.get("LANGGRAPH_API")
        or os.environ.get("LANGGRAPH_API_URL")
        or "langgraph_runtime" in str(os.environ.get("_", ""))
    )
    # Also detect by checking if langgraph_runtime_inmem is already imported
    import sys
    if "langgraph_runtime_inmem" in sys.modules:
        running_in_langgraph_api = True

    if running_in_langgraph_api:
        return workflow.compile()
    else:
        return workflow.compile(checkpointer=MemorySaver())


# ============================================================================
# Entry Point for LangGraph Studio
# ============================================================================

graph = create_pipeline_graph()
