"""
LangGraph Studio entry point for the CLAMS Pipeline Agent.

This file exposes a custom LangGraph that visualizes the CLAMS pipeline
as separate nodes for each tool category.

Run with: langgraph dev
"""

from typing import List, Optional, Literal, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
import operator
import json

from utils.clams_tools import CLAMSToolbox
from utils.config import ConfigManager
from utils.ffmpeg_tools import FFmpegTools
from utils.evaluation_rag import get_evaluation_rag, EvaluationRAG

import logging

logger = logging.getLogger(__name__)

# Initialize toolbox, ffmpeg, and evaluation RAG at module level
_toolbox = None
_ffmpeg_tools = None
_evaluation_rag = None

def get_toolbox() -> CLAMSToolbox:
    """Get or create the CLAMS toolbox singleton."""
    global _toolbox
    if _toolbox is None:
        _toolbox = CLAMSToolbox()
    return _toolbox


def get_ffmpeg_tools() -> FFmpegTools:
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


def get_eval_rag() -> EvaluationRAG:
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


def get_evaluation_context(task_type: str) -> str:
    """Get evaluation context for a specific task type."""
    rag = get_eval_rag()
    if rag is None:
        return ""

    try:
        summary = rag.get_task_summary(task_type)
        return f"\n\nHistorical Evaluation Data for {task_type}:\n{summary}"
    except Exception as e:
        logger.warning(f"Could not get evaluation context: {e}")
        return ""


def get_app_evaluation(app_name: str) -> str:
    """Get evaluation data for a specific app."""
    rag = get_eval_rag()
    if rag is None:
        return ""

    try:
        return rag.get_app_performance(app_name)
    except Exception as e:
        logger.warning(f"Could not get app evaluation: {e}")
        return ""


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
    """State for the CLAMS pipeline builder."""
    messages: Annotated[list[BaseMessage], operator.add]
    user_request: str
    video_path: str  # Path to local video file for analysis
    pipeline_steps: list[str]  # List of CLAMS tool names
    current_step_index: int
    preprocessing_tools: list[str]  # FFmpeg preprocessing steps
    detection_tools: list[str]
    extraction_tools: list[str]
    analysis_tools: list[str]
    video_metadata: dict  # Video info from FFmpeg
    pipeline_complete: bool
    final_output: str


# ============================================================================
# Helper Functions
# ============================================================================

def format_tool_info(app_name: str, app_info: dict) -> str:
    """Format tool metadata into a readable string."""
    metadata = app_info.get("metadata", {})
    description = metadata.get('description', 'No description')
    if len(description) > 150:
        description = description[:150] + "..."
    return f"- {app_name}: {description}"


def search_tools_by_category(category: str) -> list[str]:
    """Search for tools by category keyword."""
    toolbox = get_toolbox()
    matches = []

    category_keywords = {
        "detection": ["detect", "scene", "text", "slate", "chyron", "bars", "swt"],
        "extraction": ["ocr", "transcri", "whisper", "speech", "caption", "doctr", "tesseract", "easyocr"],
        "analysis": ["spacy", "ner", "entity", "keyword", "dbpedia", "understanding"]
    }

    keywords = category_keywords.get(category, [category])

    for app_name, app_info in toolbox.app_metadata.items():
        metadata = app_info.get("metadata", {})
        description = metadata.get('description', '').lower()
        name_lower = app_name.lower()

        for keyword in keywords:
            if keyword in description or keyword in name_lower:
                matches.append(app_name)
                break

    return matches


def get_tool_details(tool_name: str) -> dict:
    """Get details about a specific tool."""
    toolbox = get_toolbox()
    return toolbox.app_metadata.get(tool_name, {})


# ============================================================================
# Graph Nodes
# ============================================================================

def analyze_request(state: PipelineState) -> PipelineState:
    """Analyze the user's request and determine what pipeline is needed."""
    llm = get_llm()

    # Get available tools summary
    toolbox = get_toolbox()
    tools_summary = "\n".join([
        format_tool_info(name, info)
        for name, info in list(toolbox.app_metadata.items())[:15]
    ])

    # Check if FFmpeg is available and get video metadata if path provided
    ffmpeg = get_ffmpeg_tools()
    ffmpeg_available = ffmpeg is not None
    video_path = state.get("video_path", "")
    video_info = ""

    if video_path and ffmpeg_available:
        metadata = ffmpeg.get_video_metadata(video_path)
        if "error" not in metadata:
            state["video_metadata"] = metadata
            video_info = f"""
Video File Loaded: {metadata.get('filename', 'unknown')}
- Duration: {metadata.get('duration_formatted', 'unknown')} ({metadata.get('duration_seconds', 0):.1f}s)
- Resolution: {metadata.get('video', {}).get('width', 0)}x{metadata.get('video', {}).get('height', 0)}
- Format: {metadata.get('format', 'unknown')}
- Has Audio: {'Yes' if metadata.get('audio') else 'No'}
"""
        else:
            video_info = f"\nVideo file error: {metadata.get('error')}"

    prompt = f"""You are a CLAMS pipeline planner. Analyze this request and determine what types of tools are needed.
{video_info}

User Request: {state['user_request']}

Available CLAMS tools (partial list):
{tools_summary}

{"FFmpeg preprocessing tools are AVAILABLE for: frame extraction, audio extraction, video trimming, format conversion." if ffmpeg_available else "FFmpeg tools are NOT available."}

Based on the request, identify which categories of tools are needed:
0. PREPROCESSING tools (FFmpeg: extract frames, extract audio, trim video, convert format)
1. DETECTION tools (scene detection, text detection, slate detection, chyron detection)
2. EXTRACTION tools (OCR, speech transcription, captioning)
3. ANALYSIS tools (NER, entity linking, keyword extraction)

Respond in JSON format:
{{
    "needs_preprocessing": true/false,
    "needs_detection": true/false,
    "needs_extraction": true/false,
    "needs_analysis": true/false,
    "preprocessing_operations": ["extract_frames", "extract_audio"],
    "detection_keywords": ["keyword1", "keyword2"],
    "extraction_keywords": ["keyword1"],
    "analysis_keywords": ["keyword1"],
    "explanation": "Brief explanation of the pipeline needed"
}}

Preprocessing is useful when:
- Need to extract specific frames for OCR/detection
- Need audio-only for speech transcription
- Need to trim to a specific segment
- Need format conversion for compatibility

Only include categories that are actually needed for the request."""

    response = llm.invoke([HumanMessage(content=prompt)])

    # Parse the response
    try:
        # Extract JSON from response
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        plan = json.loads(content.strip())
    except:
        # Default plan if parsing fails
        plan = {
            "needs_preprocessing": False,
            "needs_detection": True,
            "needs_extraction": True,
            "needs_analysis": False,
            "explanation": "Default pipeline plan"
        }

    # Initialize step lists based on plan
    state["preprocessing_tools"] = [] if (plan.get("needs_preprocessing") and ffmpeg_available) else None
    state["detection_tools"] = [] if plan.get("needs_detection") else None
    state["extraction_tools"] = [] if plan.get("needs_extraction") else None
    state["analysis_tools"] = [] if plan.get("needs_analysis") else None

    state["messages"] = [AIMessage(content=f"Pipeline Analysis: {plan.get('explanation', 'Planning pipeline...')}")]

    return state


def select_preprocessing_tools(state: PipelineState) -> PipelineState:
    """Select appropriate FFmpeg preprocessing steps for the pipeline."""
    if state["preprocessing_tools"] is None:
        return state

    llm = get_llm()
    ffmpeg = get_ffmpeg_tools()

    if ffmpeg is None:
        state["preprocessing_tools"] = []
        state["messages"] = [AIMessage(content="Preprocessing: FFmpeg not available, skipping")]
        return state

    # Available FFmpeg operations
    ffmpeg_ops = """
Available FFmpeg Preprocessing Operations:
- extract_frames: Extract video frames at specified FPS (for OCR, image analysis)
- extract_audio: Extract audio track (for speech transcription)
- trim_video: Trim to specific time segment
- get_video_info: Get video metadata (duration, resolution, codec)
- convert_format: Convert video format/resolution
"""

    prompt = f"""Select the FFmpeg preprocessing operations needed for this request.

User Request: {state['user_request']}

{ffmpeg_ops}

Respond with ONLY a JSON array of operation names to use:
["extract_frames", "extract_audio"]

Choose operations that prepare the video for the CLAMS tools that will follow.
For example:
- If doing OCR, use "extract_frames"
- If doing speech transcription, use "extract_audio"
- If analyzing a specific segment, use "trim_video"

Only include operations that are actually needed."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content
        if "[" in content:
            content = content[content.index("["):content.rindex("]")+1]
        selected = json.loads(content)
        # Validate operations
        valid_ops = ["extract_frames", "extract_audio", "trim_video", "get_video_info", "convert_format"]
        selected = [op for op in selected if op in valid_ops]
    except:
        selected = []

    state["preprocessing_tools"] = selected
    state["pipeline_steps"] = state.get("pipeline_steps", []) + [f"ffmpeg:{op}" for op in selected]
    state["messages"] = [AIMessage(content=f"Preprocessing: Selected {', '.join(selected) if selected else 'none'}")]

    return state


def select_detection_tools(state: PipelineState) -> PipelineState:
    """Select appropriate detection tools for the pipeline."""
    if state["detection_tools"] is None:
        return state

    llm = get_llm()
    toolbox = get_toolbox()

    # Get detection tools
    detection_tools = search_tools_by_category("detection")
    tools_info = "\n".join([
        format_tool_info(name, toolbox.app_metadata[name])
        for name in detection_tools if name in toolbox.app_metadata
    ])

    # Get evaluation data for detection tasks
    timeframe_eval = get_evaluation_context("TimeFrameLabeling")
    timepoint_eval = get_evaluation_context("TimePointLabeling")

    prompt = f"""Select the best detection tool(s) for this request.

User Request: {state['user_request']}

Available Detection Tools:
{tools_info}

{timeframe_eval}

{timepoint_eval}

Based on the available tools and their historical performance:
- Precision: How accurate are the detections (fewer false positives)
- Recall: How complete are the detections (fewer misses)
- F1: Overall balance of precision and recall (higher is better)

Respond with ONLY a JSON array of tool names to use (1-2 tools max):
["tool-name-1", "tool-name-2"]

Choose tools that directly address the detection needs in the request, preferring those with better historical performance (higher F1)."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content
        if "[" in content:
            content = content[content.index("["):content.rindex("]")+1]
        selected = json.loads(content)
        # Validate tools exist
        selected = [t for t in selected if t in toolbox.app_metadata]
    except:
        # Default to swt-detection
        selected = ["swt-detection"] if "swt-detection" in toolbox.app_metadata else []

    state["detection_tools"] = selected
    state["pipeline_steps"] = state.get("pipeline_steps", []) + selected
    state["messages"] = [AIMessage(content=f"Detection: Selected {', '.join(selected)}")]

    return state


def select_extraction_tools(state: PipelineState) -> PipelineState:
    """Select appropriate extraction tools (OCR, transcription) for the pipeline."""
    if state["extraction_tools"] is None:
        return state

    llm = get_llm()
    toolbox = get_toolbox()

    # Get extraction tools
    extraction_tools = search_tools_by_category("extraction")
    tools_info = "\n".join([
        format_tool_info(name, toolbox.app_metadata[name])
        for name in extraction_tools if name in toolbox.app_metadata
    ])

    # Get evaluation data for OCR and ASR tasks
    ocr_eval = get_evaluation_context("TextRecognition")
    asr_eval = get_evaluation_context("AutomaticSpeechRecognition")

    prompt = f"""Select the best extraction tool(s) for this request.

User Request: {state['user_request']}
Already selected tools: {state.get('pipeline_steps', [])}

Available Extraction Tools (OCR, Transcription):
{tools_info}

{ocr_eval if "text" in state['user_request'].lower() or "ocr" in state['user_request'].lower() else ""}

{asr_eval if "speech" in state['user_request'].lower() or "transcri" in state['user_request'].lower() or "audio" in state['user_request'].lower() else ""}

Based on the available tools and their historical performance (lower error rates are better):
- For OCR: Consider CER (Character Error Rate) - lower is better
- For ASR: Consider WER (Word Error Rate) - lower is better

Respond with ONLY a JSON array of tool names to use (1-2 tools max):
["tool-name-1"]

Choose tools that handle text/speech extraction needed for the request, preferring those with better historical performance."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content
        if "[" in content:
            content = content[content.index("["):content.rindex("]")+1]
        selected = json.loads(content)
        selected = [t for t in selected if t in toolbox.app_metadata]
    except:
        selected = ["doctr-wrapper"] if "doctr-wrapper" in toolbox.app_metadata else []

    state["extraction_tools"] = selected
    state["pipeline_steps"] = state.get("pipeline_steps", []) + selected
    state["messages"] = [AIMessage(content=f"Extraction: Selected {', '.join(selected)}")]

    return state


def select_analysis_tools(state: PipelineState) -> PipelineState:
    """Select appropriate analysis tools (NER, entity linking) for the pipeline."""
    if state["analysis_tools"] is None:
        return state

    llm = get_llm()
    toolbox = get_toolbox()

    # Get analysis tools
    analysis_tools = search_tools_by_category("analysis")
    tools_info = "\n".join([
        format_tool_info(name, toolbox.app_metadata[name])
        for name in analysis_tools if name in toolbox.app_metadata
    ])

    prompt = f"""Select the best analysis tool(s) for this request.

User Request: {state['user_request']}
Already selected tools: {state.get('pipeline_steps', [])}

Available Analysis Tools (NER, Entity Linking, Keywords):
{tools_info}

Respond with ONLY a JSON array of tool names to use (1-2 tools max):
["tool-name-1"]

Choose tools that perform the analysis needed (entities, keywords, etc.)."""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content
        if "[" in content:
            content = content[content.index("["):content.rindex("]")+1]
        selected = json.loads(content)
        selected = [t for t in selected if t in toolbox.app_metadata]
    except:
        selected = ["spacy-wrapper"] if "spacy-wrapper" in toolbox.app_metadata else []

    state["analysis_tools"] = selected
    state["pipeline_steps"] = state.get("pipeline_steps", []) + selected
    state["messages"] = [AIMessage(content=f"Analysis: Selected {', '.join(selected)}")]

    return state


def build_pipeline(state: PipelineState) -> PipelineState:
    """Build the final pipeline output."""
    toolbox = get_toolbox()
    steps = state.get("pipeline_steps", [])
    video_metadata = state.get("video_metadata", {})

    if not steps:
        state["final_output"] = "No pipeline could be built for this request."
        state["pipeline_complete"] = True
        state["messages"] = [AIMessage(content=state["final_output"])]
        return state

    # Build detailed pipeline description
    output_lines = [
        f"# CLAMS Pipeline for: {state['user_request'][:50]}...",
        ""
    ]

    # Add video info if available
    if video_metadata:
        output_lines.extend([
            "## Input Video:",
            f"- **File:** {video_metadata.get('filename', 'unknown')}",
            f"- **Duration:** {video_metadata.get('duration_formatted', 'unknown')}",
            f"- **Resolution:** {video_metadata.get('video', {}).get('width', 0)}x{video_metadata.get('video', {}).get('height', 0)}",
            f"- **Format:** {video_metadata.get('format', 'unknown')}",
            ""
        ])

    output_lines.extend([
        "## Pipeline Steps:",
        ""
    ])

    for i, tool_name in enumerate(steps, 1):
        # Handle FFmpeg preprocessing steps
        if tool_name.startswith("ffmpeg:"):
            op_name = tool_name.replace("ffmpeg:", "")
            ffmpeg_descriptions = {
                "extract_frames": "Extract video frames at specified intervals for image analysis",
                "extract_audio": "Extract audio track for speech transcription",
                "trim_video": "Trim video to specific time segment",
                "get_video_info": "Get video metadata (duration, resolution, codec)",
                "convert_format": "Convert video to different format/resolution"
            }
            output_lines.append(f"### Step {i}: FFmpeg - {op_name}")
            output_lines.append(f"- **Description:** {ffmpeg_descriptions.get(op_name, 'FFmpeg operation')}")
            output_lines.append(f"- **Type:** Preprocessing")
            output_lines.append("")
            continue

        tool_info = toolbox.app_metadata.get(tool_name, {})
        metadata = tool_info.get("metadata", {})
        description = metadata.get("description", "No description")[:100]

        # Get inputs/outputs
        inputs = [inp.get('@type', '').split('/')[-1] for inp in metadata.get('input', []) if isinstance(inp, dict)]
        outputs = [out.get('@type', '').split('/')[-1] for out in metadata.get('output', []) if isinstance(out, dict)]

        output_lines.append(f"### Step {i}: {tool_name}")
        output_lines.append(f"- **Description:** {description}")
        output_lines.append(f"- **Inputs:** {', '.join(inputs) or 'VideoDocument'}")
        output_lines.append(f"- **Outputs:** {', '.join(outputs) or 'Annotations'}")
        output_lines.append("")

    output_lines.append(f"## Pipeline Flow")
    output_lines.append(f"```")
    output_lines.append(f"{' -> '.join(steps)}")
    output_lines.append(f"```")

    state["final_output"] = "\n".join(output_lines)
    state["pipeline_complete"] = True
    state["messages"] = [AIMessage(content=state["final_output"])]

    return state


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_analysis(state: PipelineState) -> str:
    """Route to the appropriate next step after analysis."""
    if state.get("preprocessing_tools") is not None:
        return "select_preprocessing"
    elif state.get("detection_tools") is not None:
        return "select_detection"
    elif state.get("extraction_tools") is not None:
        return "select_extraction"
    elif state.get("analysis_tools") is not None:
        return "select_analysis"
    else:
        return "build_pipeline"


def route_after_preprocessing(state: PipelineState) -> str:
    """Route after preprocessing tools are selected."""
    if state.get("detection_tools") is not None:
        return "select_detection"
    elif state.get("extraction_tools") is not None:
        return "select_extraction"
    elif state.get("analysis_tools") is not None:
        return "select_analysis"
    else:
        return "build_pipeline"


def route_after_detection(state: PipelineState) -> str:
    """Route after detection tools are selected."""
    if state.get("extraction_tools") is not None:
        return "select_extraction"
    elif state.get("analysis_tools") is not None:
        return "select_analysis"
    else:
        return "build_pipeline"


def route_after_extraction(state: PipelineState) -> str:
    """Route after extraction tools are selected."""
    if state.get("analysis_tools") is not None:
        return "select_analysis"
    else:
        return "build_pipeline"


# ============================================================================
# Graph Construction
# ============================================================================

def create_pipeline_graph():
    """Create the CLAMS pipeline builder graph."""

    # Initialize toolbox and ffmpeg
    get_toolbox()
    get_ffmpeg_tools()

    # Create the graph
    workflow = StateGraph(PipelineState)

    # Add nodes - these will appear in the visualization!
    workflow.add_node("analyze_request", analyze_request)
    workflow.add_node("select_preprocessing", select_preprocessing_tools)
    workflow.add_node("select_detection", select_detection_tools)
    workflow.add_node("select_extraction", select_extraction_tools)
    workflow.add_node("select_analysis", select_analysis_tools)
    workflow.add_node("build_pipeline", build_pipeline)

    # Set entry point
    workflow.set_entry_point("analyze_request")

    # Add conditional edges from analyze_request
    workflow.add_conditional_edges(
        "analyze_request",
        route_after_analysis,
        {
            "select_preprocessing": "select_preprocessing",
            "select_detection": "select_detection",
            "select_extraction": "select_extraction",
            "select_analysis": "select_analysis",
            "build_pipeline": "build_pipeline"
        }
    )

    # Add conditional edges from select_preprocessing
    workflow.add_conditional_edges(
        "select_preprocessing",
        route_after_preprocessing,
        {
            "select_detection": "select_detection",
            "select_extraction": "select_extraction",
            "select_analysis": "select_analysis",
            "build_pipeline": "build_pipeline"
        }
    )

    # Add conditional edges from select_detection
    workflow.add_conditional_edges(
        "select_detection",
        route_after_detection,
        {
            "select_extraction": "select_extraction",
            "select_analysis": "select_analysis",
            "build_pipeline": "build_pipeline"
        }
    )

    # Add conditional edges from select_extraction
    workflow.add_conditional_edges(
        "select_extraction",
        route_after_extraction,
        {
            "select_analysis": "select_analysis",
            "build_pipeline": "build_pipeline"
        }
    )

    # select_analysis always goes to build_pipeline
    workflow.add_edge("select_analysis", "build_pipeline")

    # build_pipeline ends the graph
    workflow.add_edge("build_pipeline", END)

    return workflow.compile()


# ============================================================================
# Entry Point for LangGraph Studio
# ============================================================================

# Create a wrapper that handles the input format expected by LangGraph Studio
def create_studio_graph():
    """Create the graph with input transformation for Studio."""
    base_graph = create_pipeline_graph()

    # Wrap to handle message input format
    class StudioWrapper:
        def __init__(self, graph):
            self.graph = graph

        def invoke(self, input_data, config=None):
            # Transform Studio input to our state format
            if isinstance(input_data, dict):
                if "messages" in input_data:
                    messages = input_data["messages"]
                    if messages and len(messages) > 0:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            user_request = last_message.content
                        else:
                            user_request = str(last_message)
                    else:
                        user_request = "Build a video analysis pipeline"
                else:
                    user_request = input_data.get("user_request", "Build a video analysis pipeline")
            else:
                user_request = str(input_data)

            # Extract video path if provided
            video_path = ""
            if isinstance(input_data, dict):
                video_path = input_data.get("video_path", "")

            # Create initial state
            initial_state = {
                "messages": [],
                "user_request": user_request,
                "video_path": video_path,
                "pipeline_steps": [],
                "current_step_index": 0,
                "preprocessing_tools": None,
                "detection_tools": None,
                "extraction_tools": None,
                "analysis_tools": None,
                "video_metadata": {},
                "pipeline_complete": False,
                "final_output": ""
            }

            # Run the graph
            result = self.graph.invoke(initial_state, config)

            # Return in format expected by Studio
            return {
                "messages": result.get("messages", []),
                "pipeline_steps": result.get("pipeline_steps", []),
                "final_output": result.get("final_output", "")
            }

        def stream(self, input_data, config=None):
            # For streaming, yield chunks
            result = self.invoke(input_data, config)
            yield result

        # Expose the underlying graph for visualization
        @property
        def get_graph(self):
            return self.graph.get_graph

    return StudioWrapper(base_graph)


# Export the compiled graph for LangGraph CLI/Studio
graph = create_pipeline_graph()
