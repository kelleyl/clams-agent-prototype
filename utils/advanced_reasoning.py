"""
Advanced reasoning tools for complex questions.
Includes web search, research capabilities, and multi-step reasoning.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ============================================================================
# Web Search Tools
# ============================================================================

def create_web_search_tool():
    """Create a web search tool using DuckDuckGo (free, no API key)."""
    try:
        from duckduckgo_search import DDGS

        @tool("web_search")
        def web_search(query: str, max_results: int = 5) -> str:
            """Search the web for information about video processing, CLAMS tools, or related topics.

            Use this to:
            - Find state-of-the-art techniques for video analysis
            - Research new models or approaches
            - Compare CLAMS tools with industry standards
            - Find documentation or tutorials

            Args:
                query: Search query (be specific for better results)
                max_results: Maximum number of results to return (default: 5)

            Returns:
                Search results with titles, snippets, and URLs
            """
            try:
                with DDGS() as ddgs:
                    results = list(ddgs.text(query, max_results=max_results))

                if not results:
                    return "No search results found for this query."

                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(
                        f"{i}. **{r.get('title', 'No title')}**\n"
                        f"   {r.get('body', 'No description')[:200]}...\n"
                        f"   URL: {r.get('href', 'No URL')}"
                    )

                return "\n\n".join(formatted)

            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return f"Search failed: {str(e)}"

        return web_search

    except ImportError:
        logger.warning("duckduckgo-search not installed, web search unavailable")
        return None


def create_academic_search_tool():
    """Create a tool for searching academic/research content."""
    try:
        from duckduckgo_search import DDGS

        @tool("academic_search")
        def academic_search(query: str) -> str:
            """Search for academic papers and research on video/audio processing.

            Use this to find:
            - Research papers on OCR, ASR, NER techniques
            - Benchmark comparisons
            - State-of-the-art model architectures
            - Dataset information

            Args:
                query: Research topic to search for

            Returns:
                Academic search results
            """
            try:
                # Add academic keywords to improve results
                academic_query = f"{query} research paper arxiv"

                with DDGS() as ddgs:
                    results = list(ddgs.text(academic_query, max_results=5))

                if not results:
                    return "No academic results found."

                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(
                        f"{i}. {r.get('title', 'No title')}\n"
                        f"   {r.get('body', '')[:150]}...\n"
                        f"   {r.get('href', '')}"
                    )

                return "\n\n".join(formatted)

            except Exception as e:
                return f"Academic search failed: {str(e)}"

        return academic_search

    except ImportError:
        return None


# ============================================================================
# Comparative Analysis Tools
# ============================================================================

@tool("compare_approaches")
def compare_approaches(task: str, approaches: str) -> str:
    """Compare different approaches or tools for a given task.

    Use this to help users understand trade-offs between different options.

    Args:
        task: The task being performed (e.g., "OCR", "speech transcription")
        approaches: Comma-separated list of approaches to compare

    Returns:
        Comparison analysis with pros/cons
    """
    approach_list = [a.strip() for a in approaches.split(",")]

    # Knowledge base of common trade-offs
    knowledge = {
        "whisper": {
            "pros": ["High accuracy", "Multi-language support", "Open source"],
            "cons": ["Computationally intensive", "Slower than some alternatives"],
            "best_for": "High-quality transcription where accuracy matters"
        },
        "distil-whisper": {
            "pros": ["Faster than full Whisper", "Lower resource usage"],
            "cons": ["Slightly lower accuracy", "Fewer model sizes"],
            "best_for": "Real-time or resource-constrained transcription"
        },
        "tesseract": {
            "pros": ["Mature, well-tested", "Many language packs", "Free"],
            "cons": ["Requires good preprocessing", "Struggles with complex layouts"],
            "best_for": "Clean document text, simple layouts"
        },
        "easyocr": {
            "pros": ["Deep learning based", "Good with scene text", "80+ languages"],
            "cons": ["GPU recommended", "Can be slow"],
            "best_for": "Natural scene text, varied fonts"
        },
        "paddleocr": {
            "pros": ["State-of-the-art accuracy", "Fast inference", "Good for Asian languages"],
            "cons": ["Larger model size", "More dependencies"],
            "best_for": "High-accuracy OCR, multi-language documents"
        },
        "spacy": {
            "pros": ["Fast", "Production-ready", "Good documentation"],
            "cons": ["English-focused models best", "Less customizable"],
            "best_for": "Standard NER tasks, production deployments"
        }
    }

    result_parts = [f"## Comparison for: {task}\n"]

    for approach in approach_list:
        approach_lower = approach.lower()
        matched_key = None
        for key in knowledge:
            if key in approach_lower or approach_lower in key:
                matched_key = key
                break

        if matched_key:
            info = knowledge[matched_key]
            result_parts.append(f"### {approach}")
            result_parts.append(f"**Pros:** {', '.join(info['pros'])}")
            result_parts.append(f"**Cons:** {', '.join(info['cons'])}")
            result_parts.append(f"**Best for:** {info['best_for']}")
            result_parts.append("")
        else:
            result_parts.append(f"### {approach}")
            result_parts.append("No detailed comparison data available. Consider web search.")
            result_parts.append("")

    return "\n".join(result_parts)


# ============================================================================
# Complex Reasoning Tools
# ============================================================================

@tool("analyze_video_quality")
def analyze_video_quality(metadata: str) -> str:
    """Analyze video quality and recommend preprocessing/tools accordingly.

    Use this when the user has video metadata and needs recommendations
    based on the specific characteristics of their video.

    Args:
        metadata: Video metadata as JSON or description (resolution, codec, etc.)

    Returns:
        Quality analysis and recommendations
    """
    import json

    recommendations = []
    warnings = []

    try:
        # Try to parse as JSON
        if "{" in metadata:
            meta = json.loads(metadata)
        else:
            # Parse text description
            meta = {"description": metadata}

        # Analyze resolution
        width = meta.get("video", {}).get("width", 0) or 0
        height = meta.get("video", {}).get("height", 0) or 0

        if width and height:
            if width < 480 or height < 360:
                warnings.append("Low resolution video - OCR accuracy may be reduced")
                recommendations.append("Consider upscaling or using noise reduction")
                recommendations.append("Use robust OCR models like PaddleOCR")
            elif width >= 1920:
                recommendations.append("High resolution - can use frame subsampling to speed up processing")

        # Analyze duration
        duration = meta.get("duration_seconds", 0)
        if duration > 3600:  # Over 1 hour
            recommendations.append("Long video - consider segmenting or sampling key frames")
            recommendations.append("Use scene detection to identify segments of interest")

        # Analyze codec
        codec = meta.get("video", {}).get("codec", "").lower()
        if "mpeg" in codec or "h263" in codec:
            warnings.append("Older codec detected - may have compression artifacts")
            recommendations.append("Consider preprocessing with denoising filters")

        # Check for audio
        audio = meta.get("audio")
        if audio:
            sample_rate = audio.get("sample_rate", 0)
            if sample_rate and sample_rate < 16000:
                warnings.append(f"Low audio sample rate ({sample_rate}Hz) - may affect ASR accuracy")
                recommendations.append("Consider resampling audio to 16kHz for Whisper")
        else:
            warnings.append("No audio track detected")

        # Format response
        result = ["## Video Quality Analysis\n"]

        if warnings:
            result.append("### Warnings")
            for w in warnings:
                result.append(f"- {w}")
            result.append("")

        if recommendations:
            result.append("### Recommendations")
            for r in recommendations:
                result.append(f"- {r}")
        else:
            result.append("Video quality appears good for standard processing.")

        return "\n".join(result)

    except Exception as e:
        return f"Could not analyze metadata: {str(e)}. Please provide video details."


@tool("suggest_pipeline_for_goal")
def suggest_pipeline_for_goal(goal: str, constraints: str = "") -> str:
    """Suggest a complete pipeline based on a high-level goal.

    Use this for complex requests that require multi-step reasoning
    about what tools and configurations to use.

    Args:
        goal: The user's high-level goal (e.g., "extract all text and names from news broadcasts")
        constraints: Any constraints (e.g., "fast processing", "high accuracy", "limited GPU")

    Returns:
        Detailed pipeline suggestion with reasoning
    """
    goal_lower = goal.lower()
    constraints_lower = constraints.lower() if constraints else ""

    pipeline_suggestions = []
    reasoning = []

    # Analyze goal components
    needs_text_detection = any(w in goal_lower for w in ["text", "ocr", "chyron", "caption", "title", "credit"])
    needs_speech = any(w in goal_lower for w in ["speech", "transcri", "audio", "spoken", "dialog"])
    needs_entities = any(w in goal_lower for w in ["name", "person", "entity", "people", "place", "organization"])
    needs_scenes = any(w in goal_lower for w in ["scene", "shot", "segment", "chapter"])
    needs_slate = any(w in goal_lower for w in ["slate", "title card", "opening", "credit"])

    # Analyze constraints
    prioritize_speed = any(w in constraints_lower for w in ["fast", "quick", "speed", "realtime"])
    prioritize_accuracy = any(w in constraints_lower for w in ["accurate", "quality", "best", "precise"])
    limited_resources = any(w in constraints_lower for w in ["limited", "cpu", "no gpu", "low memory"])

    # Build pipeline
    if needs_scenes:
        pipeline_suggestions.append("pyscenedetect-wrapper")
        reasoning.append("Scene detection to segment video into meaningful parts")

    if needs_slate:
        pipeline_suggestions.append("slatedetection")
        reasoning.append("Slate detection to find title cards and credits")

    if needs_text_detection:
        if "chyron" in goal_lower or "news" in goal_lower:
            pipeline_suggestions.append("chyron-detection")
            reasoning.append("Chyron detection optimized for news tickers and lower-thirds")
        else:
            pipeline_suggestions.append("swt-detection")
            reasoning.append("General text detection for scenes with text")

        # OCR selection based on constraints
        if prioritize_accuracy and not limited_resources:
            pipeline_suggestions.append("paddleocr (via external) or doctr-wrapper")
            reasoning.append("High-accuracy OCR (PaddleOCR has best evaluation scores)")
        elif prioritize_speed or limited_resources:
            pipeline_suggestions.append("tesseractocr-wrapper")
            reasoning.append("Tesseract for faster, CPU-friendly OCR")
        else:
            pipeline_suggestions.append("easyocr-wrapper")
            reasoning.append("EasyOCR as balanced choice for accuracy and compatibility")

    if needs_speech:
        if prioritize_accuracy and not limited_resources:
            pipeline_suggestions.append("whisper-wrapper (base or small model)")
            reasoning.append("Whisper-small has best WER scores in evaluations")
        elif prioritize_speed:
            pipeline_suggestions.append("distil-whisper-wrapper")
            reasoning.append("Distil-Whisper for faster transcription")
        else:
            pipeline_suggestions.append("whisper-wrapper (small model)")
            reasoning.append("Whisper-small for good accuracy/speed balance")

    if needs_entities:
        pipeline_suggestions.append("spacy-wrapper")
        reasoning.append("SpaCy NER to extract people, places, organizations")

        if "link" in goal_lower or "wikipedia" in goal_lower or "ground" in goal_lower:
            pipeline_suggestions.append("dbpedia-spotlight-wrapper")
            reasoning.append("DBpedia Spotlight to link entities to knowledge base")

    # Format response
    result = [
        f"## Suggested Pipeline for: {goal[:50]}...\n",
        "### Pipeline Steps"
    ]

    for i, (tool, reason) in enumerate(zip(pipeline_suggestions, reasoning), 1):
        result.append(f"{i}. **{tool}**")
        result.append(f"   - {reason}")

    result.append("\n### Pipeline Flow")
    result.append(f"```\n{' → '.join(pipeline_suggestions)}\n```")

    if constraints:
        result.append(f"\n### Constraints Considered")
        result.append(f"- {constraints}")

    return "\n".join(result)


# ============================================================================
# Domain Knowledge Tools
# ============================================================================

@tool("explain_metric")
def explain_metric(metric_name: str) -> str:
    """Explain what an evaluation metric means and how to interpret it.

    Args:
        metric_name: The metric to explain (e.g., "WER", "CER", "F1", "precision")

    Returns:
        Explanation of the metric with interpretation guidance
    """
    metrics = {
        "wer": {
            "full_name": "Word Error Rate",
            "formula": "(Substitutions + Deletions + Insertions) / Total Words",
            "interpretation": "Lower is better. WER of 0.10 means 10% of words had errors.",
            "good_range": "< 0.15 is good, < 0.10 is excellent for clean audio",
            "use_case": "Automatic Speech Recognition (ASR) evaluation"
        },
        "cer": {
            "full_name": "Character Error Rate",
            "formula": "(Substitutions + Deletions + Insertions) / Total Characters",
            "interpretation": "Lower is better. CER of 0.05 means 5% of characters had errors.",
            "good_range": "< 0.10 is good, < 0.05 is excellent",
            "use_case": "OCR and text recognition evaluation"
        },
        "precision": {
            "full_name": "Precision",
            "formula": "True Positives / (True Positives + False Positives)",
            "interpretation": "Higher is better. Measures how many detected items were correct.",
            "good_range": "> 0.90 is good, > 0.95 is excellent",
            "use_case": "Detection tasks (e.g., slate detection, NER)"
        },
        "recall": {
            "full_name": "Recall (Sensitivity)",
            "formula": "True Positives / (True Positives + False Negatives)",
            "interpretation": "Higher is better. Measures how many actual items were found.",
            "good_range": "> 0.85 is good, > 0.95 is excellent",
            "use_case": "Detection tasks where missing items is costly"
        },
        "f1": {
            "full_name": "F1 Score",
            "formula": "2 × (Precision × Recall) / (Precision + Recall)",
            "interpretation": "Higher is better. Harmonic mean of precision and recall.",
            "good_range": "> 0.85 is good, > 0.90 is excellent",
            "use_case": "Overall detection performance when both precision and recall matter"
        },
        "accuracy": {
            "full_name": "Accuracy",
            "formula": "(True Positives + True Negatives) / Total",
            "interpretation": "Higher is better. Overall correctness rate.",
            "good_range": "Depends on class balance; can be misleading with imbalanced data",
            "use_case": "Balanced classification tasks"
        }
    }

    metric_lower = metric_name.lower().replace("-", "").replace("_", "").replace(" ", "")

    for key, info in metrics.items():
        if key in metric_lower or metric_lower in key:
            return f"""## {info['full_name']} ({metric_name.upper()})

**Formula:** {info['formula']}

**Interpretation:** {info['interpretation']}

**Good Performance:** {info['good_range']}

**Use Case:** {info['use_case']}
"""

    return f"Unknown metric: {metric_name}. Common metrics: WER, CER, Precision, Recall, F1, Accuracy"


@tool("explain_clams_concept")
def explain_clams_concept(concept: str) -> str:
    """Explain a CLAMS-related concept or term.

    Args:
        concept: The concept to explain (e.g., "MMIF", "TimeFrame", "pipeline")

    Returns:
        Explanation of the concept
    """
    concepts = {
        "mmif": """## MMIF (Multi-Media Interchange Format)

MMIF is the standard data format used by CLAMS for exchanging multimedia annotations.

**Key Features:**
- JSON-LD based format
- Contains documents (video, audio, text) and views (annotations)
- Each view is created by a specific app
- Annotations reference documents and each other

**Structure:**
```json
{
  "documents": [...],     // Input media files
  "views": [              // Annotation layers
    {
      "app": "app-name",
      "annotations": [...]
    }
  ]
}
```
""",
        "timeframe": """## TimeFrame

A TimeFrame annotation marks a time segment in a video/audio document.

**Properties:**
- `start`: Start time in milliseconds
- `end`: End time in milliseconds
- `label`: Category label (e.g., "slate", "chyron")
- `classification`: Confidence scores per label

**Use Cases:**
- Marking scenes, slates, commercials
- Identifying segments with text
- Speech segments
""",
        "boundingbox": """## BoundingBox

A BoundingBox annotation marks a rectangular region in an image/frame.

**Properties:**
- Coordinates: top, left, bottom, right (or x, y, width, height)
- Often anchored to a TimePoint or TimeFrame
- Used with OCR for text localization

**Use Cases:**
- Text regions for OCR
- Face detection regions
- Object localization
""",
        "pipeline": """## CLAMS Pipeline

A pipeline is a sequence of CLAMS apps that process media in stages.

**Common Patterns:**
1. Detection → Extraction → Analysis
2. Scene Detection → OCR → NER
3. Audio Extraction → ASR → Entity Linking

**Pipeline Flow:**
```
Input Video → App 1 → MMIF → App 2 → MMIF → App 3 → Final MMIF
```

Each app adds its annotations to the MMIF as a new view.
""",
        "view": """## MMIF View

A view is a layer of annotations added by a single app.

**Properties:**
- `id`: Unique identifier
- `app`: The app that created it
- `contains`: Types of annotations in the view
- `annotations`: List of annotation objects

Views maintain provenance - you can trace which app created which annotations.
"""
    }

    concept_lower = concept.lower().replace("-", "").replace("_", "")

    for key, explanation in concepts.items():
        if key in concept_lower or concept_lower in key:
            return explanation

    return f"Unknown concept: {concept}. Try: MMIF, TimeFrame, BoundingBox, Pipeline, View"


# ============================================================================
# Factory Function
# ============================================================================

def create_advanced_reasoning_tools() -> List:
    """Create all advanced reasoning tools."""
    tools = [
        compare_approaches,
        analyze_video_quality,
        suggest_pipeline_for_goal,
        explain_metric,
        explain_clams_concept
    ]

    # Add web search if available
    web_search = create_web_search_tool()
    if web_search:
        tools.append(web_search)
        logger.info("Web search tool added")

    academic_search = create_academic_search_tool()
    if academic_search:
        tools.append(academic_search)
        logger.info("Academic search tool added")

    return tools
