# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLAMS-Agent is a tool-use driven LLM agent for orchestrating audiovisual processing workflows on the CLAMS platform. The agent builds hierarchical structured video indexes over archival broadcast content (AAPB) by selecting and composing specialized CLAMS tools, then reasons over those indexes to answer questions about video content.

This is a thesis research prototype. The three core research components are:
1. **Agent-orchestrated index building** — LangGraph agent selects and sequences CLAMS tools to construct multi-layer video indexes (transcripts, scene boundaries, on-screen text, visual descriptions)
2. **QA benchmark** — Evaluation dataset of question-answer pairs derived from AAPB archival annotations (`qa-data/`)
3. **Fine-tuned reasoning** — SFT training data with chain-of-thought tool-use trajectories for teaching smaller models to navigate indexes (`training_data/`)

## Development Commands

### Running the Agent

```bash
# Primary entry point — LangGraph Studio (web UI)
langgraph dev

# Terminal UI — Textual chat interface
python tui.py                        # Basic usage
python tui.py --video /path/to.mp4   # Pre-set video path
python tui.py --gpu --max-depth 5    # With GPU tools, limit iterations
python tui.py --debug                # Debug logging to tui.log

# Environment setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (Optional) Configure environment
cp .env.example .env
```

### Testing

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run tests
python -m pytest tests/                              # All tests
python -m pytest tests/test_app_directory.py         # App directory tests
python -m pytest tests/test_ffmpeg_tools.py          # FFmpeg tools tests
python -m pytest tests/ --cov=utils --cov-report=html  # With coverage
```

### Training Data Generation

```bash
cd training_data
python generate_sft_data.py    # Generate SFT training examples
```

### QA Benchmark

```bash
cd qa-data
python convert_annotations.py  # Convert AAPB annotations to QA pairs
```

## Architecture

### Core Data Flow

```
User Request (LangGraph Studio)
    |
    v
agent.py — Iterative tool-use StateGraph:
    understand_request → await_approval (interrupt) → agent_step ⟲ execute_tool → summarize_results
                                                       ↑              ↓
                                                       └──── loop ────┘
    |
    |--- CLAMSToolbox (utils/clams_tools.py) — fetches tool metadata from https://apps.clams.ai/
    |--- EvaluationRAG (utils/evaluation_rag.py) — retrieves empirical tool performance data
    |--- UnifiedCLAMSExecutor (utils/clams_executor.py) — executes CLAMS apps (CLI/Docker/HTTP)
    |--- FFmpegTools (utils/ffmpeg_tools.py) — video/image preprocessing
    |
    v
MMIF Output (multi-view, saved to data/pipeline_outputs/)
```

### Key Components

**`agent.py`** — Primary LangGraph agent (iterative tool-use loop):
- Custom `StateGraph` with 5 nodes: `understand_request`, `await_approval`, `agent_step`, `execute_tool`, `summarize_results`
- `agent_step ⟲ execute_tool` cycle: LLM decides next tool, executes it, reviews MMIF output, loops until done or `max_depth` reached
- Uses `interrupt()` at `await_approval` for human-in-the-loop plan approval before execution
- `PipelineState` TypedDict tracks MMIF state, tool history, iteration count across the loop
- GPU-aware tool filtering: only shows tools the system can run (based on `config.json` `gpu_available`)
- Supports video input (VideoDocument), image input (ImageDocument), and mixed (FFmpeg-extracted frames → image tools)
- Compiled with `MemorySaver` checkpointer for `interrupt()` support
- Supports both Ollama and OpenAI LLM backends
- Entry point: `langgraph dev` (configured in `langgraph.json`)

**`utils/clams_tools.py`** — Tool discovery:
- `CLAMSToolbox` class fetches metadata from CLAMS app directory (`https://apps.clams.ai/appdir.json`)
- Caches in `data/app_directory.json`
- Provides input/output type information for MMIF pipeline compatibility

**`utils/clams_executor.py`** — Unified CLAMS execution engine:
- Runs CLAMS apps via CLI, Docker, or HTTP backends
- Backend selection via `utils/execution_backend.py`
- Handles MMIF input/output serialization
- `create_initial_mmif()` / `create_initial_mmif_for_images()` / `create_initial_mmif_combined()` — create initial MMIF documents for video, image, or mixed inputs

**`utils/evaluation_rag.py`** — Evidence-based tool selection:
- TF-IDF retrieval over markdown evaluation reports from `aapb-evaluations` sibling repo
- Provides empirical performance data (CER, WER, F1) to guide tool selection
- Used by agent.py detection/extraction nodes

**`utils/frame_review.py`** — Human-in-the-loop review:
- `FrameReviewManager` manages review sessions for OCR/detection outputs
- `HumanInTheLoopAgent` provides review request/completion workflow
- `reviews_to_mmif_view()` writes human review feedback back as CLAMS-style MMIF views
- `export_reviewed_mmif()` produces complete MMIF with human review annotations

**`utils/ffmpeg_tools.py`** — Video preprocessing:
- Frame extraction, audio extraction, video trimming, format conversion
- Video metadata retrieval (duration, resolution, codec)

**`utils/config.py`** — Configuration management:
- `LLMConfig` class (provider, model, temperature, etc.)
- `ConfigManager` with JSON persistence (`config.json`)
- Supports Ollama and OpenAI providers

**`utils/advanced_reasoning.py`** — Static knowledge tools for pipeline reasoning

**`utils/model_providers.py`** — LLM provider abstraction layer

**`utils/gpu_classification.py`** — Static map of CLAMS apps to GPU requirements

**`utils/experiment_tracker.py`** — Experiment tracking for evaluation runs

### Thesis-Aligned Components

**`training_data/`** — SFT data generation pipeline:
- Generates chain-of-thought-with-tool trajectories for fine-tuning
- Targets training smaller models to perform tool-use reasoning

**`qa-data/`** — QA benchmark dataset:
- Question-answer pairs derived from AAPB archival annotations
- Used for evaluating agent QA accuracy over video indexes

**`modelling/`** — Fine-tuning plan and training code

**`CLAMS_AGENT_SPEC.md`** — Full thesis vision and system specification

### Configuration

`ConfigManager` (`utils/config.py`) manages LLM settings:
- Provider: `ollama` or `openai`
- Model name (default: `qwen2.5:1.5b` via Ollama)
- Temperature, top_p parameters
- GPU availability flag
- Saves to `config.json`

**Required environment variable**: `OPENAI_API_KEY` (when using OpenAI provider)

### File Layout

```
agent.py                    # LangGraph Studio graph (primary entry point)
langgraph.json              # LangGraph Studio config
config.json                 # Runtime LLM configuration
tui.py                      # Terminal UI (Textual, connected to agent.py graph)
run_experiment.py           # CLI for LLM experiments

utils/
  clams_tools.py            # Tool discovery + app directory
  clams_executor.py         # Unified CLAMS execution engine
  execution_backend.py      # CLI/Docker/HTTP backends
  config.py                 # Configuration management
  evaluation_rag.py         # Evidence-based tool selection (thesis core)
  frame_review.py           # Human-in-the-loop MMIF review
  ffmpeg_tools.py           # Video preprocessing
  advanced_reasoning.py     # Static knowledge tools
  model_providers.py        # LLM provider abstraction
  experiment_tracker.py     # Experiment tracking
  gpu_classification.py     # GPU classification map
  download_app_directory.py # App directory bootstrapping

data/
  app_directory.json        # Cached CLAMS app metadata

training_data/              # SFT data generation pipeline
qa-data/                    # QA benchmark from AAPB annotations
modelling/                  # Fine-tuning plan and code
docs/                       # Additional documentation
tests/                      # Test suite
```

## Key Patterns

### MMIF Type Compatibility

Tool compatibility checking uses MMIF types:
- Input/output types extracted from URIs like `http://mmif.clams.ai/vocabulary/TimeFrame/v5`
- Common patterns: `VideoDocument` -> `TimeFrame` -> `TextDocument` -> `NamedEntity`

### Evidence-Based Tool Selection

The agent uses `EvaluationRAG` to ground tool choices in empirical data:
- Retrieves evaluation reports for task types (e.g., `TimeFrameLabeling`, `TextRecognition`)
- Provides precision/recall/F1 scores to LLM prompts
- Example: prefers VLMs over traditional OCR for degraded archival footage (7% CER vs 90% CER)

### Human Review as CLAMS App

Human review feedback is modeled as a CLAMS app view in MMIF:
- App identifier: `http://apps.clams.ai/human-review/v1`
- Produces `Annotation/v1` (review status) and `TextDocument/v1` (corrected text)
- Links back to source annotations via `source_annotation` property
