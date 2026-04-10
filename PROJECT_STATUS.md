# CLAMS-Agent — Project Status

**Last Updated:** March 2025

## Current State

The repository has been refactored to align with the thesis research goals. The Flask web frontend, AG-UI streaming layer, and React visualization have been removed. The primary agent entry point is now `agent.py`, a custom LangGraph `StateGraph` run via `langgraph dev` (LangGraph Studio).

### What Changed

- **Removed**: Flask server (`app.py`), React frontend (`visualization/`), AG-UI integration, and all associated debug/test files
- **Removed**: Dead utils modules (`langgraph_agent.py`, `planning_agent.py`, `pipeline_model.py`, `pipeline_execution.py`, `agui_integration.py`, `agentic_pipeline.py`, `simulated_clams.py`)
- **Kept**: All thesis-aligned components — agent graph, CLAMS tool integration, evaluation RAG, frame review, ffmpeg tools, training data pipeline, QA benchmark
- **Updated**: `tui.py` temporarily disabled (Textual framework intact for future reconnection to `agent.py`)

### Active Components

| Component | Status | Purpose |
|-----------|--------|---------|
| `agent.py` (LangGraph Studio) | Working | Iterative tool-use agent (agent_step ⟲ execute_tool loop) |
| `utils/clams_tools.py` | Working | Tool discovery from CLAMS app directory |
| `utils/clams_executor.py` | Working | Unified CLAMS execution (CLI/Docker/HTTP) |
| `utils/evaluation_rag.py` | Working | Evidence-based tool selection |
| `utils/frame_review.py` | Working | Human-in-the-loop review + MMIF export |
| `utils/ffmpeg_tools.py` | Working | Video preprocessing |
| `training_data/` | In progress | SFT data generation pipeline |
| `qa-data/` | In progress | QA benchmark from AAPB annotations |
| `tui.py` | Working | Textual TUI connected to LangGraph agent (`python tui.py`) |

### Thesis Research Phases

1. **Index building** — Agent orchestrates CLAMS tools to build multi-layer video indexes (transcripts, scene boundaries, on-screen text, visual descriptions)
2. **QA benchmark** — Evaluate agent accuracy answering questions over indexes using AAPB-derived QA pairs
3. **Fine-tuning** — Train smaller models on SFT trajectories to perform tool-use reasoning
4. **Evaluation** — Ablation studies over different index layers and tool configurations

## Known Limitations

- Agent execution loop is wired but needs end-to-end testing with real CLAMS containers and video files
- TUI is temporarily disabled; interact via LangGraph Studio
- Training data generation and QA benchmark pipelines are under active development
- `EvaluationRAG` requires the `aapb-evaluations` sibling repo to be present
