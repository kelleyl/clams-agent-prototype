# CLAMS-Agent — Research Roadmap

## Phase 1: Index Building

- [x] **End-to-end pipeline execution** — Iterative tool-use loop in agent.py with execute_tool node calling `clams_executor.py` and FFmpegTools
- [ ] **Multi-layer index construction** — Build hierarchical indexes with layers: ASR transcripts, shot boundaries, on-screen text (OCR/VLM), visual captions, named entities
- [ ] **MMIF output validation** — Verify produced MMIF documents are well-formed and interoperable across CLAMS tools
- [ ] **Temporal alignment** — Ensure all index layers share consistent temporal references (millisecond timestamps)
- [ ] **Index persistence** — Store and retrieve built indexes for downstream QA

## Phase 2: QA Benchmark

- [ ] **Expand QA dataset** — Convert more AAPB annotations to question-answer pairs (`qa-data/`)
- [ ] **Question type coverage** — Ensure benchmark covers factoid, temporal, entity, and multi-hop question types
- [ ] **Baseline evaluation** — Run agent QA over indexes and measure accuracy against gold annotations
- [ ] **Ablation studies** — Evaluate QA accuracy with different index layer combinations (e.g., ASR-only vs ASR+OCR vs full index)

## Phase 3: Fine-Tuning

- [ ] **SFT trajectory generation** — Generate chain-of-thought-with-tool training examples (`training_data/`)
- [ ] **Model training** — Fine-tune smaller models (e.g., Qwen 2.5, Llama 3.2) on tool-use trajectories
- [ ] **Training infrastructure** — Set up training pipeline with experiment tracking (`modelling/`)
- [ ] **Evaluation loop** — Compare fine-tuned model QA performance against baseline (GPT-4o, Ollama models)

## Phase 4: Evaluation & Analysis

- [ ] **Systematic evaluation** — Run full evaluation pipeline across AAPB test set
- [ ] **Index layer ablations** — Measure contribution of each index layer to QA accuracy
- [ ] **Tool selection analysis** — Compare evidence-based selection (EvaluationRAG) vs. unguided selection
- [ ] **Error analysis** — Categorize failure modes (wrong tool, missing index layer, hallucination, etc.)
- [ ] **Cost/efficiency analysis** — Measure compute cost per index layer and per QA query

## Infrastructure & Agent Improvements

- [x] **Reconnect TUI** — Wire `tui.py` (Textual) to `agent.py` graph via `astream()` + `Command(resume=...)`
- [ ] **Human-in-the-loop execution** — Integrate `frame_review.py` HITL workflow into agent pipeline with LangGraph `interrupt()` nodes
- [ ] **Persistent checkpointing** — Replace LangGraph `MemorySaver` with SQLite for session persistence
- [x] **GPU-aware tool filtering** — Only offer GPU-requiring CLAMS apps when GPU is available
- [ ] **LangSmith evaluation datasets** — Create regression test cases from QA benchmark for CI

## Completed

- [x] LangGraph agent with custom `StateGraph` (agent.py)
- [x] Evidence-based tool selection via `EvaluationRAG`
- [x] CLAMS tool discovery from app directory
- [x] FFmpeg video preprocessing tools
- [x] Unified CLAMS execution engine (CLI/Docker/HTTP)
- [x] Human-in-the-loop frame review with MMIF export
- [x] Ollama and OpenAI LLM support
- [x] LangSmith tracing configuration
- [x] GPU classification map for CLAMS apps
- [x] Repo cleanup — removed Flask/React/AG-UI web frontend
- [x] Iterative tool-use execution loop (DVD/VideoAgent-style agent_step ⟲ execute_tool cycle)
- [x] GPU-aware tool filtering (hides GPU tools when GPU unavailable)
- [x] LangGraph `interrupt()` for plan-then-execute approval flow
- [x] ImageDocument MMIF creation (for image inputs and FFmpeg-extracted frames)
