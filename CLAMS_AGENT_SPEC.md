# CLAMS-Agent: Statement of Goals

## Vision

CLAMS-Agent is a tool-use driven LLM agent system for orchestrating audiovisual processing workflows on the CLAMS (Computational Linguistics Applications for Multimedia Services) platform. Unlike existing video understanding agents (e.g., VideoAgent) that rely on a narrow set of implicit tools (CLIP for retrieval, VLM for captioning), CLAMS-Agent treats tool selection as a first-class reasoning task. The agent dynamically selects from a heterogeneous toolkit of specialized, well-tested models—each with known performance characteristics—to construct complete metadata extraction workflows rather than single answers.

The system targets full-length television programs (30-60+ minutes) from cultural heritage archives, requiring genuine multimodal integration of speech recognition, on-screen text extraction, temporal segmentation, and visual analysis. This contrasts with existing benchmarks (EgoSchema, NExT-QA) that focus on short-form (3-5 minute) visual reasoning about human activities.

## Core Architectural Principles

### 1. Tool-Use as Primary Mechanism
The agent reasons about which tools to deploy based on task requirements. Available tools include ASR (Whisper), OCR, vision-language models (LLaVA, SmolVLM2), shot detection (TransNetV2), chyron/slate/credits extraction, and ffmpeg for media manipulation. The agent should be able to incorporate new CLAMS apps as they become available.

### 2. Evidence-Based Tool Selection
The agent incorporates empirical performance data to guide pipeline configuration. For example, when extracting text from historical broadcast chyrons, the agent should prefer VLMs over traditional OCR based on experimental findings (7% CER vs 90% CER on archival content). Tool selection should be grounded in quantitative evidence, not arbitrary preferences.

### 3. MMIF-Native Output
All workflows produce structured annotations in the Multi-Media Interchange Format (MMIF), ensuring interoperability with downstream CLAMS tools and archival cataloging systems. The agent produces reusable workflow artifacts, not just natural language answers.

### 4. Modular Efficiency
Leverage purpose-built models for specific subtasks rather than forcing everything through a single VLM. Use Whisper for speech content, TransNetV2 for shot boundaries, specialized extractors for chyrons/slates/credits. Avoid redundant computation—don't caption a frame when you need its speech transcript.

## Required Capabilities

### Workflow Orchestration
- Accept natural language task descriptions (e.g., "Extract and index all spoken names and on-screen text from this video")
- Decompose complex requests into subtasks mapped to appropriate tools
- Construct and execute multi-stage pipelines with proper data flow between tools
- Handle less clearly defined tasks through clarifying dialogue (e.g., "Identify moments where reporters are conducting interviews")

### Tool Integration
- Interface with existing CLAMS apps via their APIs
- Support YAML-based configuration for LM/MLLM apps (prompt templates, context construction, processing modes)
- Manage tool-specific parameters (e.g., restricting processing to temporal segments like "last three minutes for credits")
- Generate ffmpeg commands for frame extraction and media manipulation

### Evidence-Based Reasoning
- Maintain knowledge of tool performance characteristics on archival content
- Select tools based on empirical data (e.g., VLM > OCR for degraded historical footage)
- Route multi-task requests efficiently (e.g., use single VLM for both transcription and structured extraction when appropriate)

### Experiment Generation
- Generate and coordinate evaluation experiments from natural language descriptions
- Support multi-frame stitching, comparative model evaluation, and dataset manipulation tasks
- Produce code for transforming evaluation datasets and annotation formats
- Include automated quality assurance (detect malformed questions, ambiguous answer choices)

### Human-in-the-Loop
- Support interactive refinement of workflow configurations through dialogue
- Flag outputs requiring human validation
- Prioritize recall over precision when human review is expected
- Enable validation of intermediate outputs before proceeding

## Domain Context

The target content is historical broadcast material from the American Archive of Public Broadcasting—television and radio programming originally recorded on analog tape and later digitized. This content exhibits:
- Quality degradations from decades-old magnetic tape (noise, artifacts, color distortion)
- Supplementary content (b-roll, opening slates, SMPTE color bars, credits)
- Complex multimodal information (speech references on-screen content, fragmented story structures across segments)
- Chyrons with diverse fonts, backgrounds, and network logos
- Hawaiian content with underrepresented diacritics (ʻokina, kahakō) and names

## Handling LLM Challenges

The agent should address common large model failure modes:
- **Non-determinism**: Retry when outputs appear malformed or gibberish
- **Hallucination**: Use context/RAG to ground outputs; consider LLM-as-judge for verification
- **Distraction**: Filter irrelevant information (e.g., background text, watermarks when extracting chyrons)

## Framework Considerations

Candidate agent frameworks include LangGraph, smolagents, pydantic_ai, and DSPy. Key requirements:
- Support for human-in-the-loop as workflow component
- Graph/workflow representation for complex pipelines
- Integration with existing Pydantic models (if using pydantic_ai)
- Potential for prompt optimization using gold-standard annotation data
