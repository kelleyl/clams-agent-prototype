# CLAMS Agent - Architecture & Pipeline Diagram

## Full System Architecture

```mermaid
graph TB
    subgraph USER["User Interfaces"]
        LGS["LangGraph Studio<br/>(langgraph dev)"]
        TUI["Terminal UI<br/>(tui.py)"]
        API["REST API<br/>localhost:8123"]
    end

    subgraph AGENT["LangGraph Agent (agent.py)"]
        direction TB
        UR["understand_request<br/>Detect input type, metadata,<br/>build task understanding"]
        AA["await_approval<br/>interrupt() for HITL"]
        AS["agent_step<br/>LLM decides next action"]
        ET["execute_tool_node<br/>Route to executor"]
        SR["summarize_results<br/>Save MMIF, auto-index"]

        UR --> AA --> AS
        AS -->|"use_tool / use_ffmpeg<br/>use_index / use_web"| ET
        ET -->|"loop until done<br/>or max_depth"| AS
        AS -->|"done"| SR
    end

    subgraph LLM["LLM Backend"]
        OLL["Ollama<br/>qwen2.5:7b"]
        OAI["OpenAI<br/>(optional)"]
    end

    subgraph TOOLS["Tool Discovery & Selection"]
        CT["CLAMSToolbox<br/>(clams_tools.py)<br/>App directory + I/O types"]
        ER["EvaluationRAG<br/>(evaluation_rag.py)<br/>Empirical CER/WER/F1"]
        GC["GPU Classification<br/>(gpu_classification.py)<br/>Filter by hardware"]
    end

    subgraph EXEC["Execution Backends (clams_executor.py)"]
        CLI["CLI Executor<br/>python cli.py<br/>/Users/kelleylynch/clams/*"]
        DOCK["Docker Executor<br/>ghcr.io/clamsproject/*"]
        HTTP["HTTP Executor<br/>REST endpoints"]
    end

    subgraph CLAMS_APPS["CLAMS Apps"]
        SWT["swt-detection<br/>Scene boundaries"]
        WHIS["whisper-wrapper<br/>ASR transcription"]
        TSLICE["text-slicer<br/>Chunk by scene"]
        OCR["OCR Apps<br/>doctr / tesseract / parseq"]
        VLM["VLM Apps<br/>smolvlm2-captioner"]
        PYSCD["pyscenedetect<br/>Shot detection"]
    end

    subgraph FFMPEG["FFmpeg Tools (ffmpeg_tools.py)"]
        FE["extract_frames"]
        FA["extract_audio"]
        FT["trim_video"]
        FM["get_metadata"]
    end

    subgraph INDEX["Video Index (video_index.py)"]
        direction TB
        BUILD["build_from_mmif()<br/>Parse MMIF views"]
        SEG["SegmentEntry<br/>scene_label, ocr_text,<br/>asr_transcript, entities,<br/>grounded_entities, summary"]
        SEARCH["search(query)<br/>Semantic retrieval"]
        BROWSE["browse_timeline()<br/>Range query"]
        FILTER["filter_segments()<br/>Structured filters"]
        NER["extract_entities()<br/>Built-in spaCy NER"]
        ENRICH["enrich_entities()<br/>Wikidata + Wikipedia"]
    end

    subgraph STORAGE["Data Storage"]
        MMIF_OUT["MMIF Files<br/>data/pipeline_outputs/"]
        JSON_IDX["JSON Indexes<br/>data/video_indexes/*.json"]
        CHROMA["ChromaDB<br/>data/chroma_db/"]
        APP_DIR["App Directory<br/>data/app_directory.json"]
    end

    subgraph EMBED["Embeddings"]
        NOMIC["Ollama<br/>nomic-embed-text"]
    end

    subgraph QA["QA Evaluation Pipeline (qa-data/)"]
        direction TB
        CONV["Converters<br/>scene_recognition.py<br/>chyron_detection.py"]
        GEN["generate_index_qa.py<br/>6 categories, MC + free-form"]
        SCHEMA["qa_schema.json<br/>Validation"]
        REVIEW["review_app.py<br/>Gradio Web UI"]

        subgraph REVIEW_TABS["Review App Tabs"]
            TAB1["HITL Review<br/>Approve / Reject / Edit"]
            TAB2["ELO Comparison<br/>Pairwise A vs B"]
            TAB3["Dataset Stats<br/>Overview dashboard"]
        end
    end

    subgraph QA_OUT["QA Outputs"]
        JSONL["index_qa.jsonl<br/>index_qa_mc.jsonl"]
        REVIEWS["reviews/<br/>review_decisions.jsonl<br/>elo_comparisons.jsonl<br/>elo_ratings.json"]
    end

    subgraph TRAINING["Training Data (training_data/)"]
        TGEN["generate_tasks.py<br/>Synthetic tasks"]
        TSFT["prepare_sft_data.py<br/>CoT trajectories"]
        TPIPE["run_pipeline.py<br/>Full pipeline"]
        TOUT["sft_data_train.jsonl<br/>sft_data_val.jsonl<br/>sft_data_test.jsonl"]
    end

    subgraph EXTERNAL["External Services"]
        WIKI["Wikidata API"]
        WPEDIA["Wikipedia API"]
        WEBSRCH["Web Search<br/>(DuckDuckGo)"]
    end

    %% User → Agent
    LGS --> AGENT
    TUI --> AGENT
    API --> AGENT

    %% Agent → LLM
    AS --> LLM
    UR --> LLM

    %% Agent → Tools
    AS --> CT
    AS --> ER
    AS --> GC
    CT --> AS

    %% Execute → Backends
    ET --> EXEC
    ET --> FFMPEG
    ET --> INDEX
    EXEC --> CLAMS_APPS

    %% CLAMS → MMIF
    CLAMS_APPS --> MMIF_OUT

    %% Auto-index on completion
    SR --> BUILD
    BUILD --> SEG
    SEG --> JSON_IDX
    SEG --> CHROMA
    CHROMA --> NOMIC

    %% Index operations
    SEARCH --> CHROMA
    NER --> JSON_IDX
    ENRICH --> EXTERNAL

    %% QA Pipeline
    JSON_IDX --> GEN
    GEN --> JSONL
    CONV --> JSONL
    JSONL --> REVIEW
    REVIEW --> REVIEW_TABS
    REVIEW --> REVIEWS
    GEN --> SCHEMA

    %% Training
    TGEN --> TSFT --> TPIPE --> TOUT

    %% Styling
    classDef primary fill:#4F46E5,stroke:#3730A3,color:white
    classDef storage fill:#059669,stroke:#047857,color:white
    classDef external fill:#D97706,stroke:#B45309,color:white
    classDef qa fill:#7C3AED,stroke:#6D28D9,color:white
    classDef tool fill:#0284C7,stroke:#0369A1,color:white

    class UR,AA,AS,ET,SR primary
    class MMIF_OUT,JSON_IDX,CHROMA,APP_DIR storage
    class WIKI,WPEDIA,WEBSRCH,OLL,OAI external
    class GEN,REVIEW,CONV,TAB1,TAB2,TAB3 qa
    class SWT,WHIS,TSLICE,OCR,VLM,PYSCD tool
```

## Agent State Graph (Detailed)

```mermaid
stateDiagram-v2
    [*] --> understand_request

    understand_request --> await_approval: Task plan ready

    await_approval --> agent_step: User approves (interrupt resume)

    agent_step --> execute_tool: use_tool / use_ffmpeg
    agent_step --> execute_index: use_index
    agent_step --> execute_web: use_web
    agent_step --> summarize_results: done
    agent_step --> summarize_results: max_depth reached

    execute_tool --> agent_step: Result → tool_history
    execute_index --> agent_step: Result → tool_history
    execute_web --> agent_step: Result → tool_history

    summarize_results --> auto_index: Save MMIF
    auto_index --> [*]: Index updated

    state execute_tool {
        [*] --> check_spacy_redirect
        check_spacy_redirect --> builtin_ner: spacy in name
        check_spacy_redirect --> select_backend: other
        select_backend --> cli_exec: CLI
        select_backend --> docker_exec: Docker
        select_backend --> http_exec: HTTP
        cli_exec --> parse_mmif
        docker_exec --> parse_mmif
        http_exec --> parse_mmif
        builtin_ner --> [*]
        parse_mmif --> [*]
    }

    state execute_index {
        [*] --> route_operation
        route_operation --> search: search
        route_operation --> browse: browse_timeline
        route_operation --> filter: filter_segments
        route_operation --> ner: extract_entities
        route_operation --> enrich: enrich_entities
        search --> [*]
        browse --> [*]
        filter --> [*]
        ner --> [*]
        enrich --> [*]
    }
```

## Data Flow: Video → Index → QA → Evaluation

```mermaid
flowchart LR
    subgraph INPUT["Input"]
        VID["Video File<br/>.mp4"]
    end

    subgraph PIPELINE["Agent Pipeline"]
        direction TB
        P1["1. swt-detection<br/>Scene boundaries"]
        P2["2. whisper<br/>ASR transcription"]
        P3["3. text-slicer<br/>Chunk by scene"]
        P4["4. OCR / VLM<br/>On-screen text"]
        P5["5. spaCy NER<br/>Named entities"]
        P6["6. Wikipedia/Wikidata<br/>Entity grounding"]
        P1 --> P2 --> P3 --> P4 --> P5 --> P6
    end

    subgraph MMIF["MMIF Output"]
        V1["View: TimeFrame<br/>(scene boundaries)"]
        V2["View: TextDocument<br/>(transcripts)"]
        V3["View: TextDocument<br/>(OCR text)"]
        V4["View: NamedEntity<br/>(people, orgs, places)"]
    end

    subgraph IDX["Video Index"]
        direction TB
        S1["Segment 1<br/>bars | 0:00-1:12"]
        S2["Segment 2<br/>slate | 1:32-1:39"]
        S3["Segment 3<br/>person & chyron | 2:33-5:05<br/>ASR: 'Tom Sakata...'<br/>Entities: Tom Sakata (PERSON)"]
        SN["...35 segments"]
        S1 --- S2 --- S3 --- SN
    end

    subgraph VECDB["ChromaDB"]
        EMB["nomic-embed-text<br/>embeddings"]
    end

    subgraph QAGEN["QA Generation"]
        direction TB
        Q1["scene_classification (31)<br/>'What scene at 0:00:12?'"]
        Q2["text_extraction (20)<br/>'What is said between...'"]
        Q3["named_entity (13)<br/>'What people are mentioned?'"]
        Q4["entity_linking (58)<br/>'What is Hawaii Visitors Bureau?'"]
        Q5["cross_annotation (39)<br/>'Who speaks during chyron?'"]
    end

    subgraph EVAL["Evaluation"]
        MC["Multiple Choice<br/>49 MC questions"]
        FF["Free-form<br/>112 open questions"]
        REV["HITL Review<br/>Approve/Reject/Edit"]
        ELO["ELO Tournament<br/>Model A vs B"]
    end

    VID --> PIPELINE --> MMIF --> IDX
    IDX --> VECDB
    IDX --> QAGEN
    QAGEN --> MC
    QAGEN --> FF
    MC --> REV
    FF --> REV
    REV --> ELO
```

## Typical Pipeline Execution

```mermaid
sequenceDiagram
    participant U as User
    participant A as Agent (LangGraph)
    participant L as LLM (qwen2.5:7b)
    participant C as CLAMS Apps
    participant I as Video Index
    participant W as Wikidata/Wikipedia

    U->>A: "Analyze this video and tell me who appears"
    A->>L: understand_request (classify task)
    L-->>A: Plan: swt → whisper → NER → enrich
    A->>U: await_approval (interrupt)
    U->>A: Approved

    loop agent_step ⟲ execute_tool (max 10 iterations)
        A->>L: What tool next?
        L-->>A: use_tool: swt-detection
        A->>C: swt-detection(video.mp4)
        C-->>A: MMIF + TimeFrame annotations

        A->>L: What tool next?
        L-->>A: use_tool: whisper-wrapper
        A->>C: whisper(video.mp4)
        C-->>A: MMIF + TextDocument annotations

        A->>L: What tool next?
        L-->>A: use_index: extract_entities
        A->>I: spaCy NER on transcripts
        I-->>A: 45 entities found

        A->>L: What tool next?
        L-->>A: use_index: enrich_entities
        A->>W: Wikidata/Wikipedia lookups
        W-->>A: Entity descriptions + QIDs

        A->>L: What tool next?
        L-->>A: done
    end

    A->>I: Auto-index (build_from_mmif)
    A->>U: Summary: Found 35 segments,<br/>12 people including Tom Sakata,<br/>Christina Kemmer, Stephen Boyle...
```

## File Structure

```
clams-agent-prototype/
├── agent.py                     # LangGraph StateGraph (5 nodes, iterative loop)
├── langgraph.json               # LangGraph Studio config
├── config.json                  # Runtime config (LLM, execution, index)
├── tui.py                       # Terminal UI (Textual)
│
├── utils/
│   ├── clams_tools.py           # Tool discovery (apps.clams.ai)
│   ├── clams_executor.py        # Unified executor (CLI/Docker/HTTP)
│   ├── execution_backend.py     # Backend implementations
│   ├── video_index.py           # Hybrid JSON + ChromaDB index
│   ├── evaluation_rag.py        # Evidence-based tool selection
│   ├── ffmpeg_tools.py          # Video preprocessing
│   ├── frame_review.py          # HITL frame review → MMIF
│   ├── advanced_reasoning.py    # Knowledge tools
│   ├── model_providers.py       # LLM abstraction
│   ├── config.py                # Configuration management
│   ├── gpu_classification.py    # App GPU requirements
│   └── experiment_tracker.py    # Experiment logging
│
├── data/
│   ├── app_directory.json       # Cached CLAMS app metadata
│   ├── video_indexes/           # Per-video JSON indexes
│   ├── chroma_db/               # ChromaDB vector store
│   └── pipeline_outputs/        # Generated MMIF files
│
├── qa-data/
│   ├── generate_index_qa.py     # QA pair generation (6 categories)
│   ├── review_app.py            # Gradio review + ELO app
│   ├── schema/qa_schema.json    # QA entry validation schema
│   ├── converters/              # AAPB annotation → QA converters
│   ├── raw/                     # Generated JSONL files
│   └── reviews/                 # HITL decisions + ELO data
│
├── training_data/
│   ├── generate_tasks.py        # Synthetic task generation
│   ├── prepare_sft_data.py      # CoT trajectory creation
│   ├── run_pipeline.py          # Full SFT pipeline
│   └── prompts/                 # LLM prompt templates
│
└── docs/
    └── architecture.md          # This file
```
