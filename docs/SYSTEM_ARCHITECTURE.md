# CLAMS Agent System Architecture

## Overview

The CLAMS Agent system consists of two main workstreams and a unified user interface for video analysis, metadata management, and human-in-the-loop annotation.

---

## Part 1: Experimentation Framework

### 1.1 Modelling Experiments (`modelling/`)

**Purpose**: Train and evaluate agent models with different configurations, prompts, and training data.

**Tracking**: Weights & Biases (W&B)
- NOT managed in LangSmith/LangGraph/LangChain
- All experiment tracking, metrics, and artifacts in W&B
- Enables comparison of training runs, hyperparameters, data compositions

**Experiment Types**:
| Experiment | Variables | Metrics |
|------------|-----------|---------|
| Prompt optimization | DSPy optimizers, instructions, few-shot examples | Tool selection accuracy |
| Data composition | Seed vs synthetic, augmentation strategies | Workflow correctness |
| Model architecture | Base model size, LoRA rank, training duration | Task completion rate |
| RL training | Reward weights (accuracy/cost/latency) | Multi-objective Pareto |

**Directory Structure**:
```
modelling/
├── data/                    # Training data
│   ├── seed/               # Human-annotated examples
│   ├── synthetic/          # LLM-generated tasks
│   └── augmented/          # Paraphrased variations
├── training/               # Training scripts
│   ├── sft/               # Supervised fine-tuning
│   ├── rl/                # Reinforcement learning
│   └── configs/           # W&B sweep configs
├── evaluation/            # Evaluation benchmarks
│   ├── benchmarks/
│   └── metrics.py
├── experiments/           # Experiment outputs
│   └── runs/              # Model checkpoints, logs
└── PLAN.md               # Detailed methodology
```

### 1.2 Deployment & Model Comparison

**Purpose**: Deploy trained models and compare against baselines in production-like settings.

**Tracking**: Weights & Biases
- Model registry for versioning
- Inference metrics (latency, cost, accuracy)
- A/B comparison between models

**Integration**: Trained models deployed via:
- Ollama (local inference)
- vLLM (production serving)
- LangChain/LangGraph integration

---

## Part 2: User Interface & Data Management

### 2.1 Dataset Initialization

**Purpose**: Load a collection of videos or pre-existing MMIFs for analysis.

**Initialization Methods**:
1. **Video directory**: Point to folder of video files
   - Creates initial MMIF documents with VideoDocument references
   - No annotations yet

2. **MMIF directory**: Load existing MMIFs
   - May already contain views from previous pipeline runs
   - Preserves all existing annotations

**Data Structure**:
```
workspace/
├── videos/                  # Source video files
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── mmifs/                   # MMIF documents
│   ├── video_001.mmif.json
│   ├── video_002.mmif.json
│   └── ...
└── workspace.json           # Workspace metadata
```

### 2.2 Data Browsing Interface

**Purpose**: View and navigate MMIFs, their views, and annotations.

**Features**:
| Feature | Description |
|---------|-------------|
| MMIF Browser | List all MMIFs in workspace, filter/search |
| View Navigator | See all views in an MMIF, their source apps |
| Annotation Viewer | Display annotations (TimeFrames, TextDocuments, etc.) |
| Video Player | Synchronized playback with annotation timeline |
| Thumbnail Strip | Visual overview of video with annotation markers |

**View Display**:
```
MMIF: video_001.mmif.json
├── Documents
│   └── VideoDocument: file:///data/videos/video_001.mp4
└── Views
    ├── [v1] swt-detection (2024-01-15)
    │   └── 15 TimeFrame annotations (slate, chyron, credits)
    ├── [v2] doctr-wrapper (2024-01-15)
    │   └── 23 TextDocument annotations
    └── [v3] human-annotation (2024-01-16)  ← Human edits
        └── 3 TextDocument annotations (corrections)
```

### 2.3 Pipeline Execution via Chat

**Purpose**: Use natural language to run CLAMS pipelines on workspace data.

**Examples**:
```
User: "Run text detection on all videos in the workspace"
Agent: [Executes swt-detection on each video, updates MMIFs]

User: "Extract the slate metadata from video_001"
Agent: [Runs swt-detection → smolvlm2-captioner, adds views to MMIF]

User: "Show me what chyrons were detected in the news videos"
Agent: [Queries MMIFs, displays TimeFrame annotations with label=chyron]
```

### 2.4 Metadata Editing

**Purpose**: Allow users to view, correct, and augment MMIF annotations.

**Key Principle**: Human edits are treated as a CLAMS tool output.
- Creates new view with `app: "human-annotation"` or similar
- Preserves provenance - original annotations unchanged
- Edit view references the annotations being corrected

**Edit Types**:
| Edit | MMIF Representation |
|------|---------------------|
| Correct text | New TextDocument with corrected text, references original |
| Adjust timeframe | New TimeFrame with corrected start/end |
| Add missing annotation | New annotation in human view |
| Reject annotation | Annotation with `rejected: true` property |
| Add metadata | New properties on referenced annotation |

**Example MMIF after human edit**:
```json
{
  "views": [
    {
      "id": "v1",
      "app": "http://apps.clams.ai/swt-detection/v8.4",
      "annotations": [
        {
          "@type": "TimeFrame",
          "id": "tf1",
          "properties": {
            "start": 1000,
            "end": 5000,
            "label": "slate",
            "frameType": "slate"
          }
        }
      ]
    },
    {
      "id": "v2",
      "app": "http://apps.clams.ai/human-annotation/v1",
      "metadata": {
        "timestamp": "2024-01-16T10:30:00Z",
        "annotator": "user@example.com",
        "tool": "clams-agent-ui"
      },
      "annotations": [
        {
          "@type": "Annotation",
          "id": "ha1",
          "properties": {
            "corrects": "v1:tf1",
            "field": "end",
            "original_value": 5000,
            "corrected_value": 4500,
            "reason": "Slate ends before fade"
          }
        }
      ]
    }
  ]
}
```

---

## Part 3: Human-in-the-Loop (HITL) Tool

### 3.1 Concept

The agent has access to a `request_human_review` tool that:
1. Pauses agent execution
2. Presents data to user for review
3. Collects user input/corrections
4. Returns result to agent
5. Records review as new MMIF view

### 3.2 HITL as CLAMS Tool

**Treating HITL as a CLAMS app ensures**:
- Consistent provenance tracking
- Annotations follow MMIF vocabulary
- Can be chained with other tools
- History preserved in MMIF views

**App Metadata** (conceptual):
```python
AppMetadata(
    name="Human Annotation",
    description="Human-in-the-loop review and correction of annotations",
    identifier="human-annotation",
    input=[AnnotationTypes.TimeFrame, AnnotationTypes.TextDocument, ...],
    output=[AnnotationTypes.Annotation],  # Corrections, confirmations
    parameters=[
        {"name": "review_type", "type": "string",
         "choices": ["verify", "correct", "augment", "reject"]},
        {"name": "annotator_id", "type": "string"}
    ]
)
```

### 3.3 Review Interface

**Components**:
1. **Video/Frame Viewer**: Show relevant video segment or extracted frames
2. **Current Annotation Display**: What the pipeline detected
3. **Edit Controls**: Modify timeframes, text, labels
4. **Confirmation Actions**: Approve, Correct, Reject, Skip

**Workflow**:
```
Agent: "I detected these chyrons in the video. Please review:"

┌─────────────────────────────────────────────────────┐
│  Video Player                    [00:01:23.456]     │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │         [Video frame with chyron]           │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Detected Text: "JOHN SMITH - Reporter"            │
│  Timeframe: 00:01:20.000 - 00:01:25.500           │
│  Confidence: 0.87                                  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ Corrected Text: [________________________]   │  │
│  │ Start Time: [00:01:20.000] End: [00:01:25]  │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  [✓ Approve]  [✎ Correct]  [✗ Reject]  [→ Skip]   │
└─────────────────────────────────────────────────────┘
```

### 3.4 Agent Integration

**Tool Definition**:
```python
@tool("request_human_review")
def request_human_review(
    mmif_path: str,
    view_id: str,
    annotation_ids: List[str],
    review_type: str = "verify",
    instructions: str = ""
) -> str:
    """
    Request human review of specific annotations.

    Args:
        mmif_path: Path to the MMIF file
        view_id: View containing annotations to review
        annotation_ids: Specific annotations to review
        review_type: "verify", "correct", "augment", or "batch"
        instructions: Context for the reviewer

    Returns:
        JSON with review results and new view ID
    """
    # 1. Extract annotations and video context
    # 2. Send to UI for human review
    # 3. Wait for response
    # 4. Create new MMIF view with human annotations
    # 5. Return summary to agent
```

**Agent Usage**:
```
User: "Check if the slate text extraction is correct for video_001"

Agent thinking:
1. Load MMIF for video_001
2. Find slate-related views (swt-detection, smolvlm2-captioner)
3. Request human review of extracted text
4. If corrections made, note them for future reference

Agent: [calls request_human_review with slate annotations]

[User reviews and corrects "EPISODE 5" → "EPISODE 15"]

Agent: "The user corrected the episode number from 5 to 15.
        I've updated the MMIF with a human-annotation view
        recording this correction."
```

---

## Part 4: Technical Implementation

### 4.1 Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      React Frontend                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐   │
│  │   Chat   │ │  MMIF    │ │  Video   │ │   Review     │   │
│  │ Interface│ │ Browser  │ │  Player  │ │  Interface   │   │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘   │
└───────┼────────────┼────────────┼──────────────┼────────────┘
        │            │            │              │
        ▼            ▼            ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Flask Backend                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐    │
│  │  AG-UI       │ │  Workspace   │ │  HITL Review     │    │
│  │  Integration │ │  Manager     │ │  Manager         │    │
│  └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘    │
└─────────┼────────────────┼──────────────────┼───────────────┘
          │                │                  │
          ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Tools:                                               │   │
│  │  • CLAMS Execution (swt, doctr, smolvlm2, ...)      │   │
│  │  • FFmpeg (trim, extract, info)                      │   │
│  │  • Workspace (list, query, browse MMIFs)             │   │
│  │  • Human Review (request_human_review)               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                  Execution Backends                          │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌─────────────────┐   │
│  │  CLI   │  │ Docker │  │  HTTP  │  │ Human Annotation │   │
│  │Executor│  │Executor│  │Executor│  │    Service       │   │
│  └────────┘  └────────┘  └────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 New Components Needed

| Component | Location | Purpose |
|-----------|----------|---------|
| WorkspaceManager | `utils/workspace.py` | Manage video/MMIF collections |
| MMIFBrowser | `utils/mmif_browser.py` | Query and navigate MMIF content |
| HumanAnnotationApp | `utils/human_annotation.py` | CLAMS-style app for human edits |
| ReviewInterface | `visualization/src/components/Review.tsx` | React UI for HITL |
| MMIFViewer | `visualization/src/components/MMIFViewer.tsx` | Display MMIF structure |

### 4.3 API Endpoints (New)

```
# Workspace Management
POST   /api/workspace/init          # Initialize workspace from directory
GET    /api/workspace/videos        # List videos in workspace
GET    /api/workspace/mmifs         # List MMIFs in workspace

# MMIF Operations
GET    /api/mmif/<id>               # Get full MMIF document
GET    /api/mmif/<id>/views         # List views in MMIF
GET    /api/mmif/<id>/annotations   # Query annotations
POST   /api/mmif/<id>/views         # Add new view (human annotation)

# Human Review
POST   /api/review/create           # Create review request
GET    /api/review/<id>             # Get review details
POST   /api/review/<id>/submit      # Submit review results
GET    /api/review/pending          # List pending reviews

# Video Serving
GET    /api/video/<id>/stream       # Stream video
GET    /api/video/<id>/frame/<ts>   # Extract frame at timestamp
GET    /api/video/<id>/thumbnail    # Get thumbnail sprite
```

---

## Part 5: Implementation Phases

### Phase 1: Workspace & MMIF Foundation
- [ ] Implement WorkspaceManager
- [ ] Create MMIF query/browse utilities
- [ ] Add workspace API endpoints
- [ ] Basic MMIF viewer in React

### Phase 2: Video Integration
- [ ] Video serving endpoints
- [ ] React video player with timeline
- [ ] Synchronized annotation display
- [ ] Frame extraction for review

### Phase 3: Human-in-the-Loop Core
- [ ] HumanAnnotationApp (CLAMS-style)
- [ ] Review request/response flow
- [ ] Agent tool integration
- [ ] Review interface React component

### Phase 4: Full HITL Workflow
- [ ] Batch review capabilities
- [ ] Review history tracking
- [ ] Annotation diff/comparison
- [ ] Export corrected datasets

### Phase 5: Modelling Integration
- [ ] W&B experiment tracking setup
- [ ] Model deployment pipeline
- [ ] Model comparison framework
- [ ] Trained model as agent backend

---

## Appendix: MMIF Human Annotation Schema

```json
{
  "@context": "http://mmif.clams.ai/vocabulary/",
  "metadata": {
    "mmif": "http://mmif.clams.ai/1.0"
  },
  "views": [
    {
      "id": "human-v1",
      "app": "http://apps.clams.ai/human-annotation/v1",
      "metadata": {
        "app": {
          "name": "Human Annotation",
          "version": "1.0",
          "description": "Manual review and correction of annotations"
        },
        "annotator": "user@example.com",
        "session_id": "review-session-12345",
        "timestamp": "2024-01-16T10:30:00Z",
        "review_type": "correction",
        "source_view": "v1"
      },
      "annotations": [
        {
          "@type": "http://vocab.clams.ai/HumanCorrection",
          "id": "hc1",
          "properties": {
            "target": "v1:tf1",
            "field": "properties.text",
            "original": "EPISODE 5",
            "corrected": "EPISODE 15",
            "confidence": 1.0,
            "note": "OCR misread '1' as empty"
          }
        },
        {
          "@type": "http://vocab.clams.ai/HumanVerification",
          "id": "hv1",
          "properties": {
            "target": "v1:tf2",
            "status": "approved",
            "confidence": 1.0
          }
        },
        {
          "@type": "http://vocab.clams.ai/HumanRejection",
          "id": "hr1",
          "properties": {
            "target": "v1:tf3",
            "reason": "False positive - not a chyron",
            "confidence": 1.0
          }
        }
      ]
    }
  ]
}
```
