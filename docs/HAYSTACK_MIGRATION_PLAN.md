# Haystack Migration Plan

Started: 2026-05-04

This branch introduces a Haystack-based direction for the next CLAMS
agent architecture. The goal is not to immediately replace every existing
LangGraph/warm-index evaluation path. The goal is to create a better substrate
for the real deployment shape:

1. A user adds a new video with little or no metadata.
2. The system asks for cataloging fields that cannot yet be inferred.
3. The system creates analysis artifacts as needed: ASR, OCR, visual captions,
   scene summaries, speakers, entities, and metadata extraction.
4. The user can ask questions or converse about the video content.
5. Retrieval and tool use remain scoped, typed, and inspectable.

## Why Haystack Fits Better Than LangChain Here

Haystack's pipeline model maps naturally to this project:

- catalog records and analysis layers become `Document`s with typed metadata
- ASR, OCR, captions, scene summaries, and canonical metadata can have separate
  retrievers
- indexing, cataloging, retrieval, generation, verification, and review can be
  separate reusable components
- `Agent` and `SearchableToolset` support scoped tool discovery instead of
  exposing every CLAMS/index tool at once
- pipelines can branch, loop, run async, and serialize for run provenance

The key shift is from a monolithic prompt that sees all tools to a system where
metadata, documents, tools, and validation are explicit pipeline components.

## New Prototype Package

Rough implementation added under:

```text
clams_haystack/
```

Main pieces:

- `schema.py` defines catalog fields, video records, and catalog documents
- `repository.py` is a simple JSONL-backed catalog/document repository
- `ingest.py` registers raw videos, asks for missing catalog fields, and can
  convert existing layered CLAMS indexes into documents
- `agent.py` provides:
  - lexical fallback search over documents
  - a Haystack RAG pipeline
  - a Haystack tool agent with searchable tools
- `cli.py` exposes smoke commands

The JSONL repository is intentionally simple and inspectable. A proper
Haystack document store can replace it once the document schema stabilizes.

## Dependency Additions

The prototype expects:

```text
haystack-ai
ollama-haystack
chainlit
openai
```

These are added to `requirements.txt`.

## Current Prototype Status

This branch has a runnable raw-video-first skeleton:

- new videos can be registered with no derived metadata
- required catalog fields are tracked as missing questions
- technical metadata and analysis-status placeholders are stored as documents
- existing layered CLAMS indexes can be converted into searchable documents
- lexical search works without Haystack installed
- Haystack RAG and tool-agent paths are wired, but require `haystack-ai`
  and a reachable OpenAI-compatible vLLM server
- Chainlit web chat is wired for `Qwen/Qwen3.5-9B` through a vLLM
  OpenAI-compatible endpoint

This is intentionally not a full replacement for the current LangGraph agent
yet. It is a migration target that makes the catalog state, generated evidence,
retrieval scope, and tool actions explicit.

The current web app defaults to:

```text
model: Qwen/Qwen3.5-9B
vLLM URL: http://localhost:8890/v1
Chainlit port: 8784
```

The web app expects a vLLM server launched with automatic tool calling enabled,
for example `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.

Action-oriented tools such as scene text detection, scene descriptions, ASR,
and metadata writes are exposed as human-confirmed requests. By default the
app records requests without launching expensive video analysis jobs; set
`CLAMS_HAYSTACK_EXECUTE_SERVICES=1` to let the app POST MMIF to configured
CLAMS HTTP services.

## Thesis Dataset and CLAMS Services

The Chainlit app now reads thesis MMIF outputs from multiple roots instead of a
single `data/pipeline_outputs` directory:

```text
data/chicago
data/enriched
data/combined_outputs
data/pipeline_outputs
data/scene_summaries_smoke
data/qwen_outputs
```

Set `CLAMS_HAYSTACK_MMIF_ROOTS` to override this list. The inventory groups
intermediate step files by source video and prefers final `_enriched.mmif`
files when available.

HTTP endpoints for CLAMS tools live in:

```text
data/tool_services.json
```

The service registry currently covers the Aristotle images for SWT detection,
Whisper ASR, SmolVLM2 captioning, Qwen OCR/KIE, PySceneDetect, spaCy, and
TransNet. The Chainlit `/services` command shows which endpoints are up.

Launch selected services on Aristotle with:

```bash
scripts/run_clams_tool_services_aristotle.sh swt_detection
```

The script defaults to GPU 2 so it does not collide with the Qwen vLLM engines
on GPUs 0 and 1.

## Smoke Commands

Register a raw video:

```bash
python -m clams_haystack.cli ingest /path/to/video.mp4
```

Register a raw video with known metadata:

```bash
python -m clams_haystack.cli ingest /path/to/video.mp4 \
  --metadata-json '{"program_title":"Example Program","description":"Interview footage"}'
```

Register a video and ingest an existing layered CLAMS index:

```bash
python -m clams_haystack.cli ingest /path/to/video.mp4 \
  --video-id my_video \
  --index-path data/video_indexes/my_video.json
```

Search the catalog/content documents without needing Haystack installed:

```bash
python -m clams_haystack.cli search "who appears in the video" --video-id my_video
```

Ask through the Haystack RAG pipeline:

```bash
python -m clams_haystack.cli ask "What is this video about?" --video-id my_video
```

Run the Haystack tool agent:

```bash
python -m clams_haystack.cli agent "For my_video, what metadata is missing?"
```

Run the Chainlit app on Aristotle:

```bash
scripts/run_haystack_chainlit_aristotle.sh
```

If the Qwen 3.5 vLLM server is not already running:

```bash
scripts/run_qwen35_vllm_aristotle.sh
```

## Intended Document Metadata

Every document should carry enough metadata for retrieval, filtering, and final
evidence display:

```text
video_id
layer
modality
start_ms
end_ms
source_item_id
confidence
field
source
```

The converter currently fills the common fields available from existing
layered indexes. Future analysis components should preserve richer provenance.

## Near-Term Migration Steps

1. Replace the JSONL-only repository with a persistent Haystack document store
   once we choose the backend.
2. Add separate retrievers for:
   - catalog metadata
   - ASR
   - OCR/text focus
   - visual captions
   - scene summaries
   - entities/relations
3. Add ingestion pipeline components for raw videos:
   - technical metadata
   - user-supplied catalog fields
   - ASR artifact creation
   - OCR/text-scene artifact creation
   - scene-level visual summaries
4. Add a reviewer component before broad or expensive tool calls.
5. Add a metadata extraction/reconciliation pipeline that separates:
   - user supplied metadata
   - OCR/slate/credits metadata
   - ASR-derived metadata
   - uncertain/generated metadata
6. Move v5.2 scene summaries into this document model.

## Open Design Questions

- Which persistent document store should be canonical: Elasticsearch/OpenSearch,
  Qdrant, Pgvector, Chroma, or another backend?
- Should raw video analysis be event-driven in the app or explicitly triggered
  by the agent?
- How should cataloger-supplied metadata be versioned against generated
  metadata?
- Should the conversational agent be allowed to create expensive artifacts
  automatically, or should it ask first?
- How much of the existing warm-index evaluation code should remain separate
  from the production-ish cataloging architecture?

## References

- Haystack Agent docs: https://docs.haystack.deepset.ai/docs/agent
- Haystack Tool docs: https://docs.haystack.deepset.ai/docs/tool
- Haystack SearchableToolset docs: https://docs.haystack.deepset.ai/docs/searchabletoolset
- Haystack Ollama integration docs: https://docs.haystack.deepset.ai/docs/ollamachatgenerator
