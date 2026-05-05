"""Haystack-based agent for cataloged videos and content QA."""

from __future__ import annotations

import os
from typing import Any

from .ingest import VideoIngestor, build_base_documents
from .repository import CatalogRepository
from .schema import CatalogDocument


class HaystackVideoAgent:
    """Prototype agent wrapper around Haystack pipelines and tools."""

    def __init__(
        self,
        repository: CatalogRepository | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | None = None,
        ollama_model: str | None = None,
        ollama_url: str | None = None,
        top_k: int = 8,
    ):
        self.repository = repository or CatalogRepository()
        self.ingestor = VideoIngestor(self.repository)
        self.provider = provider or os.environ.get("CLAMS_HAYSTACK_PROVIDER", "vllm")
        self.model = (
            model
            or ollama_model
            or os.environ.get("CLAMS_HAYSTACK_MODEL")
            or "Qwen/Qwen3.5-9B"
        )
        self.api_base_url = (
            api_base_url
            or os.environ.get("CLAMS_HAYSTACK_VLLM_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "http://localhost:8890/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.ollama_url = (
            ollama_url
            or os.environ.get("CLAMS_HAYSTACK_OLLAMA_URL")
            or "http://localhost:11434"
        )
        self.top_k = top_k

    def register_video(self, *args, **kwargs):
        return self.ingestor.register_video(*args, **kwargs)

    def missing_catalog_questions(self, video_id: str) -> list[str]:
        return self.ingestor.catalog_questions(video_id)

    def search_content(
        self,
        query: str,
        video_id: str | None = None,
        layer: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        docs = self.repository.search_documents(
            query=query,
            video_id=video_id,
            layer=layer,
            top_k=top_k or self.top_k,
        )
        return {
            "count": len(docs),
            "documents": [doc.to_dict() for doc in docs],
            "formatted": format_documents(docs),
        }

    def get_video_overview(self, video_id: str) -> dict[str, Any]:
        record = self.repository.get_record(video_id)
        if record is None:
            return {"error": f"Unknown video_id: {video_id}"}

        docs = self.repository.list_documents(video_id)
        layers = sorted({doc.meta.get("layer", "unknown") for doc in docs})
        return {
            "video": record.to_dict(),
            "document_count": len(docs),
            "layers": layers,
            "missing_catalog_questions": self.missing_catalog_questions(video_id),
        }

    def update_catalog_metadata(
        self,
        video_id: str,
        field: str,
        value: str,
        source: str = "user",
    ) -> dict[str, Any]:
        record = self.repository.update_metadata(video_id, field, value, source)
        self.repository.upsert_documents(
            build_base_documents(record, self.ingestor.catalog_fields)
        )
        return {
            "updated": True,
            "video_id": video_id,
            "field": field,
            "remaining_missing_fields": record.missing_fields,
        }

    def propose_catalog_metadata_update(
        self,
        video_id: str,
        field: str,
        value: str,
        source: str = "agent_proposal",
    ) -> dict[str, Any]:
        return {
            "requires_confirmation": True,
            "video_id": video_id,
            "field": field,
            "value": value,
            "source": source,
            "formatted": (
                f"Proposed catalog metadata update for {video_id}: "
                f"{field} = {value}. Human confirmation is required before writing."
            ),
        }

    def request_scene_text_detection(
        self,
        video_id: str,
        include_credits: bool = True,
        include_slates: bool = True,
    ) -> dict[str, Any]:
        return {
            "requires_confirmation": True,
            "video_id": video_id,
            "artifact": "scene_text_detection",
            "pipeline": [
                "detect shots/scenes",
                "find frames with visible text",
                "run Qwen 3.5 vision/OCR text-focus prompts",
                "classify credits, slates, title cards, lower thirds, and signs",
            ],
            "include_credits": include_credits,
            "include_slates": include_slates,
            "formatted": (
                "Scene text detection can find visible text, OCR, credits, slates, "
                "title cards, lower-thirds, signs, and other text-bearing scenes. "
                "The app records the request but does not launch the expensive "
                "analysis job until a human confirms it."
            ),
        }

    def request_scene_descriptions(
        self,
        video_id: str,
        granularity: str = "scene",
    ) -> dict[str, Any]:
        return {
            "requires_confirmation": True,
            "video_id": video_id,
            "artifact": "scene_descriptions",
            "granularity": granularity,
            "pipeline": [
                "segment video into scenes or shots",
                "sample representative frames",
                "generate visual descriptions with broader video context",
                "store descriptions as timestamped retrievable documents",
            ],
            "formatted": (
                "Scene-level visual descriptions can be generated and indexed for "
                "content QA, catalog summaries, and crossmodal grounding. Human "
                "confirmation is required before launching the job."
            ),
        }

    def request_asr_analysis(self, video_id: str) -> dict[str, Any]:
        return {
            "requires_confirmation": True,
            "video_id": video_id,
            "artifact": "asr",
            "pipeline": [
                "extract audio",
                "run ASR",
                "align transcript spans to timestamps",
                "index transcript chunks for retrieval",
            ],
            "formatted": (
                "ASR can be generated for speech search and content QA. Human "
                "confirmation is required before launching the analysis job."
            ),
        }

    def ask_with_rag(self, question: str, video_id: str | None = None) -> str:
        """Answer with a straightforward Haystack RAG pipeline.

        This path is intentionally simpler than the tool agent. It is useful for
        early smoke tests and for content conversations where direct retrieval is
        enough.
        """
        _require_haystack()

        from haystack import Pipeline
        from haystack.components.builders import ChatPromptBuilder
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.dataclasses import ChatMessage
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        store = InMemoryDocumentStore()
        store.write_documents(to_haystack_documents(self.repository.list_documents(video_id)))

        prompt_template = [
            ChatMessage.from_user(
                """
You answer questions about cataloged video content using only the context below.
If the answer is not supported by the context, say what analysis or metadata is missing.
When possible, cite video_id, layer, and timestamps.

Context:
{% for doc in documents %}
[video={{ doc.meta['video_id'] }} layer={{ doc.meta['layer'] }} start={{ doc.meta.get('start_ms', '') }} end={{ doc.meta.get('end_ms', '') }}]
{{ doc.content }}
{% endfor %}

Question: {{ query }}
"""
            )
        ]

        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=self.top_k))
        pipeline.add_component("prompt_builder", ChatPromptBuilder(template=prompt_template, required_variables="*"))
        pipeline.add_component("llm", self.create_chat_generator())
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "llm.messages")

        result = pipeline.run({
            "retriever": {"query": question},
            "prompt_builder": {"query": question},
        })
        replies = result["llm"]["replies"]
        return replies[0].text if replies else ""

    def create_tool_agent(
        self,
        searchable: bool = True,
        allow_write_tools: bool = False,
    ):
        """Create a Haystack Agent with catalog/content tools."""
        _require_haystack()

        from haystack.components.agents import Agent
        from haystack.tools import SearchableToolset, Tool

        tools = [
            Tool(
                name="search_video_content",
                description=(
                    "Search catalog metadata, transcripts, OCR, visual captions, scene summaries, "
                    "and analysis status documents for videos."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "video_id": {"type": "string"},
                        "layer": {"type": "string"},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["query"],
                },
                function=self.search_content,
                outputs_to_string={"source": "formatted"},
            ),
            Tool(
                name="get_video_overview",
                description="Get catalog state, available layers, and missing metadata questions for a video.",
                parameters={
                    "type": "object",
                    "properties": {"video_id": {"type": "string"}},
                    "required": ["video_id"],
                },
                function=self.get_video_overview,
            ),
            Tool(
                name="propose_catalog_metadata_update",
                description=(
                    "Propose a catalog metadata update from user-provided or "
                    "retrieved evidence. This does not write the value until a "
                    "human approves it."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "field": {"type": "string"},
                        "value": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["video_id", "field", "value"],
                },
                function=self.propose_catalog_metadata_update,
                outputs_to_string={"source": "formatted"},
            ),
            Tool(
                name="request_scene_text_detection",
                description=(
                    "Request scene text detection and OCR for visible text in video "
                    "frames, including credits, end credits, opening slates, title "
                    "cards, lower thirds, chyrons, signs, labels, and other "
                    "text-bearing scenes."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "include_credits": {"type": "boolean"},
                        "include_slates": {"type": "boolean"},
                    },
                    "required": ["video_id"],
                },
                function=self.request_scene_text_detection,
                outputs_to_string={"source": "formatted"},
            ),
            Tool(
                name="request_scene_descriptions",
                description=(
                    "Request scene-level visual descriptions for a raw or indexed "
                    "video. Useful for content conversation, catalog summaries, "
                    "visual QA, crossmodal questions, and grounding when captions "
                    "need broader context."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "video_id": {"type": "string"},
                        "granularity": {"type": "string"},
                    },
                    "required": ["video_id"],
                },
                function=self.request_scene_descriptions,
                outputs_to_string={"source": "formatted"},
            ),
            Tool(
                name="request_asr_analysis",
                description=(
                    "Request ASR transcription for speech search, speaker/content "
                    "questions, catalog summaries, and QA over spoken content."
                ),
                parameters={
                    "type": "object",
                    "properties": {"video_id": {"type": "string"}},
                    "required": ["video_id"],
                },
                function=self.request_asr_analysis,
                outputs_to_string={"source": "formatted"},
            ),
        ]

        if allow_write_tools:
            tools.append(
                Tool(
                    name="update_catalog_metadata",
                    description=(
                        "Write a confirmed catalog metadata value. Use only after "
                        "the user explicitly approves the value."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "video_id": {"type": "string"},
                            "field": {"type": "string"},
                            "value": {"type": "string"},
                            "source": {"type": "string"},
                        },
                        "required": ["video_id", "field", "value"],
                    },
                    function=self.update_catalog_metadata,
                )
            )

        toolset = (
            SearchableToolset(catalog=tools, search_threshold=3)
            if searchable
            else tools
        )
        return Agent(
            chat_generator=self.create_chat_generator(),
            tools=toolset,
            system_prompt=(
                "You are a video cataloging and content QA agent. Use tools to stay grounded. "
                "For new videos, prioritize missing catalog metadata and clearly say when ASR, "
                "OCR, visual captions, or scene summaries have not been generated yet. "
                "Do not claim that expensive analysis jobs have run unless a retrieved document "
                "or tool result says so. Treat request_* tools as human-confirmed proposals."
            ),
            exit_conditions=["text"],
            max_agent_steps=8,
            raise_on_tool_invocation_failure=False,
        )

    def run_tool_agent(
        self,
        message: str,
        searchable: bool = True,
        allow_write_tools: bool = False,
    ) -> str:
        _require_haystack()

        from haystack.dataclasses import ChatMessage

        agent = self.create_tool_agent(
            searchable=searchable,
            allow_write_tools=allow_write_tools,
        )
        result = agent.run(messages=[ChatMessage.from_user(message)])
        last = result.get("last_message") or result["messages"][-1]
        return last.text

    def create_chat_generator(self):
        _require_haystack()
        provider = self.provider.lower()
        if provider in {"vllm", "openai", "openai-compatible", "openai_compatible"}:
            from haystack.components.generators.chat import OpenAIChatGenerator
            from haystack.utils import Secret

            return OpenAIChatGenerator(
                api_key=Secret.from_token(self.api_key),
                api_base_url=self.api_base_url,
                model=self.model,
                generation_kwargs={
                    "temperature": 0.1,
                    "extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False}
                    },
                },
                timeout=180,
            )
        if provider == "ollama":
            try:
                from haystack_integrations.components.generators.ollama import OllamaChatGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "Ollama provider requires: pip install ollama-haystack"
                ) from exc

            return OllamaChatGenerator(
                model=self.model,
                url=self.ollama_url,
                generation_kwargs={"temperature": 0.1},
                timeout=180,
            )
        raise ValueError(f"Unknown Haystack chat provider: {self.provider}")


def to_haystack_documents(documents: list[CatalogDocument]):
    from haystack.dataclasses import Document

    return [
        Document(id=doc.id, content=doc.content, meta=doc.meta)
        for doc in documents
    ]


def format_documents(documents: list[CatalogDocument]) -> str:
    if not documents:
        return "No matching documents found."
    lines = []
    for i, doc in enumerate(documents, 1):
        meta = doc.meta
        timing = ""
        if meta.get("start_ms") is not None:
            timing = f" [{meta.get('start_ms')}ms-{meta.get('end_ms')}ms]"
        lines.append(
            f"{i}. video={meta.get('video_id')} layer={meta.get('layer')}{timing}\n"
            f"{doc.content[:1200]}"
        )
    return "\n\n".join(lines)


def _require_haystack() -> None:
    try:
        import haystack  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Haystack dependencies are not installed. Install with: "
            "pip install haystack-ai"
        ) from exc
