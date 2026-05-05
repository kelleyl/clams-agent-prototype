"""Video ingestion for the Haystack catalog prototype."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from .repository import CatalogRepository
from .schema import CatalogDocument, CatalogField, DEFAULT_CATALOG_FIELDS, VideoCatalogRecord


class VideoIngestor:
    """Register raw videos and convert available analysis layers to documents."""

    def __init__(
        self,
        repository: CatalogRepository | None = None,
        catalog_fields: list[CatalogField] | None = None,
    ):
        self.repository = repository or CatalogRepository()
        self.catalog_fields = catalog_fields or DEFAULT_CATALOG_FIELDS

    def register_video(
        self,
        video_path: str | Path,
        supplied_metadata: dict[str, Any] | None = None,
        index_path: str | Path | None = None,
        video_id: str | None = None,
    ) -> VideoCatalogRecord:
        """Register a video and index whatever evidence is currently available.

        For a brand-new video with no metadata or derived layers, this still
        creates catalog prompts and technical metadata documents. If a CLAMS
        layered index is later available, pass `index_path` to ingest its ASR,
        OCR, visual captions, scene summaries, and metadata layers.
        """
        path = Path(video_path).expanduser()
        video_id = video_id or make_video_id(path)
        supplied_metadata = supplied_metadata or {}
        technical_metadata = get_video_metadata(path)

        missing_fields = [
            field.name
            for field in self.catalog_fields
            if field.required and not supplied_metadata.get(field.name)
        ]

        record = VideoCatalogRecord(
            video_id=video_id,
            video_path=str(path),
            metadata={
                key: {"value": value, "source": "user"}
                for key, value in supplied_metadata.items()
                if value not in (None, "")
            },
            technical_metadata=technical_metadata,
            missing_fields=missing_fields,
            status="indexed" if index_path else "needs_analysis",
        )
        self.repository.upsert_record(record)

        docs = build_base_documents(record, self.catalog_fields)
        if index_path:
            docs.extend(load_index_documents(video_id, Path(index_path)))
        self.repository.upsert_documents(docs)
        return record

    def catalog_questions(self, video_id: str) -> list[str]:
        record = self.repository.get_record(video_id)
        if record is None:
            raise KeyError(f"Unknown video_id: {video_id}")
        field_map = {field.name: field for field in self.catalog_fields}
        questions = []
        for name in record.missing_fields:
            field = field_map.get(name)
            if field:
                questions.append(field.prompt)
        return questions


def build_base_documents(
    record: VideoCatalogRecord,
    catalog_fields: list[CatalogField],
) -> list[CatalogDocument]:
    docs: list[CatalogDocument] = []

    metadata_lines = []
    for key, value in record.metadata.items():
        if isinstance(value, dict):
            metadata_lines.append(f"{key}: {value.get('value', '')}")
        else:
            metadata_lines.append(f"{key}: {value}")
    if record.missing_fields:
        metadata_lines.append(f"Missing required fields: {', '.join(record.missing_fields)}")

    docs.append(CatalogDocument(
        id=f"{record.video_id}:catalog:profile",
        content=(
            f"Video catalog profile for {record.video_id}.\n"
            f"Path: {record.video_path}\n"
            + ("\n".join(metadata_lines) if metadata_lines else "No supplied catalog metadata yet.")
        ),
        meta={
            "video_id": record.video_id,
            "layer": "catalog_profile",
            "modality": "metadata",
            "status": record.status,
        },
    ))

    if record.technical_metadata:
        docs.append(CatalogDocument(
            id=f"{record.video_id}:technical:metadata",
            content="Technical video metadata:\n" + json.dumps(record.technical_metadata, indent=2),
            meta={
                "video_id": record.video_id,
                "layer": "technical_metadata",
                "modality": "metadata",
            },
        ))

    question_lines = []
    for field in catalog_fields:
        if field.name in record.missing_fields:
            question_lines.append(f"- {field.name}: {field.prompt}")
    docs.append(CatalogDocument(
        id=f"{record.video_id}:catalog:questions",
        content=(
            "Cataloging questions still needed for this video:\n"
            + ("\n".join(question_lines) if question_lines else "No required catalog questions remain.")
        ),
        meta={
            "video_id": record.video_id,
            "layer": "catalog_questions",
            "modality": "metadata",
        },
    ))

    if record.status == "needs_analysis":
        docs.append(CatalogDocument(
            id=f"{record.video_id}:analysis:placeholder",
            content=(
                "No derived ASR, OCR, visual caption, or scene summary layers have been "
                "created for this raw video yet. Conversation about content should be "
                "limited to supplied catalog metadata and technical metadata until the "
                "analysis pipeline runs."
            ),
            meta={
                "video_id": record.video_id,
                "layer": "analysis_status",
                "modality": "system",
            },
        ))

    return docs


def load_index_documents(video_id: str, index_path: Path) -> list[CatalogDocument]:
    """Convert an existing layered CLAMS index JSON into CatalogDocuments."""
    with index_path.open() as f:
        data = json.load(f)

    docs: list[CatalogDocument] = []
    layers = data.get("layers", {})
    for layer_name, layer in layers.items():
        for i, item in enumerate(layer.get("items", [])):
            text = extract_item_text(item)
            if not text:
                continue
            start_ms = item.get("start_ms")
            end_ms = item.get("end_ms")
            item_id = item.get("id") or f"{i:06d}"
            docs.append(CatalogDocument(
                id=f"{video_id}:{layer_name}:{item_id}",
                content=text,
                meta={
                    "video_id": video_id,
                    "layer": layer_name,
                    "modality": classify_modality(layer_name),
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "source_item_id": item_id,
                },
            ))

    if data.get("duration_ms"):
        docs.append(CatalogDocument(
            id=f"{video_id}:index:summary",
            content=(
                f"Layered CLAMS index for {video_id}. "
                f"Duration: {data.get('duration_ms')} ms. "
                f"Layers: {', '.join(sorted(layers.keys()))}."
            ),
            meta={
                "video_id": video_id,
                "layer": "index_summary",
                "modality": "metadata",
            },
        ))

    return docs


def extract_item_text(item: dict[str, Any]) -> str:
    for key in ("text", "caption", "summary", "description"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if "context_reconciled" in item or "visual_only" in item:
        return json.dumps(
            {
                "visual_only": item.get("visual_only"),
                "context_reconciled": item.get("context_reconciled"),
                "claims": item.get("claims"),
            },
            sort_keys=True,
        )
    return ""


def classify_modality(layer_name: str) -> str:
    name = layer_name.lower()
    if name in ("asr", "asr_whisper") or "transcript" in name:
        return "speech"
    if "text_focus" in name or "ocr" in name or "credits" in name:
        return "ocr"
    if "scene_summary" in name:
        return "visual_scene"
    if "caption" in name or "visual" in name or "shot" in name:
        return "visual"
    if "speaker" in name:
        return "speaker"
    if "entity" in name or "relation" in name:
        return "metadata"
    return "metadata"


def make_video_id(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_") or "video"
    digest_src = f"{path.resolve()}:{path.stat().st_size if path.exists() else 0}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:80]}-{digest}"


def get_video_metadata(path: Path) -> dict[str, Any]:
    try:
        from utils.ffmpeg_tools import FFmpegTools

        output_dir = Path(tempfile.gettempdir()) / "clams_haystack_ffmpeg"
        return FFmpegTools(output_dir=output_dir).get_video_metadata(str(path))
    except Exception as exc:
        return {
            "filename": path.name,
            "path": str(path),
            "exists": path.exists(),
            "metadata_error": str(exc),
        }
