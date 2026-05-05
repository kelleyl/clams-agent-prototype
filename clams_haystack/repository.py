"""Lightweight persistence for the Haystack video catalog prototype.

This is intentionally simple: a JSONL-backed registry that can be replaced by a
Haystack-compatible persistent document store later. Keeping a plain JSONL copy
makes the prototype inspectable and avoids coupling catalog state to one vector
database during migration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .schema import CatalogDocument, VideoCatalogRecord, utc_now


class CatalogRepository:
    """JSONL-backed repository for video records and documents."""

    def __init__(self, root: str | Path = "data/haystack_catalog"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "videos.jsonl"
        self.documents_path = self.root / "documents.jsonl"

    def upsert_record(self, record: VideoCatalogRecord) -> None:
        records = {r.video_id: r for r in self.list_records()}
        record.updated_at = utc_now()
        records[record.video_id] = record
        self._write_jsonl(self.records_path, [r.to_dict() for r in records.values()])

    def get_record(self, video_id: str) -> VideoCatalogRecord | None:
        for record in self.list_records():
            if record.video_id == video_id:
                return record
        return None

    def list_records(self) -> list[VideoCatalogRecord]:
        return [
            VideoCatalogRecord.from_dict(item)
            for item in self._read_jsonl(self.records_path)
        ]

    def upsert_documents(self, documents: Iterable[CatalogDocument]) -> None:
        existing = {doc.id: doc for doc in self.list_documents()}
        for doc in documents:
            existing[doc.id] = doc
        self._write_jsonl(self.documents_path, [d.to_dict() for d in existing.values()])

    def list_documents(self, video_id: str | None = None) -> list[CatalogDocument]:
        docs = [
            CatalogDocument.from_dict(item)
            for item in self._read_jsonl(self.documents_path)
        ]
        if video_id is not None:
            docs = [d for d in docs if d.meta.get("video_id") == video_id]
        return docs

    def update_metadata(
        self,
        video_id: str,
        field: str,
        value: str,
        source: str = "user",
    ) -> VideoCatalogRecord:
        record = self.get_record(video_id)
        if record is None:
            raise KeyError(f"Unknown video_id: {video_id}")

        record.metadata[field] = {"value": value, "source": source}
        record.missing_fields = [f for f in record.missing_fields if f != field]
        self.upsert_record(record)

        content = f"Catalog metadata: {field} = {value}"
        self.upsert_documents([
            CatalogDocument(
                id=f"{video_id}:catalog:{_slug(field)}",
                content=content,
                meta={
                    "video_id": video_id,
                    "layer": "catalog_metadata",
                    "modality": "metadata",
                    "field": field,
                    "source": source,
                },
            )
        ])
        return record

    def record_tool_request(
        self,
        video_id: str | None,
        tool_id: str,
        request: dict,
        status: str = "requested",
        source: str = "tool_palette",
    ) -> CatalogDocument:
        """Persist a tool request even when the video has no catalog record yet."""
        now = utc_now()
        doc = CatalogDocument(
            id=(
                f"{video_id or 'global'}:tool_request:{_slug(tool_id)}:"
                f"{_slug(now)}"
            ),
            content=f"Tool request: {tool_id}\n{json.dumps(request, sort_keys=True)}",
            meta={
                "video_id": video_id,
                "layer": "tool_request",
                "modality": "service",
                "tool_id": tool_id,
                "status": status,
                "source": source,
                "created_at": now,
            },
        )
        self.upsert_documents([doc])
        return doc

    def search_documents(
        self,
        query: str,
        video_id: str | None = None,
        layer: str | None = None,
        top_k: int = 5,
    ) -> list[CatalogDocument]:
        """Tiny lexical fallback search used by tools and tests.

        Haystack BM25 retrieval is used in the RAG pipeline when Haystack is
        installed. This method keeps the prototype useful without requiring the
        optional dependencies during local code review.
        """
        docs = self.list_documents(video_id=video_id)
        if layer:
            docs = [d for d in docs if d.meta.get("layer") == layer]

        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            return docs[:top_k]

        scored = []
        for doc in docs:
            hay = f"{doc.content} {json.dumps(doc.meta, sort_keys=True)}".lower()
            score = sum(hay.count(t) for t in terms)
            if score:
                scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower() or "field"
