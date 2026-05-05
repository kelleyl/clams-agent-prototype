"""Shared schema for the Haystack video catalog prototype."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CatalogField:
    """A catalog field we should ask for when a raw video is added."""

    name: str
    prompt: str
    required: bool = True
    description: str = ""


DEFAULT_CATALOG_FIELDS: list[CatalogField] = [
    CatalogField(
        "program_title",
        "What is the program or item title?",
        description="Human-readable title for catalog display and retrieval.",
    ),
    CatalogField(
        "description",
        "Give a short description of what this video is about.",
        description="A concise abstract supplied by a cataloger or generated after analysis.",
    ),
    CatalogField(
        "record_date",
        "When was this video recorded or broadcast, if known?",
        required=False,
        description="Date or approximate date.",
    ),
    CatalogField(
        "source_collection",
        "What collection, donor, station, or archive did this video come from?",
        required=False,
        description="Provenance for cataloging and rights review.",
    ),
    CatalogField(
        "participants",
        "Who appears or speaks in the video, if known?",
        required=False,
        description="Hosts, guests, interviewees, performers, or named speakers.",
    ),
    CatalogField(
        "locations",
        "What locations are associated with the video, if known?",
        required=False,
        description="Places visible, discussed, or otherwise relevant.",
    ),
    CatalogField(
        "rights_status",
        "What rights or access restrictions apply, if known?",
        required=False,
        description="Catalog-level access and usage note.",
    ),
    CatalogField(
        "subject_keywords",
        "What subject keywords should be attached?",
        required=False,
        description="Search-oriented tags, topics, and themes.",
    ),
]


@dataclass
class VideoCatalogRecord:
    """Catalog-level state for one source video."""

    video_id: str
    video_path: str
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    status: str = "registered"
    metadata: dict[str, Any] = field(default_factory=dict)
    technical_metadata: dict[str, Any] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoCatalogRecord":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class CatalogDocument:
    """A Haystack-ready document plus stable metadata."""

    id: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatalogDocument":
        return cls(
            id=data["id"],
            content=data.get("content", ""),
            meta=data.get("meta", {}),
        )
