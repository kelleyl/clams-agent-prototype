"""Haystack and Chainlit app for CLAMS video cataloging and QA."""

from .agent import HaystackVideoAgent
from .ingest import VideoIngestor
from .repository import CatalogRepository
from .schema import DEFAULT_CATALOG_FIELDS, CatalogField, VideoCatalogRecord

__all__ = [
    "CatalogField",
    "CatalogRepository",
    "DEFAULT_CATALOG_FIELDS",
    "HaystackVideoAgent",
    "VideoCatalogRecord",
    "VideoIngestor",
]
