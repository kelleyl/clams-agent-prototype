"""
Hybrid video index: structured JSON files + ChromaDB vector search.

Parses MMIF documents into per-segment entries with OCR text, ASR transcripts,
named entities, and visual captions. Supports semantic search via ChromaDB
with Ollama nomic-embed-text embeddings, plus structured timeline/filter queries.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# App URI substrings used to classify text sources
OCR_APP_KEYWORDS = ("doctr", "tesseract", "parseq", "easyocr")
ASR_APP_KEYWORDS = ("whisper", "distil-whisper", "parakeet")
VLM_APP_KEYWORDS = ("smolvlm2", "llava", "vlm-ocr", "captioner", "qwen")
NER_APP_KEYWORDS = ("spacy",)
KEYWORD_APP_KEYWORDS = ("tfidf-keyword", "keywordextract")
CREDITS_APP_KEYWORDS = ("credits-ocr", "credits_ocr")
DIARIZATION_APP_KEYWORDS = ("speaker-diarization", "diarization")

# Map SWT scene labels to semantic categories for richer summaries.
# SWT labels: bars, slate, credits, chyron, person & chyron, other chyron,
# other text, filmed text, person & extra text, etc.
SCENE_LABEL_SEMANTICS = {
    "bars": {"category": "technical", "desc": "color bar test pattern"},
    "slate": {"category": "metadata", "desc": "production slate with metadata"},
    "credits": {"category": "metadata", "desc": "credits sequence"},
    "person & chyron": {"category": "interview", "desc": "speaker with name/title overlay"},
    "other chyron": {"category": "overlay", "desc": "text overlay (topic, location, or info)"},
    "chyron": {"category": "overlay", "desc": "text overlay"},
    "other text": {"category": "text", "desc": "on-screen text"},
    "filmed text": {"category": "text", "desc": "text filmed in the environment"},
    "person & extra text": {"category": "interview", "desc": "speaker with additional text"},
    "silence": {"category": "audio", "desc": "silence"},
    "speech": {"category": "audio", "desc": "speech segment"},
    "noise": {"category": "audio", "desc": "noise segment"},
    "music": {"category": "audio", "desc": "music segment"},
    "shot": {"category": "visual", "desc": "visual shot"},
}

# Common words that spaCy frequently misclassifies as named entities.
# These produce noise in grounding (e.g., "night" → Wikipedia article on nighttime).
_COMMON_WORD_BLOCKLIST = frozenset({
    # Time/temporal words often tagged as DATE/TIME
    "night", "tonight", "today", "tomorrow", "yesterday", "morning",
    "afternoon", "evening", "dawn", "dusk", "midnight", "noon",
    "days", "day", "weeks", "week", "months", "month", "years", "year",
    "hours", "hour", "minutes", "minute", "seconds", "second",
    "tuesdays", "mondays", "wednesdays", "thursdays", "fridays",
    "saturdays", "sundays",
    # Generic nouns often tagged as ORG/GPE/PERSON
    "greenery", "curls", "mug", "stolen", "king", "queen",
    "good morning", "good evening", "good night",
    # Descriptive phrases
    "the early 20th century", "the mid-20th century",
    "the early to mid-20th century", "the late 20th century",
    "about 10 minutes", "his day", "all these years",
})


def _is_groundable_entity(text: str, entity_type: str) -> bool:
    """Check if an entity is worth grounding (not a common word or noise).

    Filters out:
    - Common words from the blocklist
    - Very short entities (<=2 chars)
    - Entities that are all lowercase with DATE/TIME type (likely temporal phrases)
    - Pure numeric entities
    """
    if len(text) <= 2:
        return False
    if text.lower() in _COMMON_WORD_BLOCKLIST:
        return False
    # Pure numbers or numeric-like strings
    if text.replace(",", "").replace(".", "").replace("-", "").isdigit():
        return False
    # Lowercase DATE/TIME entities are usually common temporal phrases, not proper nouns
    if entity_type in ("DATE", "TIME") and text == text.lower() and not any(c.isdigit() for c in text):
        return False
    return True


import re

# Patterns indicating VLM hallucination (not real scene description)
_HALLUCINATION_PATTERNS = [
    re.compile(r"\\frac\{|\\[\[({]|\\begin\{"),  # LaTeX math
    re.compile(r"To solve the given problem|Let's solve this"),  # Math problem-solving
    re.compile(r"```python|```java|```cpp|def [a-z_]+\(.*\):"),  # Code blocks
    re.compile(r"<html|<div |<script", re.IGNORECASE),  # HTML
]


# ============================================================================
# Time-Overlap Utilities (for multi-layer index)
# ============================================================================

def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two time ranges overlap."""
    return a_start < b_end and a_end > b_start


def overlap_ms(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Compute overlap in milliseconds between two time ranges."""
    o = min(a_end, b_end) - max(a_start, b_start)
    return max(0, o)


def query_time_range(
    layers: Dict[str, dict], start_ms: int, end_ms: int,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, list]:
    """Return all items from selected layers that overlap [start_ms, end_ms).

    Args:
        layers: The layers dict from a video index document.
        start_ms: Start of query range.
        end_ms: End of query range.
        layer_names: If given, restrict to these layers. Otherwise all layers.

    Returns:
        Dict mapping layer name → list of overlapping items.
    """
    result: Dict[str, list] = {}
    names = layer_names or list(layers.keys())
    for name in names:
        layer = layers.get(name)
        if not layer:
            continue
        hits = [
            item for item in layer.get("items", [])
            if overlaps(item["start_ms"], item["end_ms"], start_ms, end_ms)
        ]
        if hits:
            result[name] = hits
    return result


def cross_layer_join(
    layer_a_items: List[dict], layer_b_items: List[dict],
    min_overlap_ms: int = 0,
) -> List[tuple]:
    """Find all (item_a, item_b) pairs with temporal overlap >= min_overlap_ms."""
    pairs = []
    # Sort both by start time for efficiency
    a_sorted = sorted(layer_a_items, key=lambda x: x["start_ms"])
    b_sorted = sorted(layer_b_items, key=lambda x: x["start_ms"])

    for a in a_sorted:
        for b in b_sorted:
            if b["start_ms"] >= a["end_ms"]:
                break  # No more b items can overlap with a
            ov = overlap_ms(a["start_ms"], a["end_ms"], b["start_ms"], b["end_ms"])
            if ov > min_overlap_ms:
                pairs.append((a, b))
    return pairs


def _extract_broadcast_date(video_id: str) -> Optional[str]:
    """Extract broadcast date from video ID/filename.

    Looks for patterns like _9_8_1975 (month_day_year) or _1973 (year only).
    Returns ISO date string (YYYY-MM-DD) or partial (YYYY) or None.
    """
    # Full date: month_day_year at end
    m = re.search(r'(\d{1,2})_(\d{1,2})_(\d{4})$', video_id)
    if m:
        month, day, year = m.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    # Year only at end
    m = re.search(r'(\d{4})$', video_id)
    if m:
        return m.group(1)
    return None


def _is_hallucinated_caption(text: str) -> bool:
    """Detect VLM hallucinations (math solutions, code, etc.) in captions."""
    for pat in _HALLUCINATION_PATTERNS:
        if pat.search(text):
            return True
    return False


def _get_span_text(token) -> str:
    """Get the full noun phrase or compound for a token."""
    # Walk left through compound modifiers and right through conjunctions
    subtree = list(token.subtree)
    # Filter to keep only relevant tokens (no punctuation, determiners)
    words = []
    for t in subtree:
        if t.pos_ in ("PUNCT", "SPACE"):
            continue
        # Keep determiners only if they're part of a proper noun phrase
        if t.dep_ == "det" and token.pos_ != "PROPN":
            continue
        words.append(t.text)
    phrase = " ".join(words)
    # Limit length
    if len(phrase) > 80:
        phrase = token.text
    return phrase


def _extract_svo_triples(sent) -> List[tuple]:
    """Extract (subject, verb, object) triples from a spaCy sentence.

    Uses dependency parsing to find:
    - nsubj → ROOT verb → dobj/attr patterns
    - Passive constructions (nsubjpass)
    - Prepositional complements (prep + pobj)
    """
    triples = []

    for token in sent:
        if token.pos_ != "VERB":
            continue

        subjects = []
        objects = []
        preps = []

        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subjects.append(_get_span_text(child))
            elif child.dep_ in ("dobj", "attr", "oprd"):
                objects.append(_get_span_text(child))
            elif child.dep_ == "prep":
                # Prepositional phrases: "reports from Florida"
                for pobj in child.children:
                    if pobj.dep_ == "pobj":
                        preps.append((child.text, _get_span_text(pobj)))

        verb = token.lemma_

        # Skip trivially common verbs that don't carry meaning
        if verb in ("be", "have", "do", "say", "get", "go", "make",
                     "come", "take", "know", "see", "look", "show"):
            continue

        for subj in subjects:
            # Skip very short/generic subjects
            if len(subj) <= 2 or subj.lower() in ("it", "he", "she", "they",
                                                     "we", "you", "this", "that",
                                                     "there", "i", "one"):
                continue
            for obj in objects:
                if len(obj) <= 2:
                    continue
                triples.append((subj, verb, obj))
            for prep_word, pobj in preps:
                if len(pobj) <= 2:
                    continue
                triples.append((subj, f"{verb} {prep_word}", pobj))

    return triples


# ============================================================================
# Data Model
# ============================================================================

@dataclass
class SegmentEntry:
    """Per-segment structured data."""
    segment_id: str
    start_ms: int
    end_ms: int
    scene_label: str = ""
    segment_type: str = ""  # semantic category: metadata, interview, overlay, text, audio, visual, technical
    classification: Dict[str, float] = field(default_factory=dict)
    ocr_text: str = ""
    asr_transcript: str = ""
    visual_caption: str = ""
    named_entities: List[Dict[str, str]] = field(default_factory=list)
    # Grounded entities: entity text → description from Wikipedia/web
    grounded_entities: Dict[str, str] = field(default_factory=dict)
    # Relations: list of (subject, predicate, object) triples
    relations: List[List[str]] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_summary(self) -> str:
        """Generate a dense fused text summary for embedding.

        Incorporates all available information: scene semantics, text content,
        named entities with grounded descriptions, and keywords.
        """
        parts = []

        # Scene type context
        semantics = SCENE_LABEL_SEMANTICS.get(self.scene_label, {})
        desc = semantics.get("desc", "")
        if desc:
            parts.append(f"[{desc}]")
        elif self.scene_label:
            parts.append(f"[{self.scene_label}]")

        dur_s = (self.end_ms - self.start_ms) / 1000
        parts.append(f"({dur_s:.1f}s)")

        if self.ocr_text:
            parts.append(f"On-screen text: {self.ocr_text.strip()}")
        if self.asr_transcript:
            transcript = self.asr_transcript.strip()
            if len(transcript) > 300:
                transcript = transcript[:300] + "..."
            parts.append(f"Speech: {transcript}")
        if self.visual_caption:
            parts.append(f"Visual: {self.visual_caption.strip()}")

        # Dense entity descriptions — this is the key enrichment
        if self.grounded_entities:
            grounded_strs = []
            for ent_text, ent_desc in self.grounded_entities.items():
                if ent_desc:
                    grounded_strs.append(f"{ent_text}: {ent_desc}")
                else:
                    grounded_strs.append(ent_text)
            parts.append(f"About: {'; '.join(grounded_strs)}")
        elif self.named_entities:
            ent_strs = [f"{e['text']} ({e.get('type', '?')})" for e in self.named_entities]
            parts.append(f"Entities: {', '.join(ent_strs)}")

        if self.relations:
            rel_strs = [f"{s} → {p} → {o}" for s, p, o in self.relations[:10]]
            parts.append(f"Relations: {'; '.join(rel_strs)}")

        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords)}")

        return " ".join(parts) if parts else f"[segment {self.start_ms}-{self.end_ms}ms]"


# ============================================================================
# Ollama Embedding Function for ChromaDB
# ============================================================================

def _check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _make_ollama_embedding_function(model: str, base_url: str):
    """Create a ChromaDB-compatible embedding function using Ollama.

    Returns None if Ollama is unavailable.
    Uses chromadb.utils.embedding_functions if available, otherwise builds a
    simple wrapper that satisfies ChromaDB's EmbeddingFunction protocol.
    """
    if not _check_ollama_available(base_url):
        return None

    class _OllamaEF:
        """Minimal ChromaDB EmbeddingFunction wrapping Ollama."""

        @staticmethod
        def name() -> str:
            return "ollama_nomic_embed"

        @staticmethod
        def build_from_config(config):
            return _OllamaEF()

        def get_config(self):
            return {"model": model, "base_url": base_url}

        @staticmethod
        def validate_config(config):
            pass

        def validate_config_update(self, old_config, new_config):
            pass

        def supported_spaces(self):
            return ["cosine", "l2", "ip"]

        def default_space(self):
            return "cosine"

        def __call__(self, input: List[str]) -> List[List[float]]:
            embeddings = []
            for text in input:
                resp = requests.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                embeddings.append(resp.json().get("embedding", []))
            return embeddings

        def embed_query(self, input: List[str]) -> List[List[float]]:
            return self(input)

    return _OllamaEF()


# ============================================================================
# VideoIndex
# ============================================================================

class VideoIndex:
    """Hybrid JSON + ChromaDB index for video segments."""

    def __init__(
        self,
        index_dir: str = "data/video_indexes",
        chroma_dir: str = "data/chroma_db",
        embedding_model: str = "nomic-embed-text",
        fallback_segment_duration_ms: int = 10000,
    ):
        self.index_dir = Path(index_dir)
        self.chroma_dir = Path(chroma_dir)
        self.embedding_model = embedding_model
        self.fallback_segment_duration_ms = fallback_segment_duration_ms

        self.index_dir.mkdir(parents=True, exist_ok=True)

        self._chroma_client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy ChromaDB init
    # ------------------------------------------------------------------

    def _get_collection(self):
        """Lazily initialize ChromaDB client and collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb

            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

            # Try Ollama embeddings, fall back to ChromaDB default
            ef = _make_ollama_embedding_function(self.embedding_model, "http://localhost:11434")
            if ef is not None:
                logger.info(f"Using Ollama embeddings ({self.embedding_model})")
                self._collection = self._chroma_client.get_or_create_collection(
                    name="video_segments",
                    embedding_function=ef,
                )
            else:
                logger.info("Using ChromaDB default embeddings")
                self._collection = self._chroma_client.get_or_create_collection(
                    name="video_segments",
                )

            return self._collection

        except ImportError:
            logger.warning("chromadb not installed; vector search disabled")
            return None
        except Exception as e:
            logger.warning(f"ChromaDB init failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Build index from MMIF
    # ------------------------------------------------------------------

    def build_from_mmif(
        self, video_id: str, video_path: str, mmif_dict: dict, source_mmif_path: str = ""
    ) -> Dict[str, Any]:
        """Parse MMIF → segments → save JSON + upsert ChromaDB."""
        segments = self._extract_segments(mmif_dict, video_id)

        # Compute duration from video metadata in MMIF or from segments
        duration_ms = 0
        if segments:
            duration_ms = max(s.end_ms for s in segments)

        # Build index document
        broadcast_date = _extract_broadcast_date(video_id)
        index_doc = {
            "video_id": video_id,
            "video_path": video_path,
            "broadcast_date": broadcast_date,
            "duration_ms": duration_ms,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "source_mmif": source_mmif_path,
            "segments": [s.to_dict() for s in segments],
        }

        # Save JSON
        json_path = self.index_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(index_doc, f, indent=2)
        logger.info(f"Saved index for {video_id}: {len(segments)} segments → {json_path}")

        # Upsert to ChromaDB
        self._upsert_segments(video_id, segments)

        return index_doc

    # ------------------------------------------------------------------
    # Multi-layer index builder (v2 schema)
    # ------------------------------------------------------------------

    def build_layers_from_mmif(
        self, video_id: str, video_path: str, mmif_dict: dict,
        source_mmif_path: str = "",
        diarization_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse MMIF → independent timeline layers → save JSON.

        Each CLAMS tool's output becomes its own layer with items that have
        independent time boundaries. Cross-modal relations are generated
        after extraction.

        Args:
            video_id: Unique identifier for the video.
            video_path: Path to the video file.
            mmif_dict: Parsed MMIF document.
            source_mmif_path: Path to the source MMIF file.
            diarization_path: Optional path to diarization JSON
                (data/diarization/<video_id>.json).

        Returns:
            The index document dict (also saved to disk).
        """
        views = mmif_dict.get("views", [])

        # Build global annotation index (same as _extract_segments)
        ann_index: Dict[str, dict] = {}
        for view in views:
            view_id = view.get("id", "")
            for ann in view.get("annotations", []):
                aid = ann.get("properties", {}).get("id", "")
                if aid:
                    ann_index[aid] = ann
                    if view_id and ":" not in aid:
                        ann_index[f"{view_id}:{aid}"] = ann

        duration_ms = self._estimate_duration(mmif_dict, ann_index)

        # Build frame-rate map for cross-view time conversion
        ann_frame_rates: Dict[str, float] = {}
        for view in views:
            vfr = self._detect_view_frame_rate(view, ann_index, duration_ms)
            if vfr:
                view_id = view.get("id", "")
                for ann in view.get("annotations", []):
                    aid = ann.get("properties", {}).get("id", "")
                    if aid:
                        ann_frame_rates[aid] = vfr
                        if view_id and ":" not in aid:
                            ann_frame_rates[f"{view_id}:{aid}"] = vfr

        # Extract each layer from MMIF views
        layers: Dict[str, dict] = {}
        now = datetime.now(timezone.utc).isoformat()

        for view in views:
            app_uri = view.get("metadata", {}).get("app", "").lower()
            view_fr = self._detect_view_frame_rate(view, ann_index, duration_ms)

            if "swt-detection" in app_uri:
                layer = self._extract_scene_layer(view, ann_index, video_id, duration_ms, view_fr)
                if layer["items"]:
                    layers["scenes"] = {**layer, "indexed_at": now}

            elif "transnet" in app_uri or "pyscenedetect" in app_uri:
                layer = self._extract_shot_layer(view, ann_index, video_id, duration_ms, view_fr)
                if layer["items"]:
                    layers["shots"] = {**layer, "indexed_at": now}

            elif any(kw in app_uri for kw in ASR_APP_KEYWORDS):
                layer = self._extract_asr_layer(view, ann_index, video_id, view_fr)
                if layer["items"]:
                    layers["asr"] = {**layer, "indexed_at": now}

            elif any(kw in app_uri for kw in CREDITS_APP_KEYWORDS):
                layer = self._extract_credits_layer(view, ann_index, video_id, duration_ms, view_fr)
                if layer["items"]:
                    layers["credits"] = {**layer, "indexed_at": now}

            elif any(kw in app_uri for kw in DIARIZATION_APP_KEYWORDS):
                spk_layer, ch_layer = self._extract_speaker_chapter_from_mmif(
                    view, ann_index, video_id, now
                )
                if spk_layer["items"]:
                    layers["speakers"] = spk_layer
                if ch_layer["items"]:
                    layers["chapters"] = ch_layer

            elif any(kw in app_uri for kw in VLM_APP_KEYWORDS):
                # Distinguish OCR (swt_transcription) vs visual captions (transnet_shots)
                vlm_type = self._refine_vlm_source_type(view)
                if vlm_type == "ocr":
                    layer = self._extract_ocr_layer(view, ann_index, video_id, view_fr, ann_frame_rates)
                    if layer["items"]:
                        # Merge if we already have an ocr layer
                        if "ocr" in layers:
                            layers["ocr"]["items"].extend(layer["items"])
                        else:
                            layers["ocr"] = {**layer, "indexed_at": now}
                else:
                    layer = self._extract_caption_layer(view, ann_index, video_id, view_fr, ann_frame_rates)
                    if layer["items"]:
                        if "visual_captions" in layers:
                            layers["visual_captions"]["items"].extend(layer["items"])
                        else:
                            layers["visual_captions"] = {**layer, "indexed_at": now}

        # Extract speaker and chapter layers from diarization JSON (fallback if not in MMIF)
        if "speakers" not in layers:
            if diarization_path is None:
                diar_dir = Path(self.index_dir).parent / "diarization"
                diar_file = diar_dir / f"{video_id}.json"
                if diar_file.exists():
                    diarization_path = str(diar_file)

            if diarization_path and Path(diarization_path).exists():
                speaker_layer, chapter_layer = self._extract_speaker_chapter_layers(
                    diarization_path, video_id, now
                )
                if speaker_layer["items"]:
                    layers["speakers"] = speaker_layer
                if chapter_layer["items"]:
                    layers["chapters"] = chapter_layer

        # Initialize empty layers for post-processing
        layers.setdefault("entities", {"source": "spacy+grounding", "indexed_at": now, "items": []})
        layers.setdefault("relations", {"source": "spacy-dep-parse+cross-modal", "indexed_at": now, "items": []})

        # Generate cross-modal relations (speaker↔chyron identity)
        self._generate_cross_modal_relations(layers, video_id)

        # Build index document
        broadcast_date = _extract_broadcast_date(video_id)
        index_doc = {
            "schema_version": 2,
            "video_id": video_id,
            "video_path": video_path,
            "broadcast_date": broadcast_date,
            "duration_ms": duration_ms,
            "indexed_at": now,
            "source_mmif": source_mmif_path,
            "layers": layers,
        }

        # Save JSON
        json_path = self.index_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(index_doc, f, indent=2)

        layer_summary = {k: len(v.get("items", [])) for k, v in layers.items()}
        logger.info(f"Saved layered index for {video_id}: {layer_summary} → {json_path}")

        return index_doc

    # --- Per-layer extraction methods ---

    def _extract_shot_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, duration_ms: int, view_fr: Optional[float],
    ) -> dict:
        """Extract TransNet/PySceneDetect shots as a layer."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []
        idx = 0

        for ann in view.get("annotations", []):
            if "TimeFrame" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            start = props.get("start")
            end = props.get("end")

            if start is None or end is None:
                targets = props.get("targets", [])
                if targets:
                    start, end = self._resolve_timeframe_range(targets, ann_index)

            if view_fr and start is not None and end is not None:
                start = int(start / view_fr * 1000)
                end = int(end / view_fr * 1000)

            if (start is not None and end is not None
                    and isinstance(start, (int, float))
                    and isinstance(end, (int, float))
                    and end > start):
                items.append({
                    "id": f"shot_{idx}",
                    "start_ms": int(start),
                    "end_ms": int(end),
                })
                idx += 1

        return {"source": app_uri.split("/")[-2] if "/" in app_uri else app_uri, "items": items}

    def _extract_scene_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, duration_ms: int, view_fr: Optional[float],
    ) -> dict:
        """Extract SWT scene labels as a layer."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []
        idx = 0

        for ann in view.get("annotations", []):
            if "TimeFrame" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            label = props.get("label", props.get("frameType", ""))
            classification = props.get("classification", {})

            start = props.get("start")
            end = props.get("end")
            if start is None or end is None:
                targets = props.get("targets", [])
                if targets:
                    start, end = self._resolve_timeframe_range(targets, ann_index)

            if view_fr and start is not None and end is not None:
                start = int(start / view_fr * 1000)
                end = int(end / view_fr * 1000)

            if (start is not None and end is not None
                    and isinstance(start, (int, float))
                    and isinstance(end, (int, float))
                    and end > start):
                item = {
                    "id": f"scene_{idx}",
                    "start_ms": int(start),
                    "end_ms": int(end),
                    "label": str(label),
                }
                if isinstance(classification, dict) and classification:
                    item["classification"] = classification
                items.append(item)
                idx += 1

        return {"source": "swt-detection", "items": items}

    def _extract_asr_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, view_fr: Optional[float],
    ) -> dict:
        """Extract ASR (Whisper/Parakeet) as word-level or sentence-level items."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []

        # Build per-view annotation map
        view_anns = {}
        for ann in view.get("annotations", []):
            aid = ann.get("properties", {}).get("id", "")
            if aid:
                view_anns[aid] = ann

        if self._has_alignment_chain(view):
            # Whisper-style: Token + Alignment + TimeFrame chain
            # Group tokens into contiguous speaker turns / sentences
            alignment_map: Dict[str, str] = {}
            for ann in view.get("annotations", []):
                if "Alignment" in ann.get("@type", ""):
                    props = ann.get("properties", {})
                    source = props.get("source", "")
                    target = props.get("target", "")
                    if source and target:
                        alignment_map[target] = source

            token_times: List[tuple] = []
            for ann in view.get("annotations", []):
                if "Token" not in ann.get("@type", ""):
                    continue
                props = ann.get("properties", {})
                token_text = props.get("word", props.get("text", ""))
                token_id = props.get("id", "")
                if not token_text or not token_id:
                    continue

                tf_id = alignment_map.get(token_id, "")
                start_ms = None
                end_ms = None
                if tf_id:
                    tf_ann = ann_index.get(tf_id) or view_anns.get(tf_id)
                    if tf_ann:
                        tf_props = tf_ann.get("properties", {})
                        s = tf_props.get("start")
                        e = tf_props.get("end")
                        if isinstance(s, (int, float)):
                            start_ms = int(s)
                        if isinstance(e, (int, float)):
                            end_ms = int(e)

                if start_ms is None:
                    for key in ("start", "timePoint"):
                        if key in props and isinstance(props[key], (int, float)):
                            start_ms = int(props[key])
                            break

                if start_ms is not None:
                    token_times.append((token_text, start_ms, end_ms or start_ms))

            # Group tokens into ~sentence-sized ASR items (by gaps > 500ms)
            if token_times:
                token_times.sort(key=lambda t: t[1])
                current_tokens = [token_times[0][0]]
                current_start = token_times[0][1]
                current_end = token_times[0][2]
                idx = 0

                for token_text, t_start, t_end in token_times[1:]:
                    gap = t_start - current_end
                    if gap > 500 and current_tokens:
                        items.append({
                            "id": f"asr_{idx}",
                            "start_ms": current_start,
                            "end_ms": current_end,
                            "text": " ".join(current_tokens),
                        })
                        idx += 1
                        current_tokens = [token_text]
                        current_start = t_start
                        current_end = t_end
                    else:
                        current_tokens.append(token_text)
                        current_end = max(current_end, t_end)

                if current_tokens:
                    items.append({
                        "id": f"asr_{idx}",
                        "start_ms": current_start,
                        "end_ms": current_end,
                        "text": " ".join(current_tokens),
                    })
        else:
            # Parakeet-style: TextDocument with time info, or TimeFrame annotations
            idx = 0
            for ann in view.get("annotations", []):
                atype = ann.get("@type", "")
                props = ann.get("properties", {})

                if "TextDocument" in atype:
                    text_content = self._extract_text(props)
                    if not text_content:
                        continue
                    time_ms = self._find_time_for_annotation(props, ann_index)
                    if time_ms is not None:
                        items.append({
                            "id": f"asr_{idx}",
                            "start_ms": time_ms,
                            "end_ms": time_ms,  # Point annotation
                            "text": text_content,
                        })
                        idx += 1

        source_name = app_uri.split("/")[-2] if "/" in app_uri else app_uri
        return {"source": source_name, "items": items}

    def _extract_ocr_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, view_fr: Optional[float],
        ann_frame_rates: Dict[str, float],
    ) -> dict:
        """Extract OCR/transcription text from VLM captioner (swt_transcription config)."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []
        idx = 0

        for ann in view.get("annotations", []):
            if "TextDocument" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            text_content = self._extract_text(props)
            if not text_content:
                continue

            # Resolve origin to get time range and scene label
            origin_ref = props.get("origin", "")
            start_ms = 0
            end_ms = 0
            scene_label = ""
            origin_layer = ""

            if origin_ref:
                resolved = ann_index.get(origin_ref)
                if resolved:
                    rprops = resolved.get("properties", {})
                    scene_label = rprops.get("label", "")
                    # Check if origin annotation's view uses frames
                    origin_fr = ann_frame_rates.get(origin_ref)
                    # Get the full time range of the origin
                    o_start = rprops.get("start")
                    o_end = rprops.get("end")
                    if o_start is not None and o_end is not None:
                        if origin_fr:
                            start_ms = int(o_start / origin_fr * 1000)
                            end_ms = int(o_end / origin_fr * 1000)
                        else:
                            start_ms = int(o_start)
                            end_ms = int(o_end)
                    else:
                        targets = rprops.get("targets", [])
                        if targets:
                            s, e = self._resolve_timeframe_range(targets, ann_index)
                            if s is not None and e is not None:
                                # TimePoints from SWT may also be in frames
                                if origin_fr:
                                    s = int(s / origin_fr * 1000)
                                    e = int(e / origin_fr * 1000)
                                start_ms = s
                                end_ms = e
                    origin_layer = "scenes"

            if start_ms == 0 and end_ms == 0:
                # Fallback: try direct time on the annotation
                time_ms = self._find_time_for_annotation(props, ann_index)
                fr = view_fr or ann_frame_rates.get(origin_ref)
                if time_ms is not None:
                    if fr:
                        time_ms = int(time_ms / fr * 1000)
                    start_ms = time_ms
                    end_ms = time_ms

            if end_ms <= start_ms:
                end_ms = start_ms + 1000  # Default 1s duration

            item = {
                "id": f"ocr_{idx}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text_content,
            }
            if scene_label:
                item["scene_label"] = scene_label
            if origin_layer:
                item["origin_layer"] = origin_layer
            items.append(item)
            idx += 1

        source_name = app_uri.split("/")[-2] if "/" in app_uri else app_uri
        return {"source": f"{source_name}/swt_transcription", "items": items}

    def _extract_caption_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, view_fr: Optional[float],
        ann_frame_rates: Dict[str, float],
    ) -> dict:
        """Extract visual captions from VLM captioner (transnet_shots config)."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []
        idx = 0

        for ann in view.get("annotations", []):
            if "TextDocument" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            text_content = self._extract_text(props)
            if not text_content:
                continue

            # Filter hallucinated captions
            if _is_hallucinated_caption(text_content):
                logger.debug(f"Filtered hallucinated caption: {text_content[:60]}...")
                continue

            # Resolve origin to get time range, handling frame-based views
            origin_ref = props.get("origin", "")
            start_ms = 0
            end_ms = 0

            if origin_ref:
                resolved = ann_index.get(origin_ref)
                if resolved:
                    rprops = resolved.get("properties", {})
                    o_start = rprops.get("start")
                    o_end = rprops.get("end")
                    # Check if the origin annotation's view uses frames
                    origin_fr = ann_frame_rates.get(origin_ref)
                    if o_start is not None and o_end is not None:
                        if origin_fr:
                            start_ms = int(o_start / origin_fr * 1000)
                            end_ms = int(o_end / origin_fr * 1000)
                        else:
                            start_ms = int(o_start)
                            end_ms = int(o_end)

            if end_ms <= start_ms:
                # Fallback: try direct time on the annotation
                time_ms = self._find_time_for_annotation(props, ann_index)
                if time_ms is not None:
                    fr = view_fr or ann_frame_rates.get(origin_ref)
                    if fr:
                        time_ms = int(time_ms / fr * 1000)
                    start_ms = time_ms
                    end_ms = time_ms + 1000

            if end_ms <= start_ms:
                end_ms = start_ms + 1000

            item = {
                "id": f"vcap_{idx}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text_content,
                "origin_layer": "shots",
            }
            items.append(item)
            idx += 1

        source_name = app_uri.split("/")[-2] if "/" in app_uri else app_uri
        return {"source": f"{source_name}/transnet_shots", "items": items}

    def _extract_credits_layer(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, duration_ms: int, view_fr: Optional[float],
    ) -> dict:
        """Extract structured credits from credits-ocr app."""
        app_uri = view.get("metadata", {}).get("app", "")
        items = []
        idx = 0

        for ann in view.get("annotations", []):
            if "TextDocument" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            mime = props.get("mime", "")
            text_content = self._extract_text(props)
            if not text_content:
                continue

            # Only process the final JSON output (not per-frame OCR)
            if mime != "application/json" and not text_content.strip().startswith("["):
                continue

            time_ms = self._find_time_for_annotation(props, ann_index)
            if time_ms is not None and view_fr:
                time_ms = int(time_ms / view_fr * 1000)

            # Get time range from origin
            origin_ref = props.get("origin", "")
            start_ms = time_ms or (duration_ms - 60000 if duration_ms > 60000 else 0)
            end_ms = duration_ms or start_ms + 60000

            if origin_ref:
                resolved = ann_index.get(origin_ref)
                if resolved:
                    rprops = resolved.get("properties", {})
                    o_start = rprops.get("start")
                    o_end = rprops.get("end")
                    if o_start is not None and o_end is not None:
                        if view_fr:
                            o_start = int(o_start / view_fr * 1000)
                            o_end = int(o_end / view_fr * 1000)
                        start_ms = int(o_start)
                        end_ms = int(o_end)
                    else:
                        targets = rprops.get("targets", [])
                        if targets:
                            s, e = self._resolve_timeframe_range(targets, ann_index)
                            if s is not None and e is not None:
                                start_ms = s
                                end_ms = e

            try:
                credits = json.loads(text_content)
                if not isinstance(credits, list):
                    credits = [credits]

                entries = []
                for entry in credits:
                    if not isinstance(entry, dict):
                        continue
                    name = entry.get("name", "").strip()
                    role = entry.get("role", "").strip()
                    if not name or role.lower() == "copyright":
                        continue

                    entity_type = "PERSON"
                    org_keywords = ("studio", "picture", "network",
                                    "production", "entertainment",
                                    "broadcast", "foundation", "inc",
                                    "corp", "llc", "company", "associate")
                    if any(kw in name.lower() for kw in org_keywords):
                        entity_type = "ORG"

                    credit_entry = {"name": name, "role": role, "entity_type": entity_type}
                    if entry.get("character"):
                        credit_entry["character"] = entry["character"]
                    entries.append(credit_entry)

                if entries:
                    items.append({
                        "id": f"cred_{idx}",
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "entries": entries,
                    })
                    idx += 1

            except (json.JSONDecodeError, TypeError):
                pass

        return {"source": "credits-ocr", "items": items}

    def _extract_speaker_chapter_from_mmif(
        self, view: dict, ann_index: Dict[str, dict],
        video_id: str, indexed_at: str,
    ) -> tuple:
        """Extract speaker turns and chapters from diarization MMIF view.

        The speaker-diarization CLAMS app produces:
        - TimeFrame annotations with frameType="speaker" and label=speaker_name
        - Chapter annotations with title and time range
        - Span + Alignment linking speaker turns to ASR transcript

        Returns (speaker_layer, chapter_layer) dicts.
        """
        speaker_items = []
        chapter_items = []
        speaker_names = {}  # speaker_label → name
        spk_idx = 0
        ch_idx = 0

        # Also build alignment map to get transcript text per speaker turn
        alignment_map: Dict[str, str] = {}  # source (tf_id) → target (span_id)
        span_map: Dict[str, dict] = {}  # span_id → span annotation

        for ann in view.get("annotations", []):
            atype = ann.get("@type", "")
            props = ann.get("properties", {})

            if "Alignment" in atype:
                source = props.get("source", "")
                target = props.get("target", "")
                if source and target:
                    alignment_map[source] = target

            elif "Span" in atype:
                aid = props.get("id", "")
                if aid:
                    span_map[aid] = props

        # Extract ASR text document for span lookups
        asr_text = ""
        for ann in view.get("annotations", []):
            props = ann.get("properties", {})
            if "Span" in ann.get("@type", ""):
                doc_ref = props.get("document", "")
                if doc_ref:
                    doc_ann = ann_index.get(doc_ref)
                    if doc_ann:
                        asr_text = self._extract_text(doc_ann.get("properties", {}))
                        if asr_text:
                            break

        for ann in view.get("annotations", []):
            atype = ann.get("@type", "")
            props = ann.get("properties", {})

            if "TimeFrame" in atype and props.get("frameType") == "speaker":
                start = props.get("start")
                end = props.get("end")
                label = props.get("label", "")

                if start is not None and end is not None and end > start:
                    item = {
                        "id": f"spk_{spk_idx}",
                        "start_ms": int(start),
                        "end_ms": int(end),
                        "speaker_id": label,
                    }

                    # Resolve speaker name (label may already be a name like "Dr. Krimm")
                    if label and not label.startswith("SPEAKER_"):
                        item["speaker_name"] = label

                    # Try to get transcript text via alignment → span
                    ann_id = props.get("id", "")
                    span_id = alignment_map.get(ann_id, "")
                    if span_id and span_id in span_map:
                        span_props = span_map[span_id]
                        s_start = span_props.get("start")
                        s_end = span_props.get("end")
                        if asr_text and s_start is not None and s_end is not None:
                            item["text"] = asr_text[int(s_start):int(s_end)]

                    speaker_items.append(item)
                    spk_idx += 1

            elif "Chapter" in atype:
                start = props.get("start")
                end = props.get("end")
                title = props.get("title", "")

                if start is not None and end is not None:
                    chapter_items.append({
                        "id": f"ch_{ch_idx}",
                        "start_ms": int(start),
                        "end_ms": int(end),
                        "title": title,
                    })
                    ch_idx += 1

        # Collect unique speaker names
        for item in speaker_items:
            if item.get("speaker_name"):
                speaker_names[item["speaker_id"]] = item["speaker_name"]

        speaker_layer = {
            "source": "pyannote+llm",
            "indexed_at": indexed_at,
            "items": speaker_items,
        }
        if speaker_names:
            speaker_layer["speaker_names"] = speaker_names

        chapter_layer = {
            "source": "diarization/llm",
            "indexed_at": indexed_at,
            "items": chapter_items,
        }

        return speaker_layer, chapter_layer

    def _extract_speaker_chapter_layers(
        self, diarization_path: str, video_id: str, indexed_at: str,
    ) -> tuple:
        """Extract speaker turns and chapters from diarization JSON.

        Diarization JSON format:
        {
            "segments": [{"start": 0.0, "end": 0.875, "speaker": "speaker_1"}, ...],
            "speaker_names": {"speaker_1": "Jim Lehrer", ...},
            "chapters": [{"title": "...", "start": 0, "end": 434}, ...]
        }

        Returns (speaker_layer, chapter_layer) dicts.
        """
        with open(diarization_path) as f:
            diar = json.load(f)

        diar_segs = diar.get("segments", [])
        speaker_names = diar.get("speaker_names", {})
        speaker_details = diar.get("speaker_details", {})
        chapters = diar.get("chapters", [])

        # Build speaker turns — merge consecutive segments with same speaker
        speaker_items = []
        if diar_segs:
            current = diar_segs[0]
            idx = 0
            for seg in diar_segs[1:]:
                if (seg["speaker"] == current["speaker"]
                        and seg["start"] - current["end"] < 0.5):
                    # Merge consecutive same-speaker segments
                    current["end"] = seg["end"]
                else:
                    # Emit current turn
                    spk_id = current["speaker"]
                    item = {
                        "id": f"spk_{idx}",
                        "start_ms": int(current["start"] * 1000),
                        "end_ms": int(current["end"] * 1000),
                        "speaker_id": spk_id,
                    }
                    if spk_id in speaker_names:
                        item["speaker_name"] = speaker_names[spk_id]
                    if spk_id in speaker_details:
                        details = speaker_details[spk_id]
                        if details.get("title"):
                            item["speaker_title"] = details["title"]
                    speaker_items.append(item)
                    idx += 1
                    current = dict(seg)

            # Emit last turn
            spk_id = current["speaker"]
            item = {
                "id": f"spk_{idx}",
                "start_ms": int(current["start"] * 1000),
                "end_ms": int(current["end"] * 1000),
                "speaker_id": spk_id,
            }
            if spk_id in speaker_names:
                item["speaker_name"] = speaker_names[spk_id]
            if spk_id in speaker_details:
                details = speaker_details[spk_id]
                if details.get("title"):
                    item["speaker_title"] = details["title"]
            speaker_items.append(item)

        speaker_layer = {
            "source": "pyannote+llm",
            "indexed_at": indexed_at,
            "items": speaker_items,
        }
        if speaker_names:
            speaker_layer["speaker_names"] = speaker_names
        if speaker_details:
            speaker_layer["speaker_details"] = speaker_details

        # Build chapter layer
        chapter_items = []
        if isinstance(chapters, list):
            for i, ch in enumerate(chapters):
                if isinstance(ch, dict):
                    chapter_items.append({
                        "id": f"ch_{i}",
                        "start_ms": int(ch.get("start", 0) * 1000),
                        "end_ms": int(ch.get("end", 0) * 1000),
                        "title": ch.get("title", ""),
                    })

        chapter_layer = {
            "source": "diarization/llm",
            "indexed_at": indexed_at,
            "items": chapter_items,
        }

        return speaker_layer, chapter_layer

    def _generate_cross_modal_relations(
        self, layers: Dict[str, dict], video_id: str,
    ):
        """Generate cross-modal relations from layer data.

        Currently generates:
        - speaker↔chyron identity relations (from diarization speaker_names)
        - credited_as relations (from credits entries)
        """
        relations_layer = layers.get("relations", {"source": "cross-modal", "items": []})
        idx = len(relations_layer.get("items", []))

        # Speaker↔chyron identity relations
        speakers = layers.get("speakers", {})
        speaker_names = speakers.get("speaker_names", {})
        speaker_details = speakers.get("speaker_details", {})
        ocr_items = layers.get("ocr", {}).get("items", [])

        for spk_id, name in speaker_names.items():
            if not name:
                continue
            # Find a speaker turn for this speaker
            spk_turns = [
                s for s in speakers.get("items", [])
                if s.get("speaker_id") == spk_id
            ]
            if not spk_turns:
                continue
            first_turn = spk_turns[0]

            # Find the chyron that matched (if we have OCR items)
            chyron_evidence = None
            for ocr in ocr_items:
                if ocr.get("scene_label") in ("chyron", "person & chyron", "other chyron"):
                    if name.lower() in ocr.get("text", "").lower():
                        chyron_evidence = ocr["id"]
                        break

            rel = {
                "id": f"rel_{idx}",
                "start_ms": first_turn["start_ms"],
                "end_ms": first_turn["end_ms"],
                "subject": spk_id,
                "predicate": "identified_as",
                "object": name,
                "source_layers": ["speakers", "ocr"],
                "relation_type": "cross-modal-identity",
            }
            if chyron_evidence:
                rel["evidence"] = {"chyron": chyron_evidence, "speaker_turn": first_turn["id"]}

            # Add title info from speaker_details
            details = speaker_details.get(spk_id, {})
            if details.get("title"):
                rel["object_title"] = details["title"]

            relations_layer.setdefault("items", []).append(rel)
            idx += 1

        # Credits → credited_as relations
        credits_items = layers.get("credits", {}).get("items", [])
        for cred_item in credits_items:
            for entry in cred_item.get("entries", []):
                rel = {
                    "id": f"rel_{idx}",
                    "start_ms": cred_item["start_ms"],
                    "end_ms": cred_item["end_ms"],
                    "subject": entry["name"],
                    "predicate": "credited_as",
                    "object": entry["role"],
                    "source_layers": ["credits"],
                    "relation_type": "credited_as",
                }
                relations_layer.setdefault("items", []).append(rel)
                idx += 1

        layers["relations"] = relations_layer

    def _extract_segments(self, mmif_dict: dict, video_id: str) -> List[SegmentEntry]:
        """Parse MMIF into SegmentEntry list.

        Handles real MMIF structures:
        - SWT TimeFrames that use `targets` → TimePoint references (no inline start/end)
        - inaSpeechSegmenter TimeFrames with inline start/end
        - Whisper ASR: single TextDocument + Token/Alignment/TimeFrame chain
        - OCR TextDocuments with direct time properties
        """
        views = mmif_dict.get("views", [])
        if not views:
            return []

        # Build global annotation index: ID → annotation dict
        # Some apps use full prefixed IDs ("v_3:tf_1"), others use short
        # IDs ("tf_1"). Index both forms so cross-view lookups work.
        ann_index: Dict[str, dict] = {}
        for view in views:
            view_id = view.get("id", "")
            for ann in view.get("annotations", []):
                aid = ann.get("properties", {}).get("id", "")
                if aid:
                    ann_index[aid] = ann
                    # Also add prefixed form if not already prefixed
                    if view_id and ":" not in aid:
                        ann_index[f"{view_id}:{aid}"] = ann

        # Step 1: Extract segment boundaries from TimeFrame annotations.
        # Strategy: collect SWT segments (semantically labeled text scenes) and
        # PySceneDetect segments (shot boundaries). Use SWT where available, fill
        # timeline gaps with PySceneDetect shots. Fall back to inaSpeechSegmenter
        # if neither is present.
        segments: List[SegmentEntry] = []
        swt_segments: List[SegmentEntry] = []
        shot_segments: List[SegmentEntry] = []
        audio_segments: List[SegmentEntry] = []

        # Estimate duration early (needed for frame-rate detection)
        duration_ms = self._estimate_duration(mmif_dict, ann_index)

        for view in views:
            app_uri = view.get("metadata", {}).get("app", "").lower()

            # Determine which bucket this view's TimeFrames go into
            if "swt-detection" in app_uri:
                bucket = swt_segments
            elif "pyscenedetect" in app_uri or "transnet" in app_uri:
                bucket = shot_segments
            elif any(kw in app_uri for kw in ("inaspeechsegmenter", "spoken-lid")):
                bucket = audio_segments
            else:
                continue

            # Check if this view uses frame-based time units
            view_fr = self._detect_view_frame_rate(view, ann_index, duration_ms)

            for ann in view.get("annotations", []):
                atype = ann.get("@type", "")
                if "TimeFrame" not in atype:
                    continue

                props = ann.get("properties", {})
                label = props.get("label", props.get("frameType", ""))
                classification = props.get("classification", {})

                # Try inline start/end first (inaSpeechSegmenter/pyscenedetect/transnet)
                start = props.get("start")
                end = props.get("end")

                # If no inline start/end, resolve from targets (SWT pattern)
                if start is None or end is None:
                    targets = props.get("targets", [])
                    if targets:
                        start, end = self._resolve_timeframe_range(targets, ann_index)

                # Convert frames to milliseconds if needed
                if view_fr and start is not None and end is not None:
                    start = int(start / view_fr * 1000)
                    end = int(end / view_fr * 1000)

                if (start is not None and end is not None
                        and isinstance(start, (int, float))
                        and isinstance(end, (int, float))
                        and end > start):
                    seg_id = f"{video_id}_{int(start)}_{int(end)}"
                    label_str = str(label)
                    semantics = SCENE_LABEL_SEMANTICS.get(label_str, {})
                    seg_type = semantics.get("category", "unknown")
                    bucket.append(SegmentEntry(
                        segment_id=seg_id,
                        start_ms=int(start),
                        end_ms=int(end),
                        scene_label=label_str,
                        segment_type=seg_type,
                        classification=classification if isinstance(classification, dict) else {},
                    ))

        # Merge: TransNet shots are the primary segmentation (fine-grained),
        # SWT labels overlay onto shots (a shot containing a chyron gets that label).
        # This ensures every shot gets its own caption/ASR slice rather than
        # piling everything into giant SWT "Neg" segments.
        if shot_segments:
            if swt_segments:
                segments = self._overlay_swt_on_shots(swt_segments, shot_segments, video_id)
            else:
                segments = shot_segments
        elif swt_segments:
            segments = swt_segments
        elif audio_segments:
            segments = audio_segments

        # Deduplicate segments with identical time ranges
        if segments:
            seen = {}
            for seg in segments:
                key = (seg.start_ms, seg.end_ms)
                if key not in seen or (seg.scene_label and not seen[key].scene_label):
                    seen[key] = seg
            segments = list(seen.values())

        # Fallback: fixed-duration clips if no scene-level TimeFrames found
        if not segments:
            duration_ms = self._estimate_duration(mmif_dict, ann_index)
            if duration_ms > 0:
                step = self.fallback_segment_duration_ms
                t = 0
                while t < duration_ms:
                    end_t = min(t + step, duration_ms)
                    seg_id = f"{video_id}_{t}_{end_t}"
                    segments.append(SegmentEntry(
                        segment_id=seg_id, start_ms=t, end_ms=end_t
                    ))
                    t = end_t

        if not segments:
            return []

        segments.sort(key=lambda s: s.start_ms)

        # Fill timeline gaps with "Neg" placeholder segments so that captions
        # from fixed-window VLM captioning (e.g., every 5s) have segments to
        # land in even when SWT only covers text-containing scenes.
        duration_ms = self._estimate_duration(mmif_dict, ann_index)
        if segments and duration_ms > 0:
            gap_segs = []
            prev_end = 0
            for seg in segments:
                if seg.start_ms - prev_end > 1000:  # gap > 1s
                    gap_segs.append(SegmentEntry(
                        segment_id=f"{video_id}_{prev_end}_{seg.start_ms}",
                        start_ms=prev_end,
                        end_ms=seg.start_ms,
                        scene_label="Neg",
                        segment_type="gap",
                    ))
                prev_end = max(prev_end, seg.end_ms)
            # Fill trailing gap
            if duration_ms - prev_end > 1000:
                gap_segs.append(SegmentEntry(
                    segment_id=f"{video_id}_{prev_end}_{duration_ms}",
                    start_ms=prev_end,
                    end_ms=duration_ms,
                    scene_label="Neg",
                    segment_type="gap",
                ))
            if gap_segs:
                logger.debug(f"Added {len(gap_segs)} gap-filler segments")
                segments.extend(gap_segs)
                segments.sort(key=lambda s: s.start_ms)

        # Build a map: annotation_id → frame_rate (if that annotation's view uses frames)
        # Must include both short ("tf_1") and prefixed ("v_3:tf_1") forms
        # since cross-view origin references use the prefixed form.
        ann_frame_rates: Dict[str, float] = {}
        for view in views:
            vfr = self._detect_view_frame_rate(view, ann_index, duration_ms)
            if vfr:
                view_id = view.get("id", "")
                for ann in view.get("annotations", []):
                    aid = ann.get("properties", {}).get("id", "")
                    if aid:
                        ann_frame_rates[aid] = vfr
                        if view_id and ":" not in aid:
                            ann_frame_rates[f"{view_id}:{aid}"] = vfr

        # Step 2: Collect text, entities, and keywords — assign to segments
        for view in views:
            app_uri = view.get("metadata", {}).get("app", "").lower()
            source_type = self._classify_app_source(app_uri)

            # Refine VLM source type: if a captioner app was configured with
            # OCR-style prompts (e.g., "transcribe"), treat it as OCR.
            if source_type == "vlm":
                source_type = self._refine_vlm_source_type(view)

            # Build per-view annotation map for local lookups
            view_anns = {}
            for ann in view.get("annotations", []):
                aid = ann.get("properties", {}).get("id", "")
                if aid:
                    view_anns[aid] = ann

            # Handle Whisper-style ASR: single TextDocument + Token/Alignment/TimeFrame
            if source_type == "asr" and self._has_alignment_chain(view):
                self._assign_asr_via_alignments(view, view_anns, ann_index, segments)
                continue

            # Handle spaCy NER: NamedEntity annotations with category + document ref
            if source_type == "ner":
                self._assign_ner_from_view(view, ann_index, segments)
                continue

            # Handle TF-IDF keywords
            if source_type == "keywords":
                self._assign_keywords_from_view(view, ann_index, segments)
                continue

            # Handle structured credits: parse JSON with role/name pairs
            # directly into named_entities, preserving role information
            if source_type == "credits":
                self._assign_credits_from_view(view, ann_index, segments, duration_ms)
                continue

            # Detect time unit for this view: check contains metadata for
            # TimePoint/TimeFrame timeUnit. If not declared, infer from values.
            view_frame_rate = self._detect_view_frame_rate(view, ann_index, duration_ms)

            # Handle TextDocuments (OCR, VLM, text-slicer output)
            for ann in view.get("annotations", []):
                atype = ann.get("@type", "")
                props = ann.get("properties", {})

                if "TextDocument" in atype:
                    text_content = self._extract_text(props)
                    if not text_content:
                        continue

                    time_ms = self._find_time_for_annotation(props, ann_index)
                    # Convert frames to milliseconds if needed.
                    # Check this view's frame rate first, then the referenced
                    # annotation's view (for cross-view origin refs like
                    # Qwen TextDoc → TransNet TimeFrame).
                    fr = view_frame_rate
                    if fr is None and time_ms is not None:
                        origin_ref = props.get("origin", "")
                        if origin_ref and origin_ref in ann_frame_rates:
                            fr = ann_frame_rates[origin_ref]
                    if time_ms is not None and fr:
                        time_ms = int(time_ms / fr * 1000)
                    target_seg = self._find_segment_for_time(segments, time_ms)
                    if target_seg is None:
                        target_seg = segments[0]

                    if source_type == "ocr":
                        target_seg.ocr_text = (target_seg.ocr_text + " " + text_content).strip()
                    elif source_type == "vlm":
                        if not _is_hallucinated_caption(text_content):
                            target_seg.visual_caption = (target_seg.visual_caption + " " + text_content).strip()
                        else:
                            logger.debug(f"Filtered hallucinated caption at {time_ms}ms: {text_content[:60]}...")
                    else:
                        # text-slicer or unknown source — assign as ASR if it
                        # references a Whisper TextDocument, else OCR
                        target_seg.ocr_text = (target_seg.ocr_text + " " + text_content).strip()

                # MMIF-native NamedEntity (non-spaCy sources)
                elif "NamedEntity" in atype and "lappsgrid" not in atype:
                    entity_text = props.get("text", "")
                    entity_type = props.get("category", props.get("entityType", "UNKNOWN"))
                    if not entity_text:
                        continue
                    time_ms = self._find_time_for_annotation(props, ann_index)
                    target_seg = self._find_segment_for_time(segments, time_ms)
                    if target_seg is None:
                        target_seg = segments[0]
                    target_seg.named_entities.append({"text": entity_text, "type": str(entity_type)})

        # Step 3: Generate summaries
        for seg in segments:
            seg.summary = seg.to_summary()

        return segments

    def _assign_ner_from_view(
        self, view: dict, ann_index: Dict[str, dict], segments: List[SegmentEntry]
    ):
        """Assign spaCy NER annotations to segments.

        spaCy produces NamedEntity annotations (vocab.lappsgrid.org/NamedEntity)
        with `category` (PERSON, ORG, GPE, DATE, etc.), `text`, and character
        offsets (`start`/`end`) into a source TextDocument.

        Time assignment: resolve the source TextDocument's time, or if the
        TextDocument was produced by text-slicer (already time-aligned), use
        that alignment.
        """
        for ann in view.get("annotations", []):
            atype = ann.get("@type", "")
            if "NamedEntity" not in atype:
                continue

            props = ann.get("properties", {})
            entity_text = props.get("text", "")
            entity_type = props.get("category", props.get("entityType", "UNKNOWN"))
            if not entity_text:
                continue

            # Try to resolve time from the annotation's document reference
            time_ms = None
            doc_ref = props.get("document", "")
            if doc_ref:
                doc_ann = ann_index.get(doc_ref)
                if doc_ann:
                    doc_props = doc_ann.get("properties", {})
                    time_ms = self._find_time_for_annotation(doc_props, ann_index)

            # Fallback: direct time on the entity annotation
            if time_ms is None:
                time_ms = self._find_time_for_annotation(props, ann_index)

            target_seg = self._find_segment_for_time(segments, time_ms)
            if target_seg is None:
                target_seg = segments[0] if segments else None
            if target_seg:
                # Deduplicate entities within a segment
                existing = {(e["text"], e["type"]) for e in target_seg.named_entities}
                if (entity_text, str(entity_type)) not in existing:
                    target_seg.named_entities.append({
                        "text": entity_text,
                        "type": str(entity_type),
                    })

    def _assign_keywords_from_view(
        self, view: dict, ann_index: Dict[str, dict], segments: List[SegmentEntry]
    ):
        """Assign TF-IDF keyword annotations to segments.

        tfidf-keywordextractor produces TextDocument annotations where the
        `text` property contains extracted keywords and `scores` contains
        TF-IDF values.
        """
        for ann in view.get("annotations", []):
            atype = ann.get("@type", "")
            if "TextDocument" not in atype:
                continue

            props = ann.get("properties", {})
            text_content = self._extract_text(props)
            if not text_content:
                continue

            # Keywords may be comma-separated or space-separated
            keywords = [kw.strip() for kw in text_content.replace(",", " ").split() if kw.strip()]

            time_ms = self._find_time_for_annotation(props, ann_index)
            target_seg = self._find_segment_for_time(segments, time_ms)
            if target_seg is None:
                target_seg = segments[0] if segments else None
            if target_seg:
                for kw in keywords:
                    if kw not in target_seg.keywords:
                        target_seg.keywords.append(kw)

    def _assign_credits_from_view(
        self, view: dict, ann_index: Dict[str, dict],
        segments: List[SegmentEntry], duration_ms: int
    ):
        """Parse structured credits JSON and inject into named_entities.

        The credits-ocr app produces TextDocuments with mime=application/json
        containing arrays of {role, name, character?, copyright?} objects.
        We parse these directly into named_entities, preserving the role as
        metadata, rather than dumping the JSON into ocr_text for SpaCy to
        re-discover.
        """
        view_frame_rate = self._detect_view_frame_rate(view, ann_index, duration_ms)

        for ann in view.get("annotations", []):
            atype = ann.get("@type", "")
            if "TextDocument" not in atype:
                continue

            props = ann.get("properties", {})
            mime = props.get("mime", "")
            text_content = self._extract_text(props)
            if not text_content:
                continue

            # Find the segment this credits annotation belongs to
            time_ms = self._find_time_for_annotation(props, ann_index)
            if time_ms is not None and view_frame_rate:
                time_ms = int(time_ms / view_frame_rate * 1000)
            target_seg = self._find_segment_for_time(segments, time_ms)
            if target_seg is None:
                target_seg = segments[-1] if segments else None
            if target_seg is None:
                continue

            # Try to parse as JSON (the final deduplicated output)
            if mime == "application/json" or text_content.strip().startswith("["):
                try:
                    credits = json.loads(text_content)
                    if not isinstance(credits, list):
                        credits = [credits]

                    existing = {(e["text"], e["type"]) for e in target_seg.named_entities}

                    for entry in credits:
                        if not isinstance(entry, dict):
                            continue
                        name = entry.get("name", "").strip()
                        role = entry.get("role", "").strip()
                        if not name:
                            continue

                        # Copyright entries: store as ORG or skip
                        if role.lower() == "copyright":
                            continue

                        # Determine entity type from role
                        entity_type = "PERSON"
                        org_keywords = ("studio", "picture", "network",
                                        "production", "entertainment",
                                        "broadcast", "foundation", "inc",
                                        "corp", "llc", "company", "associate")
                        if any(kw in name.lower() for kw in org_keywords):
                            entity_type = "ORG"

                        key = (name, entity_type)
                        if key not in existing:
                            target_seg.named_entities.append({
                                "text": name,
                                "type": entity_type,
                                "role": role,
                                "source": "credits",
                            })
                            existing.add(key)

                    logger.debug(f"Parsed {len(credits)} credit entries at {time_ms}ms")
                    continue

                except (json.JSONDecodeError, TypeError):
                    pass

            # Fallback: per-frame OCR text (not JSON) — store as ocr_text
            target_seg.ocr_text = (target_seg.ocr_text + " " + text_content).strip()

    @staticmethod
    def _merge_swt_and_shots(
        swt_segments: List[SegmentEntry],
        shot_segments: List[SegmentEntry],
        video_id: str,
    ) -> List[SegmentEntry]:
        """Merge SWT text-scene segments with PySceneDetect shot boundaries.

        SWT segments are kept as-is (they have semantic labels like Slate, Chyron).
        Shot segments fill gaps in the timeline where SWT has no coverage — these
        are scenes without text that still deserve visual captioning.
        """
        swt_sorted = sorted(swt_segments, key=lambda s: s.start_ms)
        merged = list(swt_sorted)

        # Build list of covered intervals from SWT
        for shot in sorted(shot_segments, key=lambda s: s.start_ms):
            # Check if this shot overlaps with any SWT segment
            overlaps = False
            for swt in swt_sorted:
                # Overlap if shot and SWT intersect
                if shot.start_ms < swt.end_ms and shot.end_ms > swt.start_ms:
                    overlaps = True
                    break
            if not overlaps:
                # This shot fills a gap — add it with "shot" label
                shot.segment_id = f"{video_id}_{shot.start_ms}_{shot.end_ms}"
                merged.append(shot)

        return merged

    @staticmethod
    def _overlay_swt_on_shots(
        swt_segments: List[SegmentEntry],
        shot_segments: List[SegmentEntry],
        video_id: str,
    ) -> List[SegmentEntry]:
        """Use TransNet shots as base segmentation, overlay SWT labels.

        Each TransNet shot becomes a segment. If a shot overlaps with an SWT
        text-scene (Chyron, Slate, Credits, etc.), it inherits that label.
        Shots not overlapping any SWT scene get label "Neg".
        This ensures fine-grained segments (~10s each) rather than giant
        SWT "Neg" blocks that can span minutes.
        """
        swt_sorted = sorted(swt_segments, key=lambda s: s.start_ms)
        # Filter SWT to only text-containing labels (not Neg)
        swt_text = [s for s in swt_sorted if s.scene_label and s.scene_label != "Neg"]

        result = []
        for shot in sorted(shot_segments, key=lambda s: s.start_ms):
            # Find best overlapping SWT label
            best_label = ""
            best_overlap = 0
            best_classification = {}
            for swt in swt_text:
                overlap_start = max(shot.start_ms, swt.start_ms)
                overlap_end = min(shot.end_ms, swt.end_ms)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label = swt.scene_label
                    best_classification = swt.classification

            shot.scene_label = best_label or "Neg"
            shot.classification = best_classification
            shot.segment_id = f"{video_id}_{shot.start_ms}_{shot.end_ms}"
            result.append(shot)

        return result

    @staticmethod
    def _refine_vlm_source_type(view: dict) -> str:
        """Distinguish OCR vs caption when a VLM captioner does both.

        Checks the view's appConfiguration.promptMap for OCR-like keywords
        ("transcribe", "key-value") to classify as OCR. Otherwise keeps "vlm".
        """
        app_config = view.get("metadata", {}).get("appConfiguration", {})
        prompt_map = app_config.get("promptMap", {})

        # promptMap can be a dict or list of "Label:prompt" strings
        prompt_text = ""
        if isinstance(prompt_map, dict):
            prompt_text = " ".join(str(v) for v in prompt_map.values())
        elif isinstance(prompt_map, list):
            prompt_text = " ".join(str(v) for v in prompt_map)

        prompt_text_lower = prompt_text.lower()
        if any(kw in prompt_text_lower for kw in ("transcribe", "key-value", "key: value")):
            return "ocr"
        return "vlm"

    @staticmethod
    def _classify_app_source(app_uri: str) -> str:
        """Classify an app URI into source type."""
        for kw in OCR_APP_KEYWORDS:
            if kw in app_uri:
                return "ocr"
        for kw in ASR_APP_KEYWORDS:
            if kw in app_uri:
                return "asr"
        for kw in VLM_APP_KEYWORDS:
            if kw in app_uri:
                return "vlm"
        for kw in NER_APP_KEYWORDS:
            if kw in app_uri:
                return "ner"
        for kw in KEYWORD_APP_KEYWORDS:
            if kw in app_uri:
                return "keywords"
        for kw in CREDITS_APP_KEYWORDS:
            if kw in app_uri:
                return "credits"
        return "unknown"

    @staticmethod
    def _extract_text(props: dict) -> str:
        """Extract text content from annotation properties."""
        text_value = props.get("text", {})
        if isinstance(text_value, dict):
            return text_value.get("@value", "")
        return str(text_value) if text_value else ""

    @staticmethod
    def _has_alignment_chain(view: dict) -> bool:
        """Check if a view contains Token+Alignment+TimeFrame (Whisper pattern)."""
        contains = view.get("metadata", {}).get("contains", {})
        type_names = {k.split("/")[-2] if "/" in k else k for k in contains.keys()}
        return "Token" in type_names and "Alignment" in type_names

    def _assign_asr_via_alignments(
        self, view: dict, view_anns: Dict[str, dict],
        ann_index: Dict[str, dict], segments: List[SegmentEntry]
    ):
        """Break Whisper ASR into per-segment transcript chunks.

        Whisper produces: TextDocument (full transcript) + Tokens (words with
        char offsets into the TextDocument) + Alignments (Token ↔ TimeFrame)
        + TimeFrames (word-level time ranges).

        We resolve each Token's time via its Alignment → TimeFrame, then assign
        the token text to the appropriate segment.
        """
        # Collect token text + time
        token_times: List[tuple] = []  # (token_text, start_ms)

        # Build alignment map: token_id → timeframe_id
        alignment_map: Dict[str, str] = {}  # target (token) → source (timeframe)
        for ann in view.get("annotations", []):
            if "Alignment" in ann.get("@type", ""):
                props = ann.get("properties", {})
                source = props.get("source", "")
                target = props.get("target", "")
                if source and target:
                    alignment_map[target] = source

        for ann in view.get("annotations", []):
            if "Token" not in ann.get("@type", ""):
                continue
            props = ann.get("properties", {})
            token_text = props.get("word", props.get("text", ""))
            token_id = props.get("id", "")

            if not token_text or not token_id:
                continue

            # Resolve time via alignment chain: Token → Alignment.source → TimeFrame
            tf_id = alignment_map.get(token_id, "")
            if tf_id:
                tf_ann = ann_index.get(tf_id) or view_anns.get(tf_id)
                if tf_ann:
                    tf_props = tf_ann.get("properties", {})
                    start = tf_props.get("start")
                    if isinstance(start, (int, float)):
                        token_times.append((token_text, int(start)))
                        continue

            # Fallback: token may have direct time
            for key in ("start", "timePoint"):
                if key in props and isinstance(props[key], (int, float)):
                    token_times.append((token_text, int(props[key])))
                    break

        # Assign tokens to segments
        for token_text, time_ms in token_times:
            seg = self._find_segment_for_time(segments, time_ms)
            if seg:
                seg.asr_transcript = (seg.asr_transcript + " " + token_text).strip()

    def _resolve_timeframe_range(
        self, targets: List[str], ann_index: Dict[str, dict]
    ) -> tuple:
        """Resolve start/end ms from a list of TimePoint target references.

        SWT TimeFrames reference TimePoints like ["v_2:tp_369", "v_2:tp_370", ...].
        Each TimePoint has a `timePoint` property with the ms value.
        We take min/max of resolved timepoints as the range.
        """
        times = []
        for ref_id in targets:
            ann = ann_index.get(ref_id)
            if ann:
                tp = ann.get("properties", {}).get("timePoint")
                if isinstance(tp, (int, float)):
                    times.append(int(tp))
        if times:
            return min(times), max(times)
        return None, None

    def _find_time_for_annotation(self, props: dict, ann_index: Dict[str, dict]) -> Optional[int]:
        """Try to resolve a time in ms for an annotation's properties."""
        # Direct time properties
        for key in ("start", "timePoint", "time"):
            if key in props and isinstance(props[key], (int, float)):
                return int(props[key])

        # Check targets — may reference TimePoints or TimeFrames
        targets = props.get("targets", props.get("target", []))
        if isinstance(targets, str):
            targets = [targets]
        if isinstance(targets, list):
            for ref_id in targets:
                resolved = ann_index.get(ref_id)
                if resolved:
                    rprops = resolved.get("properties", {})
                    for key in ("start", "timePoint", "time"):
                        if key in rprops and isinstance(rprops[key], (int, float)):
                            return int(rprops[key])

        # Check origin reference (VLM captioner pattern: TextDocument.origin → TimeFrame)
        origin_ref = props.get("origin", "")
        if origin_ref and isinstance(origin_ref, str):
            resolved = ann_index.get(origin_ref)
            if resolved:
                rprops = resolved.get("properties", {})
                for key in ("start", "timePoint", "time"):
                    if key in rprops and isinstance(rprops[key], (int, float)):
                        return int(rprops[key])
                # TimeFrame may use targets → TimePoints (SWT pattern)
                tf_targets = rprops.get("targets", [])
                if tf_targets:
                    start, _ = self._resolve_timeframe_range(tf_targets, ann_index)
                    if start is not None:
                        return int(start)

        # Check document reference (skip VideoDocument refs like "d1")
        doc_ref = props.get("document", "")
        if doc_ref and isinstance(doc_ref, str):
            resolved = ann_index.get(doc_ref)
            if resolved:
                # Don't recurse into VideoDocument/AudioDocument refs
                rtype = resolved.get("@type", "")
                if "VideoDocument" in rtype or "AudioDocument" in rtype:
                    return None
                return self._find_time_for_annotation(
                    resolved.get("properties", {}), ann_index
                )

        return None

    @staticmethod
    def _detect_view_frame_rate(view: dict, ann_index: Dict[str, dict], duration_ms: int) -> Optional[float]:
        """Detect if a view uses frame numbers instead of milliseconds.

        Checks the view's ``contains`` metadata for a declared ``timeUnit``.
        If not declared, heuristically compares max TimePoint values against
        the known video duration to decide if values are frames.

        Returns:
            Frame rate (e.g. 29.97) if values appear to be frame numbers,
            or None if values are already in milliseconds.
        """
        contains = view.get("metadata", {}).get("contains", {})
        for key, meta in contains.items():
            if "TimePoint" in key or "TimeFrame" in key:
                tu = meta.get("timeUnit", "")
                if tu == "milliseconds":
                    return None
                if tu in ("frames", "frame"):
                    # Try to get fps from metadata
                    return meta.get("fps", 29.97)

        # No timeUnit declared — heuristic: if the max TimePoint value is much
        # smaller than duration_ms, values are probably frame numbers.
        if duration_ms <= 0:
            return None

        max_tp = 0
        for ann in view.get("annotations", []):
            tp = ann.get("properties", {}).get("timePoint")
            if tp is not None and isinstance(tp, (int, float)):
                max_tp = max(max_tp, tp)

        if max_tp <= 0:
            return None

        # If max value is < duration_ms / 10, it's likely frames
        # (e.g., 24000 frames vs 800000 ms for a 13-min video)
        if max_tp < duration_ms / 10:
            # Estimate fps from the ratio
            estimated_fps = max_tp / (duration_ms / 1000)
            if 20 < estimated_fps < 65:  # reasonable fps range
                logger.debug(f"Detected frame-based TimePoints in view (est. {estimated_fps:.1f} fps)")
                return estimated_fps

        return None

    def _find_segment_for_time(self, segments: List[SegmentEntry], time_ms: Optional[int]) -> Optional[SegmentEntry]:
        """Find the segment that contains the given time point."""
        if time_ms is None:
            return None
        for seg in segments:
            if seg.start_ms <= time_ms <= seg.end_ms:
                return seg
        # If time is beyond all segments, assign to nearest
        if segments:
            closest = min(segments, key=lambda s: min(abs(s.start_ms - time_ms), abs(s.end_ms - time_ms)))
            return closest
        return None

    def _estimate_duration(self, mmif_dict: dict, ann_index: Dict[str, dict]) -> int:
        """Estimate video duration in ms from MMIF metadata, annotations, or ffprobe."""
        # Check document properties
        for doc in mmif_dict.get("documents", []):
            props = doc.get("properties", {})
            if "duration" in props:
                dur = props["duration"]
                if isinstance(dur, (int, float)):
                    return int(dur) if dur > 1000 else int(dur * 1000)

        # Fall back to max time found in any annotation
        max_time = 0
        for ann in ann_index.values():
            props = ann.get("properties", {})
            for key in ("end", "start", "timePoint"):
                val = props.get(key)
                if isinstance(val, (int, float)) and val > max_time:
                    max_time = int(val)
        if max_time > 0:
            return max_time

        # Last resort: ffprobe on the video file
        for doc in mmif_dict.get("documents", []):
            loc = doc.get("properties", {}).get("location", "")
            if loc.startswith("file://"):
                video_path = loc[7:]
                dur = self._ffprobe_duration_ms(video_path)
                if dur > 0:
                    return dur
        return 0

    @staticmethod
    def _ffprobe_duration_ms(video_path: str) -> int:
        """Get video duration in ms via ffprobe."""
        import subprocess
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(float(result.stdout.strip()) * 1000)
        except Exception as e:
            logger.debug(f"ffprobe failed for {video_path}: {e}")
        return 0

    # ------------------------------------------------------------------
    # ChromaDB operations
    # ------------------------------------------------------------------

    def _upsert_segments(self, video_id: str, segments: List[SegmentEntry]):
        """Upsert segments into ChromaDB collection.

        Skips silently if ChromaDB or embedding service is unavailable.
        """
        if not _check_ollama_available():
            return  # Skip ChromaDB when embeddings are unavailable
        collection = self._get_collection()
        if collection is None:
            return

        # Delete existing segments for this video (idempotent re-index)
        try:
            existing = collection.get(where={"video_id": video_id})
            if existing and existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.debug(f"Deleted {len(existing['ids'])} existing segments for {video_id}")
        except Exception as e:
            logger.warning(f"Could not clean existing segments: {e}")

        if not segments:
            return

        # Prepare batch
        ids = []
        documents = []
        metadatas = []

        for seg in segments:
            if not seg.summary.strip():
                continue
            ids.append(seg.segment_id)
            documents.append(seg.summary)
            metadatas.append({
                "video_id": video_id,
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "scene_label": seg.scene_label or "",
                "segment_type": seg.segment_type or "",
                "has_ocr": bool(seg.ocr_text),
                "has_asr": bool(seg.asr_transcript),
            })

        if not ids:
            return

        try:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(f"Upserted {len(ids)} segments to ChromaDB for {video_id}")
        except Exception as e:
            logger.warning(f"ChromaDB upsert failed: {e}")

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Semantic search across indexed video segments."""
        collection = self._get_collection()
        if collection is None:
            return self._fallback_keyword_search(query, top_k, video_id)

        where_filter = {"video_id": video_id} if video_id else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
        except Exception as e:
            logger.warning(f"ChromaDB query failed: {e}")
            return self._fallback_keyword_search(query, top_k, video_id)

        hits = []
        if results and results.get("ids") and results["ids"][0]:
            for i, seg_id in enumerate(results["ids"][0]):
                hit = {
                    "segment_id": seg_id,
                    "summary": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                }
                hits.append(hit)

        return hits

    def _fallback_keyword_search(self, query: str, top_k: int, video_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Simple keyword search over JSON files when ChromaDB is unavailable."""
        query_terms = query.lower().split()
        hits = []

        for json_path in self.index_dir.glob("*.json"):
            if video_id and json_path.stem != video_id:
                continue
            try:
                with open(json_path) as f:
                    doc = json.load(f)
                for seg in doc.get("segments", []):
                    summary = seg.get("summary", "").lower()
                    score = sum(1 for t in query_terms if t in summary)
                    if score > 0:
                        hits.append({
                            "segment_id": seg["segment_id"],
                            "summary": seg.get("summary", ""),
                            "metadata": {
                                "video_id": doc["video_id"],
                                "start_ms": seg["start_ms"],
                                "end_ms": seg["end_ms"],
                                "scene_label": seg.get("scene_label", ""),
                            },
                            "distance": -score,  # Lower is better for distance
                        })
            except Exception:
                continue

        hits.sort(key=lambda h: h["distance"])
        return hits[:top_k]

    def browse_timeline(self, video_id: str, start_ms: int = 0, end_ms: int = -1) -> List[Dict[str, Any]]:
        """Load JSON index and filter by time range.

        For v2 layered indexes, returns shot-based context dicts.
        For v1, returns flat segment dicts.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return []

        with open(json_path) as f:
            doc = json.load(f)

        if end_ms < 0:
            end_ms = doc.get("duration_ms", float("inf"))

        if doc.get("schema_version", 1) >= 2:
            # v2: return layer data for overlapping time ranges
            layers = doc.get("layers", {})
            results = query_time_range(layers, start_ms, end_ms)
            # Flatten into a list of items with layer info
            items = []
            for layer_name, layer_items in results.items():
                for item in layer_items:
                    flat = dict(item)
                    flat["_layer"] = layer_name
                    items.append(flat)
            items.sort(key=lambda x: x.get("start_ms", 0))
            return items

        segments = doc.get("segments", [])
        return [
            s for s in segments
            if s["start_ms"] < end_ms and s["end_ms"] > start_ms
        ]

    def filter_segments(
        self, video_id: str, scene_label: Optional[str] = None, has_text: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Structured filtering on JSON index."""
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return []

        with open(json_path) as f:
            doc = json.load(f)

        results = []
        for seg in doc.get("segments", []):
            if scene_label and seg.get("scene_label", "") != scene_label:
                continue
            if has_text is True:
                if not seg.get("ocr_text") and not seg.get("asr_transcript"):
                    continue
            if has_text is False:
                if seg.get("ocr_text") or seg.get("asr_transcript"):
                    continue
            results.append(seg)

        return results

    def get_video_summary(self, video_id: str) -> Dict[str, Any]:
        """Get segment/layer counts, label distribution, text stats for a video."""
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return {"error": f"No index found for {video_id}"}

        with open(json_path) as f:
            doc = json.load(f)

        # v2: delegate to layer-aware summary
        if doc.get("schema_version", 1) >= 2:
            return self.get_video_summary_v2(video_id)

        segments = doc.get("segments", [])
        label_counts: Dict[str, int] = {}
        total_ocr_len = 0
        total_asr_len = 0
        all_entities: List[Dict[str, str]] = []

        for seg in segments:
            lbl = seg.get("scene_label", "unlabeled") or "unlabeled"
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            total_ocr_len += len(seg.get("ocr_text", ""))
            total_asr_len += len(seg.get("asr_transcript", ""))
            all_entities.extend(seg.get("named_entities", []))

        entity_type_counts: Dict[str, int] = {}
        for ent in all_entities:
            etype = ent.get("type", "UNKNOWN")
            entity_type_counts[etype] = entity_type_counts.get(etype, 0) + 1

        return {
            "video_id": video_id,
            "video_path": doc.get("video_path", ""),
            "duration_ms": doc.get("duration_ms", 0),
            "indexed_at": doc.get("indexed_at", ""),
            "total_segments": len(segments),
            "label_counts": label_counts,
            "total_ocr_chars": total_ocr_len,
            "total_asr_chars": total_asr_len,
            "total_entities": len(all_entities),
            "entity_type_counts": entity_type_counts,
        }

    def build_context(
        self,
        video_id: str,
        layers: Optional[set] = None,
        max_chars: int = 8000,
    ) -> Optional[str]:
        """Build a text context string from a video index for LLM consumption.

        Supports layer-selective ablation: only include specified index layers.

        Args:
            video_id: Video identifier.
            layers: Set of layer names to include. If None, include all.
                Valid: "asr", "ocr", "captions", "entities", "labels"
            max_chars: Maximum context length (truncated if exceeded).

        Returns:
            Context string, or None if video not found or layers empty.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return None
        if layers is not None and len(layers) == 0:
            return None

        with open(json_path) as f:
            doc = json.load(f)

        include_all = layers is None
        segments = doc.get("segments", [])
        lines = [f"Video: {video_id}", f"Segments: {len(segments)}", ""]

        for i, seg in enumerate(segments):
            parts = [f"[Seg {i}: {seg['start_ms'] / 1000:.1f}s-{seg['end_ms'] / 1000:.1f}s]"]
            if (include_all or "labels" in layers) and seg.get("scene_label"):
                parts.append(f"type={seg['scene_label']}")
            if (include_all or "ocr" in layers) and seg.get("ocr_text", "").strip():
                parts.append(f"OCR: {seg['ocr_text'][:100]}")
            if (include_all or "asr" in layers) and seg.get("asr_transcript", "").strip():
                parts.append(f"ASR: {seg['asr_transcript'][:100]}")
            if (include_all or "captions" in layers) and seg.get("visual_caption", "").strip():
                parts.append(f"Visual: {seg['visual_caption'][:100]}")
            if include_all or "entities" in layers:
                ents = seg.get("named_entities", [])
                if ents:
                    parts.append(f"Entities: {', '.join(e['text'] for e in ents[:5])}")
                descs = seg.get("entity_descriptions", {})
                if descs:
                    for name, desc in list(descs.items())[:3]:
                        parts.append(f"  {name}: {desc[:80]}")
            if len(parts) > 1:
                lines.append(" | ".join(parts))

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[... truncated ...]"
        return context

    # ------------------------------------------------------------------
    # Layer-aware query methods (v2 schema)
    # ------------------------------------------------------------------

    def _load_doc(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load a video index document from disk."""
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return None
        with open(json_path) as f:
            return json.load(f)

    def _is_v2(self, doc: dict) -> bool:
        """Check if a document uses the v2 layered schema."""
        return doc.get("schema_version", 1) >= 2

    def get_layer(self, video_id: str, layer_name: str) -> List[dict]:
        """Get all items from a specific layer."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return []
        return doc.get("layers", {}).get(layer_name, {}).get("items", [])

    def query_at_time(
        self, video_id: str, time_ms: int,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, list]:
        """Get everything active at a specific time point."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return {}
        return query_time_range(doc["layers"], time_ms, time_ms + 1, layer_names)

    def query_time_range_v2(
        self, video_id: str, start_ms: int, end_ms: int,
        layer_names: Optional[List[str]] = None,
    ) -> Dict[str, list]:
        """Get everything overlapping a time range across selected layers."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return {}
        return query_time_range(doc["layers"], start_ms, end_ms, layer_names)

    def browse_timeline_v2(
        self, video_id: str, start_ms: int = 0, end_ms: int = -1,
    ) -> Dict[str, list]:
        """Browse all layers in a time window."""
        doc = self._load_doc(video_id)
        if not doc:
            return {}

        if self._is_v2(doc):
            if end_ms < 0:
                end_ms = doc.get("duration_ms", float("inf"))
            return query_time_range(doc["layers"], start_ms, end_ms)
        else:
            # Fallback for v1
            return {"segments": self.browse_timeline(video_id, start_ms, end_ms)}

    def search_layers(
        self, video_id: str, query: str,
        layer_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Text search across layer items.

        Returns list of {layer, item, score} dicts sorted by relevance.
        """
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return []

        query_terms = query.lower().split()
        hits = []
        names = layer_names or list(doc.get("layers", {}).keys())

        for name in names:
            layer = doc.get("layers", {}).get(name)
            if not layer:
                continue
            for item in layer.get("items", []):
                # Search in all string fields of the item
                searchable = " ".join(
                    str(v) for v in item.values()
                    if isinstance(v, str)
                ).lower()
                # Also search in nested entries (credits)
                for entry in item.get("entries", []):
                    searchable += " " + " ".join(str(v) for v in entry.values() if isinstance(v, str)).lower()
                score = sum(1 for t in query_terms if t in searchable)
                if score > 0:
                    hits.append({"layer": name, "item": item, "score": score})

        hits.sort(key=lambda h: -h["score"])
        return hits

    def build_context_v2(
        self, video_id: str,
        layer_names: Optional[List[str]] = None,
        max_chars: int = 8000,
        time_range: Optional[tuple] = None,
    ) -> Optional[str]:
        """Build LLM-consumable text context from a layered video index.

        Args:
            video_id: Video identifier.
            layer_names: Layers to include. None = all layers.
            max_chars: Maximum context length.
            time_range: Optional (start_ms, end_ms) to restrict.

        Returns:
            Formatted context string, or None if not found.
        """
        doc = self._load_doc(video_id)
        if not doc:
            return None

        # Fall back to v1 build_context for old indexes
        if not self._is_v2(doc):
            return self.build_context(video_id, layers=set(layer_names) if layer_names else None, max_chars=max_chars)

        layers = doc.get("layers", {})
        names = layer_names or list(layers.keys())

        # If time range specified, filter items
        if time_range:
            layers = query_time_range(layers, time_range[0], time_range[1], names)
            # Wrap back into layer format
            layers = {k: {"items": v} for k, v in layers.items()}

        lines = [
            f"Video: {video_id}",
            f"Duration: {doc.get('duration_ms', 0) / 1000:.1f}s",
            "",
        ]

        # Format each layer
        for name in names:
            layer_data = layers.get(name)
            if not layer_data:
                continue
            items = layer_data.get("items", [])
            if not items:
                continue

            lines.append(f"=== {name.upper()} ({len(items)} items) ===")

            for item in items:
                t_start = item.get("start_ms", 0) / 1000
                t_end = item.get("end_ms", 0) / 1000
                parts = [f"[{t_start:.1f}s-{t_end:.1f}s]"]

                if name == "scenes":
                    parts.append(f"label={item.get('label', '')}")
                elif name == "asr":
                    text = item.get("text", "")
                    if len(text) > 150:
                        text = text[:150] + "..."
                    parts.append(text)
                elif name == "ocr":
                    parts.append(f"OCR: {item.get('text', '')[:120]}")
                    if item.get("scene_label"):
                        parts.append(f"({item['scene_label']})")
                elif name == "visual_captions":
                    parts.append(f"Visual: {item.get('text', '')[:120]}")
                elif name == "speakers":
                    spk = item.get("speaker_name", item.get("speaker_id", "?"))
                    parts.append(f"Speaker: {spk}")
                elif name == "chapters":
                    parts.append(f"Chapter: {item.get('title', '')}")
                elif name == "credits":
                    entries = item.get("entries", [])
                    names_str = ", ".join(f"{e.get('name')} ({e.get('role')})" for e in entries[:5])
                    parts.append(f"Credits: {names_str}")
                elif name == "entities":
                    parts.append(f"{item.get('text', '')} ({item.get('type', '')})")
                    if item.get("grounding"):
                        parts.append(f"= {item['grounding'][:80]}")
                elif name == "relations":
                    parts.append(f"{item.get('subject', '')} → {item.get('predicate', '')} → {item.get('object', '')}")
                else:
                    # Generic: show all string fields
                    for k, v in item.items():
                        if k not in ("id", "start_ms", "end_ms") and isinstance(v, str) and v:
                            parts.append(f"{k}: {v[:80]}")
                            break

                lines.append(" | ".join(parts))
            lines.append("")

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[... truncated ...]"
        return context

    def get_video_summary_v2(self, video_id: str) -> Dict[str, Any]:
        """Get per-layer stats for a video."""
        doc = self._load_doc(video_id)
        if not doc:
            return {"error": f"No index found for {video_id}"}

        if not self._is_v2(doc):
            return self.get_video_summary(video_id)

        layers = doc.get("layers", {})
        layer_stats = {}
        for name, layer in layers.items():
            items = layer.get("items", [])
            stat = {"count": len(items), "source": layer.get("source", "")}
            if items:
                stat["time_range_ms"] = [
                    min(i["start_ms"] for i in items),
                    max(i["end_ms"] for i in items),
                ]
                # Count text items
                text_items = sum(1 for i in items if i.get("text"))
                if text_items:
                    stat["with_text"] = text_items
            layer_stats[name] = stat

        return {
            "video_id": video_id,
            "schema_version": 2,
            "video_path": doc.get("video_path", ""),
            "broadcast_date": doc.get("broadcast_date", ""),
            "duration_ms": doc.get("duration_ms", 0),
            "indexed_at": doc.get("indexed_at", ""),
            "layers": layer_stats,
        }

    # ------------------------------------------------------------------
    # Layer-aware NER / relation extraction
    # ------------------------------------------------------------------

    def extract_entities_from_layers(self, video_id: str) -> int:
        """Run spaCy NER on all text in layer items, write to entities layer.

        Processes text from ASR, OCR, and visual_captions layers.

        Returns:
            Number of entities extracted.
        """
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return self.extract_entities_from_transcripts(video_id)

        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as e:
            logger.warning(f"spaCy not available: {e}")
            return 0

        layers = doc.get("layers", {})
        entities_layer = layers.setdefault("entities", {
            "source": "spacy+grounding",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
        })

        existing_entities = {
            (e.get("text", ""), e.get("type", ""), e.get("source_id", ""))
            for e in entities_layer.get("items", [])
        }
        idx = len(entities_layer.get("items", []))
        total = 0

        # Process text-bearing layers
        for layer_name in ("asr", "ocr", "visual_captions"):
            layer = layers.get(layer_name, {})
            for item in layer.get("items", []):
                text = item.get("text", "")
                if len(text) < 10:
                    continue

                spacy_doc = nlp(text)
                for ent in spacy_doc.ents:
                    if ent.label_ in ("CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"):
                        continue
                    ent_text = ent.text.strip()
                    if len(ent_text) <= 1:
                        continue
                    if not _is_groundable_entity(ent_text, ent.label_):
                        continue

                    key = (ent_text, ent.label_, item["id"])
                    if key not in existing_entities:
                        entities_layer.setdefault("items", []).append({
                            "id": f"ent_{idx}",
                            "start_ms": item["start_ms"],
                            "end_ms": item["end_ms"],
                            "text": ent_text,
                            "type": ent.label_,
                            "source_layer": layer_name,
                            "source_id": item["id"],
                        })
                        existing_entities.add(key)
                        idx += 1
                        total += 1

        # Save
        json_path = self.index_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        logger.info(f"Extracted {total} entities from layers for {video_id}")
        return total

    def extract_relations_from_layers(self, video_id: str) -> int:
        """Extract SVO triples from ASR/caption text, write to relations layer."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return self.extract_relations(video_id)

        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as e:
            logger.warning(f"spaCy not available: {e}")
            return 0

        layers = doc.get("layers", {})
        relations_layer = layers.setdefault("relations", {
            "source": "spacy-dep-parse+cross-modal",
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "items": [],
        })

        existing = {
            (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
            for r in relations_layer.get("items", [])
        }
        idx = len(relations_layer.get("items", []))
        total = 0

        for layer_name in ("asr", "visual_captions"):
            layer = layers.get(layer_name, {})
            for item in layer.get("items", []):
                text = item.get("text", "")
                if len(text) < 15:
                    continue

                spacy_doc = nlp(text)
                for sent in spacy_doc.sents:
                    triples = _extract_svo_triples(sent)
                    for subj, pred, obj in triples:
                        key = (subj, pred, obj)
                        if key not in existing:
                            relations_layer.setdefault("items", []).append({
                                "id": f"rel_{idx}",
                                "start_ms": item["start_ms"],
                                "end_ms": item["end_ms"],
                                "subject": subj,
                                "predicate": pred,
                                "object": obj,
                                "source_layers": [layer_name],
                                "relation_type": "svo",
                            })
                            existing.add(key)
                            idx += 1
                            total += 1

        # Save
        json_path = self.index_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        logger.info(f"Extracted {total} SVO relations from layers for {video_id}")
        return total

    def enrich_entities_v2(self, video_id: str) -> Dict[str, str]:
        """Ground entities in the entities layer via Wikidata/Wikipedia."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return self.enrich_entities(video_id)

        layers = doc.get("layers", {})
        entities_layer = layers.get("entities", {})
        items = entities_layer.get("items", [])

        if not items:
            return {}

        # Collect unique entity texts
        unique: Dict[str, str] = {}
        for item in items:
            text = item.get("text", "").strip()
            if text and text not in unique:
                unique[text] = item.get("type", "UNKNOWN")

        # Look up each entity
        grounded: Dict[str, str] = {}
        for entity_text, entity_type in unique.items():
            if not _is_groundable_entity(entity_text, entity_type):
                grounded[entity_text] = ""
                continue

            parts = []
            wd_desc = wikidata_lookup(entity_text)
            if wd_desc:
                parts.append(wd_desc)
            wiki_desc = wikipedia_lookup(entity_text)
            if wiki_desc and (not wd_desc or len(wiki_desc) > len(wd_desc)):
                parts.append(wiki_desc)

            grounded[entity_text] = " ".join(parts).strip()

        # Write grounding back into entity items
        for item in items:
            text = item.get("text", "").strip()
            if text in grounded and grounded[text]:
                item["grounding"] = grounded[text]

        entities_layer["indexed_at"] = datetime.now(timezone.utc).isoformat()

        json_path = self.index_dir / f"{video_id}.json"
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        grounded_count = sum(1 for v in grounded.values() if v)
        logger.info(f"Enriched {grounded_count}/{len(grounded)} entities for {video_id}")
        return grounded

    def resolve_entity_variants_v2(self, video_id: str) -> Dict[str, int]:
        """Merge variant spellings in the entities layer."""
        doc = self._load_doc(video_id)
        if not doc or not self._is_v2(doc):
            return self.resolve_entity_variants(video_id)

        layers = doc.get("layers", {})
        entities_layer = layers.get("entities", {})
        items = entities_layer.get("items", [])
        if not items:
            return {"merged": 0, "total": 0}

        # Collect all (text, type) pairs
        entity_occurrences: Dict[str, Dict] = {}
        ocr_texts = " ".join(
            i.get("text", "").lower()
            for i in layers.get("ocr", {}).get("items", [])
        )

        for item in items:
            text = item.get("text", "").strip()
            etype = item.get("type", "")
            if not text or len(text) <= 1:
                continue
            key = text.upper().strip()
            if key not in entity_occurrences:
                entity_occurrences[key] = {
                    "text": text, "type": etype,
                    "count": 0, "from_ocr": False,
                }
            entity_occurrences[key]["count"] += 1
            if text.lower() in ocr_texts:
                entity_occurrences[key]["from_ocr"] = True

        # Build merge groups (same logic as v1)
        keys = sorted(entity_occurrences.keys(), key=len, reverse=True)
        merged_into: Dict[str, str] = {}
        for i, key_a in enumerate(keys):
            if key_a in merged_into:
                continue
            words_a = set(key_a.split())
            for key_b in keys[i + 1:]:
                if key_b in merged_into:
                    continue
                words_b = set(key_b.split())
                if len(words_b) < 1 or len(min(words_b, key=len)) < 4:
                    continue
                if words_b.issubset(words_a):
                    merged_into[key_b] = key_a

        groups: Dict[str, list] = {}
        for key in keys:
            canonical_key = merged_into.get(key, key)
            groups.setdefault(canonical_key, []).append(key)

        canonical_map: Dict[str, str] = {}
        merge_count = 0
        for canonical_key, group_keys in groups.items():
            if len(group_keys) <= 1:
                continue
            candidates = [entity_occurrences[k] for k in group_keys]
            best = max(candidates, key=lambda c: (c["from_ocr"], len(c["text"]), c["count"]))
            for k in group_keys:
                variant = entity_occurrences[k]["text"]
                if variant != best["text"]:
                    canonical_map[variant] = best["text"]
                    merge_count += 1

        if canonical_map:
            for item in items:
                text = item.get("text", "").strip()
                if text in canonical_map:
                    item["text"] = canonical_map[text]

            json_path = self.index_dir / f"{video_id}.json"
            with open(json_path, "w") as f:
                json.dump(doc, f, indent=2)

        logger.info(f"Entity resolution v2 for {video_id}: {merge_count} variants merged")
        return {"merged": merge_count, "total": len(entity_occurrences)}

    def extract_entities_from_transcripts(self, video_id: str) -> int:
        """Run spaCy NER on all text in indexed segments to extract named entities.

        Processes ALL text fields — ASR transcripts, OCR text, and visual
        captions — extracts PERSON/ORG/GPE/etc. entities, and writes them back.

        Returns:
            Number of entities extracted.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return 0

        try:
            import spacy
        except ImportError:
            logger.warning("spaCy not installed; cannot extract entities from transcripts")
            return 0

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model en_core_web_sm not found; run: python -m spacy download en_core_web_sm")
            return 0

        with open(json_path) as f:
            doc = json.load(f)

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]
        total_entities = 0

        for seg in segments:
            # Combine ALL text fields for NER: ASR, OCR, and visual captions
            text_parts = []
            if seg.asr_transcript:
                text_parts.append(seg.asr_transcript)
            if seg.ocr_text:
                text_parts.append(seg.ocr_text)
            if seg.visual_caption:
                text_parts.append(seg.visual_caption)
            text = " ".join(text_parts)
            if len(text) < 10:
                continue

            spacy_doc = nlp(text)
            existing = {(e["text"], e["type"]) for e in seg.named_entities}

            for ent in spacy_doc.ents:
                # Filter to useful entity types
                if ent.label_ in ("CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"):
                    continue
                text = ent.text.strip()
                if len(text) <= 1:
                    continue
                # Skip common words that spaCy misclassifies as entities
                if not _is_groundable_entity(text, ent.label_):
                    continue
                key = (text, ent.label_)
                if key not in existing:
                    seg.named_entities.append({
                        "text": text,
                        "type": ent.label_,
                    })
                    existing.add(key)
                    total_entities += 1

            # Regenerate summary
            seg.summary = seg.to_summary()

        # Save
        doc["segments"] = [s.to_dict() for s in segments]
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        # Re-upsert ChromaDB
        self._upsert_segments(video_id, segments)

        logger.info(f"Extracted {total_entities} entities from transcripts for {video_id}")
        return total_entities

    def extract_relations(self, video_id: str) -> int:
        """Extract (subject, predicate, object) triples from segment text.

        Uses spaCy dependency parsing to find:
        - Subject-verb-object triples from ASR transcripts
        - Entity-action-entity patterns
        - Named entity co-occurrence with verbal predicates

        Returns:
            Number of relations extracted.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return 0

        try:
            import spacy
        except ImportError:
            logger.warning("spaCy not installed; cannot extract relations")
            return 0

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found")
            return 0

        with open(json_path) as f:
            doc = json.load(f)

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]
        total_relations = 0

        # Pre-collect next-segment text for overlap context.
        # We append the start of the next segment's ASR so that sentences
        # split across a boundary are parsed whole, with any resulting
        # triples attributed to the segment where the sentence begins.
        OVERLAP_WORDS = 50

        for i, seg in enumerate(segments):
            # Combine text sources
            texts = []
            if seg.asr_transcript:
                texts.append(seg.asr_transcript)
            if seg.visual_caption:
                texts.append(seg.visual_caption)
            text = " ".join(texts)

            # Append overlap from next segment's ASR
            next_asr_prefix = ""
            if i + 1 < len(segments) and segments[i + 1].asr_transcript:
                words = segments[i + 1].asr_transcript.split()
                next_asr_prefix = " ".join(words[:OVERLAP_WORDS])
                text = text + " " + next_asr_prefix

            if len(text) < 15:
                continue

            spacy_doc = nlp(text)
            existing = {tuple(r) for r in seg.relations}

            # Track where the overlap starts so we can filter triples
            # that originate entirely within the overlap zone
            own_text_len = len(text) - len(next_asr_prefix) - (1 if next_asr_prefix else 0)

            for sent in spacy_doc.sents:
                # Skip sentences that start entirely in the overlap zone
                if next_asr_prefix and sent.start_char >= own_text_len:
                    break
                triples = _extract_svo_triples(sent)
                for subj, pred, obj in triples:
                    triple = [subj, pred, obj]
                    key = tuple(triple)
                    if key not in existing:
                        seg.relations.append(triple)
                        existing.add(key)
                        total_relations += 1

            seg.summary = seg.to_summary()

        doc["segments"] = [s.to_dict() for s in segments]
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        logger.info(f"Extracted {total_relations} relations for {video_id}")
        return total_relations

    def enrich_entities(self, video_id: str) -> Dict[str, str]:
        """Look up named entities via Wikidata/Wikipedia/web and store descriptions.

        Uses a layered lookup strategy per entity:
        1. Wikidata — structured facts (positions, occupations, relationships)
        2. Wikipedia — prose description (intro paragraph)
        3. Web search — DuckDuckGo fallback for entities not in knowledge bases

        If both Wikidata and Wikipedia return results, they are combined:
        the Wikipedia prose provides narrative context while Wikidata adds
        structured facts (e.g., "position held: Secretary of State").

        Stores results in each segment's `grounded_entities` dict,
        regenerates summaries, and re-saves both JSON and ChromaDB.

        Returns:
            Dict mapping entity text → description (or empty string if not found).
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return {}

        with open(json_path) as f:
            doc = json.load(f)

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]

        # Collect unique entity texts across all segments
        unique_entities: Dict[str, str] = {}  # text → type
        for seg in segments:
            for ent in seg.named_entities:
                text = ent.get("text", "").strip()
                if text and text not in unique_entities:
                    unique_entities[text] = ent.get("type", "UNKNOWN")

        if not unique_entities:
            logger.info(f"No named entities to enrich for {video_id}")
            return {}

        # Look up each entity with layered strategy
        grounded: Dict[str, str] = {}
        for entity_text, entity_type in unique_entities.items():
            # Skip common/short entities unlikely to have knowledge base entries
            if not _is_groundable_entity(entity_text, entity_type):
                grounded[entity_text] = ""
                continue

            parts = []

            # Layer 1: Wikidata structured facts (best for cross-referencing)
            wd_desc = wikidata_lookup(entity_text)
            if wd_desc:
                parts.append(wd_desc)

            # Layer 2: Wikipedia prose (narrative context)
            wiki_desc = wikipedia_lookup(entity_text)
            if wiki_desc:
                # Avoid duplication — skip if Wikidata already captured the gist
                if not wd_desc or len(wiki_desc) > len(wd_desc):
                    parts.append(wiki_desc)

            # Layer 3: Web search fallback (disabled — adds noise)
            # if not parts:
            #     web_desc = web_search_entity(entity_text)
            #     if web_desc:
            #         parts.append(web_desc)

            desc = " ".join(parts).strip()
            grounded[entity_text] = desc
            if desc:
                logger.debug(f"Grounded '{entity_text}': {desc[:80]}...")

        # Write grounded descriptions back into segments
        for seg in segments:
            for ent in seg.named_entities:
                text = ent.get("text", "").strip()
                if text in grounded and grounded[text]:
                    seg.grounded_entities[text] = grounded[text]
            # Regenerate summary with enriched content
            seg.summary = seg.to_summary()

        # Re-save JSON
        doc["segments"] = [s.to_dict() for s in segments]
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)
        logger.info(f"Enriched {sum(1 for v in grounded.values() if v)} / {len(grounded)} entities for {video_id}")

        # Re-upsert to ChromaDB with enriched summaries
        self._upsert_segments(video_id, segments)

        return grounded

    def disambiguate_with_date(self, video_id: str) -> Dict[str, str]:
        """Use broadcast date to disambiguate entities via Wikidata temporal queries.

        For entities like "the president", "the governor", etc., uses the
        broadcast date to query Wikidata for who held that position at that time.

        Also resolves ambiguous person names by checking if Wikidata returns
        multiple candidates and selecting the one active during the broadcast era.

        Returns:
            Dict of entity_text → disambiguated description.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return {}

        with open(json_path) as f:
            doc = json.load(f)

        broadcast_date = doc.get("broadcast_date", "")
        if not broadcast_date:
            broadcast_date = _extract_broadcast_date(video_id)
        if not broadcast_date:
            logger.info(f"No broadcast date for {video_id}, skipping disambiguation")
            return {}

        # Extract year from broadcast date
        broadcast_year = int(broadcast_date[:4])
        logger.info(f"Disambiguating entities for {video_id} (broadcast: {broadcast_date})")

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]

        # Collect entities that could benefit from temporal disambiguation
        disambiguated = {}
        unique_entities = {}
        for seg in segments:
            for ent in seg.named_entities:
                text = ent.get("text", "").strip()
                etype = ent.get("type", "")
                if text and text not in unique_entities:
                    unique_entities[text] = etype

        for entity_text, entity_type in unique_entities.items():
            # Skip entities already well-grounded
            if entity_text in disambiguated:
                continue

            result = _disambiguate_entity_temporal(
                entity_text, entity_type, broadcast_year
            )
            if result:
                disambiguated[entity_text] = result

        # Write disambiguated results back as improved groundings
        if disambiguated:
            for seg in segments:
                for ent in seg.named_entities:
                    text = ent.get("text", "").strip()
                    if text in disambiguated:
                        seg.grounded_entities[text] = disambiguated[text]
                seg.summary = seg.to_summary()

            doc["segments"] = [s.to_dict() for s in segments]
            with open(json_path, "w") as f:
                json.dump(doc, f, indent=2)

        logger.info(f"Disambiguated {len(disambiguated)} entities for {video_id}")
        return disambiguated

    def clean_entities(self, video_id: str) -> Dict[str, int]:
        """Remove noisy entities and bad groundings from an existing index.

        Strips:
        - Entities failing the _is_groundable_entity check
        - Grounded descriptions for removed entities
        - Hallucinated captions matching known patterns

        Returns dict with counts of removed items.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return {}

        with open(json_path) as f:
            doc = json.load(f)

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]
        removed_entities = 0
        removed_groundings = 0
        cleaned_captions = 0

        for seg in segments:
            # Clean entities
            original_count = len(seg.named_entities)
            seg.named_entities = [
                e for e in seg.named_entities
                if _is_groundable_entity(e.get("text", ""), e.get("type", ""))
            ]
            removed_entities += original_count - len(seg.named_entities)

            # Clean groundings — remove entries for deleted entities
            kept_entity_texts = {e["text"] for e in seg.named_entities}
            original_grounding_count = len(seg.grounded_entities)
            seg.grounded_entities = {
                k: v for k, v in seg.grounded_entities.items()
                if k in kept_entity_texts
            }
            removed_groundings += original_grounding_count - len(seg.grounded_entities)

            # Clean hallucinated captions
            if seg.visual_caption and _is_hallucinated_caption(seg.visual_caption):
                seg.visual_caption = ""
                cleaned_captions += 1

            seg.summary = seg.to_summary()

        doc["segments"] = [s.to_dict() for s in segments]
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        result = {
            "removed_entities": removed_entities,
            "removed_groundings": removed_groundings,
            "cleaned_captions": cleaned_captions,
        }
        logger.info(f"Cleaned {video_id}: {result}")
        return result

    def resolve_entity_variants(self, video_id: str) -> Dict[str, int]:
        """Merge variant spellings of the same entity across all segments.

        Resolution strategy (same as kg/entity_resolution.py):
        1. Exact normalized name match → merge to longest form
        2. Substring containment → merge (e.g., "Woodruff" + "Judy Woodruff")
        3. Prefer chyron/OCR spellings over ASR-derived ones

        Updates all segments in-place: replaces variant spellings with
        the canonical form so grounding and QA generation see consistent names.

        Returns dict with merge stats.
        """
        json_path = self.index_dir / f"{video_id}.json"
        if not json_path.exists():
            return {}

        with open(json_path) as f:
            doc = json.load(f)

        segments = [SegmentEntry.from_dict(s) for s in doc.get("segments", [])]

        # Collect all (text, type) pairs with source info
        entity_occurrences: Dict[str, Dict] = {}  # normalized_key → info
        for seg in segments:
            ocr_text = (seg.ocr_text or "").lower()
            for ent in seg.named_entities:
                text = ent.get("text", "").strip()
                etype = ent.get("type", "")
                if not text or len(text) <= 1:
                    continue
                key = text.upper().strip()
                if key not in entity_occurrences:
                    entity_occurrences[key] = {
                        "text": text,
                        "type": etype,
                        "count": 0,
                        "from_ocr": False,
                    }
                entity_occurrences[key]["count"] += 1
                # Check if this entity text appears in OCR (chyron authority)
                if text.lower() in ocr_text:
                    entity_occurrences[key]["from_ocr"] = True

        if not entity_occurrences:
            return {"merged": 0, "total": 0}

        # Build merge groups using substring containment
        keys = sorted(entity_occurrences.keys(), key=len, reverse=True)
        # Map each key to its canonical form
        canonical_map: Dict[str, str] = {}  # original_text → canonical_text
        merged_into: Dict[str, str] = {}    # key → canonical_key

        for i, key_a in enumerate(keys):
            if key_a in merged_into:
                continue
            words_a = set(key_a.split())
            for key_b in keys[i + 1:]:
                if key_b in merged_into:
                    continue
                words_b = set(key_b.split())
                # Check if shorter is a word-subset of longer
                if len(words_b) < 1 or len(min(words_b, key=len)) < 4:
                    continue
                if words_b.issubset(words_a):
                    merged_into[key_b] = key_a

        # Determine canonical text for each group
        # Prefer: OCR spelling > longest form > most frequent
        groups: Dict[str, list] = {}
        for key in keys:
            canonical_key = merged_into.get(key, key)
            if canonical_key not in groups:
                groups[canonical_key] = []
            groups[canonical_key].append(key)

        merge_count = 0
        for canonical_key, group_keys in groups.items():
            if len(group_keys) <= 1:
                continue
            # Pick the best spelling
            candidates = [entity_occurrences[k] for k in group_keys]
            # Prefer OCR (chyron), then longest, then most frequent
            best = max(candidates, key=lambda c: (
                c["from_ocr"],
                len(c["text"]),
                c["count"],
            ))
            canonical_text = best["text"]
            # Build mapping for all variants
            for k in group_keys:
                variant_text = entity_occurrences[k]["text"]
                if variant_text != canonical_text:
                    canonical_map[variant_text] = canonical_text
                    merge_count += 1

        if not canonical_map:
            return {"merged": 0, "total": len(entity_occurrences)}

        # Apply canonical mapping to all segments
        for seg in segments:
            updated_entities = []
            seen = set()
            for ent in seg.named_entities:
                text = ent.get("text", "").strip()
                etype = ent.get("type", "")
                # Replace with canonical form
                canonical = canonical_map.get(text, text)
                key = (canonical, etype)
                if key not in seen:
                    updated_entities.append({"text": canonical, "type": etype})
                    seen.add(key)
            seg.named_entities = updated_entities

            # Update grounded_entities keys
            if seg.grounded_entities:
                updated_grounding = {}
                for k, v in seg.grounded_entities.items():
                    canonical = canonical_map.get(k, k)
                    if canonical not in updated_grounding:
                        updated_grounding[canonical] = v
                seg.grounded_entities = updated_grounding

            seg.summary = seg.to_summary()

        # Save
        doc["segments"] = [s.to_dict() for s in segments]
        with open(json_path, "w") as f:
            json.dump(doc, f, indent=2)

        self._upsert_segments(video_id, segments)

        logger.info(
            f"Entity resolution for {video_id}: "
            f"{merge_count} variants merged, "
            f"{len(entity_occurrences)} → {len(entity_occurrences) - merge_count} unique"
        )
        return {"merged": merge_count, "total": len(entity_occurrences)}

    def get_indexed_videos(self) -> List[str]:
        """List all indexed video IDs."""
        return [p.stem for p in sorted(self.index_dir.glob("*.json"))]


# ============================================================================
# LangChain tool wrappers
# ============================================================================

def create_video_index_tools(index: VideoIndex):
    """Create LangChain @tool functions for the video index."""
    from langchain_core.tools import tool

    @tool("search_video_segments")
    def search_video_segments(query: str, top_k: int = 10) -> str:
        """Semantic search across all indexed video segments.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default 10).

        Returns:
            Matching segments with scores.
        """
        results = index.search(query, top_k=top_k)
        if not results:
            return "No matching segments found."
        lines = []
        for r in results:
            meta = r.get("metadata", {})
            dist = r.get("distance")
            dist_str = f" (dist={dist:.3f})" if dist is not None else ""
            lines.append(
                f"- [{meta.get('video_id', '?')} {meta.get('start_ms', 0)}-{meta.get('end_ms', 0)}ms]"
                f"{dist_str}: {r.get('summary', '')}"
            )
        return "\n".join(lines)

    @tool("browse_video_timeline")
    def browse_video_timeline(video_id: str, start_ms: int = 0, end_ms: int = -1) -> str:
        """Browse segments of a video within a time range.

        Args:
            video_id: The video identifier.
            start_ms: Start time in milliseconds (default 0).
            end_ms: End time in milliseconds (default -1 for entire video).

        Returns:
            Segments within the time range.
        """
        results = index.browse_timeline(video_id, start_ms, end_ms)
        if not results:
            return f"No segments found for {video_id} in range {start_ms}-{end_ms}ms."
        lines = []
        for seg in results:
            lines.append(
                f"- [{seg['start_ms']}-{seg['end_ms']}ms] {seg.get('scene_label', '')}:"
                f" {seg.get('summary', '')[:200]}"
            )
        return "\n".join(lines)

    @tool("filter_video_segments")
    def filter_video_segments(video_id: str, scene_label: str = "", has_text: str = "") -> str:
        """Filter segments by scene label or text presence.

        Args:
            video_id: The video identifier.
            scene_label: Filter by scene label (e.g., 'slate', 'chyron'). Empty for no filter.
            has_text: 'true' to find segments with text, 'false' for without, empty for no filter.

        Returns:
            Matching segments.
        """
        has_text_bool = None
        if has_text.lower() == "true":
            has_text_bool = True
        elif has_text.lower() == "false":
            has_text_bool = False

        results = index.filter_segments(
            video_id,
            scene_label=scene_label or None,
            has_text=has_text_bool,
        )
        if not results:
            return "No matching segments."
        lines = []
        for seg in results:
            lines.append(
                f"- [{seg['start_ms']}-{seg['end_ms']}ms] {seg.get('scene_label', '')}:"
                f" {seg.get('summary', '')[:200]}"
            )
        return "\n".join(lines)

    @tool("get_video_summary")
    def get_video_summary_tool(video_id: str) -> str:
        """Get an overview of an indexed video: segment counts, labels, text stats.

        Args:
            video_id: The video identifier.

        Returns:
            Summary statistics for the video.
        """
        summary = index.get_video_summary(video_id)
        if "error" in summary:
            return summary["error"]
        lines = [
            f"Video: {summary['video_id']}",
            f"Path: {summary['video_path']}",
            f"Duration: {summary['duration_ms']}ms",
            f"Indexed: {summary['indexed_at']}",
            f"Total segments: {summary['total_segments']}",
            f"Labels: {json.dumps(summary['label_counts'])}",
            f"OCR text: {summary['total_ocr_chars']} chars",
            f"ASR text: {summary['total_asr_chars']} chars",
            f"Entities: {summary['total_entities']} ({json.dumps(summary['entity_type_counts'])})",
        ]
        return "\n".join(lines)

    return [
        search_video_segments,
        browse_video_timeline,
        filter_video_segments,
        get_video_summary_tool,
    ]


# ============================================================================
# Singleton
# ============================================================================

# ============================================================================
# Wikipedia / Web Lookup for Entity Grounding
# ============================================================================

def wikipedia_lookup(entity: str, sentences: int = 2) -> Optional[str]:
    """Look up an entity on Wikipedia and return a short extract.

    Uses the MediaWiki API (free, no key required) to fetch the introductory
    extract for the best-matching article.

    Args:
        entity: Entity name to look up (e.g., "KHET", "Hawaii Public Television").
        sentences: Number of sentences to return (default 2).

    Returns:
        Short description string, or None if not found.
    """
    headers = {"User-Agent": "CLAMSAgent/1.0 (research prototype; kelleylynch)"}

    try:
        # Step 1: Search for the best matching article
        search_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": entity,
                "srlimit": 1,
                "format": "json",
            },
            headers=headers,
            timeout=10,
        )
        search_resp.raise_for_status()
        search_results = search_resp.json().get("query", {}).get("search", [])
        if not search_results:
            return None

        title = search_results[0]["title"]

        # Step 2: Get the plain-text extract
        extract_resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsentences": sentences,
                "titles": title,
                "format": "json",
            },
            headers=headers,
            timeout=10,
        )
        extract_resp.raise_for_status()
        pages = extract_resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract

    except Exception as e:
        logger.debug(f"Wikipedia lookup failed for '{entity}': {e}")

    return None


def web_search_entity(entity: str, max_results: int = 3) -> Optional[str]:
    """Search the web for an entity description using DuckDuckGo.

    Fallback when Wikipedia doesn't have a match (e.g., local TV stations,
    obscure people). Returns the top snippet.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{entity} who what is", max_results=max_results))
        if results:
            # Return the most informative snippet
            best = max(results, key=lambda r: len(r.get("body", "")))
            body = best.get("body", "").strip()
            if len(body) > 200:
                body = body[:200].rsplit(" ", 1)[0] + "..."
            return body
    except ImportError:
        logger.debug("duckduckgo-search not installed, web fallback unavailable")
    except Exception as e:
        logger.debug(f"Web search failed for '{entity}': {e}")
    return None


def _disambiguate_entity_temporal(
    entity_text: str, entity_type: str, broadcast_year: int
) -> Optional[str]:
    """Disambiguate an entity using broadcast year context.

    Queries Wikidata with temporal constraints to resolve:
    - "the president" → who was president in broadcast_year
    - Ambiguous person names → select candidate active in that era

    Returns improved description or None if no disambiguation needed/possible.
    """
    text_lower = entity_text.lower().strip()

    # --- Pattern 1: Title/role references → resolve via Wikidata SPARQL ---
    # Map common title phrases to Wikidata position IDs
    ROLE_POSITIONS = {
        "the president": "Q11696",          # President of the United States
        "president": "Q11696",
        "the vice president": "Q11699",     # VP of the United States
        "vice president": "Q11699",
        "the secretary of state": "Q248577",
        "secretary of state": "Q248577",
        "the pope": "Q19546",               # Pope
    }

    position_qid = None
    for phrase, qid in ROLE_POSITIONS.items():
        if text_lower == phrase or text_lower.startswith(phrase + " "):
            position_qid = qid
            break

    if position_qid:
        result = _wikidata_position_holder(position_qid, broadcast_year)
        if result:
            return f"{result} (holding office in {broadcast_year})"

    # --- Pattern 2: Person entities → re-query Wikidata with year context ---
    # Catch short/ambiguous names and single-word surnames
    if entity_type == "PERSON" and len(entity_text) > 2:
        try:
            resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbsearchentities",
                    "search": entity_text,
                    "language": "en",
                    "limit": "10",
                    "format": "json",
                },
                headers={"User-Agent": "CLAMS-Agent/1.0"},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            results = resp.json().get("search", [])
            # Filter to human entities (skip family names, given names, etc.)
            human_candidates = []
            for r in results:
                desc = (r.get("description", "") or "").lower()
                qid = r.get("id", "")
                # Skip meta-entities (names, disambiguation pages)
                if any(skip in desc for skip in (
                    "family name", "given name", "disambiguation",
                    "surname", "Wikimedia", "village", "township",
                )):
                    continue
                # Check if description suggests a person
                if any(kw in desc for kw in (
                    "politician", "president", "actor", "actress",
                    "journalist", "reporter", "singer", "writer",
                    "american", "british", "french", "born", "died",
                    "senator", "governor", "minister", "leader",
                    "athlete", "player", "coach", "director",
                    "anchor", "correspondent",
                )):
                    human_candidates.append(r)

            if not human_candidates:
                return None

            # If only one human candidate, use it
            if len(human_candidates) == 1:
                r = human_candidates[0]
                return f"{r.get('description', '')} [Wikidata:{r['id']}]"

            # Multiple candidates — pick the most prominent (first result that's human)
            r = human_candidates[0]
            return f"{r.get('description', '')} [Wikidata:{r['id']}]"

        except Exception as e:
            logger.debug(f"Wikidata disambiguation failed for '{entity_text}': {e}")

    return None


def _wikidata_position_holder(position_qid: str, year: int) -> Optional[str]:
    """Query Wikidata SPARQL for who held a position in a given year."""
    sparql_url = "https://query.wikidata.org/sparql"
    query = f"""
    SELECT ?person ?personLabel ?start ?end WHERE {{
      ?person p:P39 ?stmt .
      ?stmt ps:P39 wd:{position_qid} .
      ?person wdt:P31 wd:Q5 .
      OPTIONAL {{ ?stmt pq:P580 ?start . }}
      OPTIONAL {{ ?stmt pq:P582 ?end . }}
      FILTER(
        BOUND(?start) &&
        YEAR(?start) <= {year} &&
        (!BOUND(?end) || YEAR(?end) >= {year})
      )
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" . }}
    }}
    ORDER BY DESC(?start)
    LIMIT 3
    """
    try:
        resp = requests.get(
            sparql_url,
            params={"query": query, "format": "json"},
            headers={"User-Agent": "CLAMS-Agent/1.0"},
            timeout=15,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", {}).get("bindings", [])
            if results:
                person = results[0].get("personLabel", {}).get("value", "")
                if person:
                    return person
    except Exception as e:
        logger.debug(f"Wikidata SPARQL failed for position {position_qid}: {e}")

    return None


def wikidata_lookup(entity: str) -> Optional[str]:
    """Look up structured facts about an entity from Wikidata.

    Uses the Wikidata API to search for an entity, then fetches key properties
    (description, positions held, occupation, member of, etc.) and resolves
    QID references to human-readable labels.

    Returns a compact structured summary like:
        "American politician and diplomat (born 1947). Positions: United States
         Secretary of State, United States senator. Party: Democratic Party."
    """
    headers = {"User-Agent": "CLAMSAgent/1.0 (research prototype)"}

    # Key Wikidata properties to extract
    PROPERTIES = {
        "P39": "position held",
        "P106": "occupation",
        "P27": "country of citizenship",
        "P102": "political party",
        "P108": "employer",
        "P69": "educated at",
        "P463": "member of",
        "P17": "country",
        "P131": "located in",
        "P159": "headquarters",
        "P112": "founded by",
        "P571": "inception",
        "P452": "industry",
    }

    try:
        # Step 1: Search for entity QID
        search_resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbsearchentities",
                "search": entity,
                "language": "en",
                "limit": 1,
                "format": "json",
            },
            headers=headers,
            timeout=10,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("search", [])
        if not results:
            return None

        qid = results[0]["id"]

        # Step 2: Fetch entity data
        entity_resp = requests.get(
            "https://www.wikidata.org/w/api.php",
            params={
                "action": "wbgetentities",
                "ids": qid,
                "format": "json",
                "languages": "en",
                "props": "labels|descriptions|claims",
            },
            headers=headers,
            timeout=10,
        )
        entity_resp.raise_for_status()
        ent_data = entity_resp.json().get("entities", {}).get(qid, {})

        desc = ent_data.get("descriptions", {}).get("en", {}).get("value", "")
        claims = ent_data.get("claims", {})

        # Step 3: Extract QIDs from relevant properties
        qids_to_resolve: set = set()
        prop_qids: Dict[str, List[str]] = {}

        for prop_id, prop_label in PROPERTIES.items():
            if prop_id not in claims:
                continue
            qid_list = []
            for claim in claims[prop_id][:5]:  # limit per property
                snak = claim.get("mainsnak", {})
                dv = snak.get("datavalue", {})
                if dv.get("type") == "wikibase-entityid":
                    ref_qid = dv["value"]["id"]
                    qid_list.append(ref_qid)
                    qids_to_resolve.add(ref_qid)
                elif dv.get("type") == "time":
                    # Date value like inception
                    time_val = dv["value"].get("time", "")
                    if time_val.startswith("+"):
                        time_val = time_val[1:]
                    year = time_val.split("-")[0] if "-" in time_val else time_val[:4]
                    qid_list.append(year)
                elif dv.get("type") == "string":
                    qid_list.append(dv["value"])
            if qid_list:
                prop_qids[prop_label] = qid_list

        # Step 4: Batch-resolve QIDs to labels
        labels: Dict[str, str] = {}
        if qids_to_resolve:
            # API allows up to 50 IDs per request
            qid_batch = list(qids_to_resolve)[:50]
            label_resp = requests.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(qid_batch),
                    "format": "json",
                    "languages": "en",
                    "props": "labels",
                },
                headers=headers,
                timeout=10,
            )
            label_resp.raise_for_status()
            for rid, rent in label_resp.json().get("entities", {}).items():
                labels[rid] = rent.get("labels", {}).get("en", {}).get("value", rid)

        # Step 5: Build summary
        parts = []
        if desc:
            parts.append(desc.rstrip(".") + ".")

        for prop_label, qid_list in prop_qids.items():
            resolved = [labels.get(q, q) for q in qid_list]
            parts.append(f"{prop_label.capitalize()}: {', '.join(resolved)}.")

        return " ".join(parts) if parts else None

    except Exception as e:
        logger.debug(f"Wikidata lookup failed for '{entity}': {e}")
        return None


def pubmed_search(query: str, max_results: int = 3) -> Optional[str]:
    """Search PubMed for biomedical/health literature.

    Uses the LangChain PubMed wrapper if available, otherwise falls back
    to direct NCBI E-utilities API.

    Returns formatted search results with titles and snippets.
    """
    # Try LangChain wrapper first
    try:
        from langchain_community.utilities.pubmed import PubMedAPIWrapper
        wrapper = PubMedAPIWrapper(top_k_results=max_results)
        result = wrapper.run(query)
        if result and result != "No good PubMed Result was found":
            return result
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"LangChain PubMed failed: {e}")

    # Direct NCBI fallback
    try:
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
            },
            headers={"User-Agent": "CLAMSAgent/1.0"},
            timeout=10,
        )
        search_resp.raise_for_status()
        ids = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None

        # Fetch summaries
        summary_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
            headers={"User-Agent": "CLAMSAgent/1.0"},
            timeout=10,
        )
        summary_resp.raise_for_status()
        result_data = summary_resp.json().get("result", {})

        lines = []
        for pmid in ids:
            article = result_data.get(pmid, {})
            title = article.get("title", "")
            source = article.get("source", "")
            pubdate = article.get("pubdate", "")
            if title:
                lines.append(f"- {title} ({source}, {pubdate}) [PMID:{pmid}]")

        return "\n".join(lines) if lines else None

    except Exception as e:
        logger.debug(f"PubMed search failed for '{query}': {e}")
        return None


_video_index: Optional[VideoIndex] = None


def get_video_index() -> VideoIndex:
    """Get or create the VideoIndex singleton, configured from config.json."""
    global _video_index
    if _video_index is None:
        # Read index config from config.json
        config_path = Path(__file__).parent.parent / "config.json"
        index_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    index_config = json.load(f).get("index", {})
            except Exception:
                pass

        _video_index = VideoIndex(
            index_dir=index_config.get("index_dir", "data/video_indexes"),
            chroma_dir=index_config.get("chroma_dir", "data/chroma_db"),
            embedding_model=index_config.get("embedding_model", "nomic-embed-text"),
            fallback_segment_duration_ms=index_config.get("fallback_segment_duration_ms", 10000),
        )
    return _video_index
