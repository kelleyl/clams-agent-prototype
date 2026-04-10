"""
Generate QA evaluation pairs from video index JSON files.

Reads structured video indexes (produced by the CLAMS agent pipeline) and
generates question-answer pairs across 6 categories:
  1. scene_classification — scene type at timestamp, existence, count, boundary
  2. text_extraction — OCR and ASR transcript questions
  3. named_entity — entity listing, type filtering, aggregation
  4. entity_linking — Wikidata descriptions, disambiguation
  5. cross_annotation — multi-field questions (speaker+scene, entity+OCR)
  6. temporal_reasoning — ordering, before/after, duration

Supports both free-form and multiple-choice question formats.

Outputs JSONL compatible with qa_schema.json.

Inspired by:
  - Neptune (semi-auto QA from structured captions, LLM generation + human verify)
  - CinePile (adversarial "deaf-blind" filtering)
  - TVBench (avoid single-frame / world-knowledge solvable questions)

Usage:
    python generate_index_qa.py                          # All indexes, free-form
    python generate_index_qa.py --mc                     # Multiple choice
    python generate_index_qa.py --video cpb-aacip-225-009w0w1j  # Single video
    python generate_index_qa.py --stats                  # Print stats only
"""

import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Generator

# Add parent dir so we can import from converters
sys.path.insert(0, str(Path(__file__).parent / "converters"))
from base_converter import QAEntry, QUESTION_TEMPLATES, select_question_template


# Scene labels found in video indexes (from swt-detection)
SCENE_LABELS_ALL = {
    "bars", "slate", "other text", "person & chyron",
    "person & extra text", "other chyron", "filmed text",
    "credits", "person", "image",
}

# Map NER types to schema entity types
NER_TYPE_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "EVENT": "event",
    "DATE": "other",
    "TIME": "other",
    "NORP": "other",
    "WORK_OF_ART": "other",
    "PRODUCT": "other",
    "MONEY": "other",
    "QUANTITY": "other",
    "ORDINAL": "other",
    "CARDINAL": "other",
}

# Entity types worth asking about (skip DATE, TIME, CARDINAL etc.)
INTERESTING_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT"}


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm format."""
    total_seconds = ms / 1000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def ms_to_human(ms: int) -> str:
    """Convert milliseconds to human-readable like '2m30s'."""
    total_seconds = ms / 1000
    if total_seconds < 60:
        return f"{total_seconds:.0f}s"
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    if seconds == 0:
        return f"{minutes}m"
    return f"{minutes}m{seconds}s"


def is_substantive_transcript(text: str) -> bool:
    """Check if transcript is substantive (not just filler like 'Thank you.')."""
    if not text or not text.strip():
        return False
    stripped = text.strip().rstrip(".")
    filler = {"thank you", "thanks", "thank you thank you", "thank you thank you thank you"}
    return stripped.lower() not in filler and len(stripped.split()) >= 5


def pick_distractors(correct: str, pool: set | list, n: int = 3) -> list[str]:
    """Pick n distractor options from pool, excluding the correct answer."""
    candidates = [x for x in pool if x != correct]
    random.shuffle(candidates)
    return candidates[:n]


def shuffle_choices(correct: str, distractors: list[str]) -> tuple[list[str], int]:
    """Shuffle correct answer into distractors, return (choices, correct_index)."""
    choices = [correct] + distractors
    random.shuffle(choices)
    return choices, choices.index(correct)


CHOICE_LETTERS = "ABCDEFGH"


def format_mc(choices: list[str], correct_idx: int) -> dict:
    """Format multiple choice answer dict."""
    return {
        "choices": {CHOICE_LETTERS[i]: c for i, c in enumerate(choices)},
        "correct": CHOICE_LETTERS[correct_idx],
        "correct_text": choices[correct_idx],
    }


class IndexQAGenerator:
    """Generate QA pairs from a video index JSON file."""

    SOURCE_PROJECT = "derived"

    def __init__(self, index_dir: str | Path, multiple_choice: bool = False):
        self.index_dir = Path(index_dir)
        self.mc = multiple_choice
        self._id_counter = 0

    def generate_id(self, category: str) -> str:
        """Generate a unique QA entry ID."""
        self._id_counter += 1
        prefix = category.replace("_", "-")
        return f"qa-{prefix}-idx-{self._id_counter:06d}"

    def load_index(self, video_id: str) -> dict:
        """Load a video index JSON file."""
        path = self.index_dir / f"{video_id}.json"
        with open(path) as f:
            return json.load(f)

    def list_indexed_videos(self) -> list[str]:
        """List all indexed video IDs."""
        return [p.stem for p in sorted(self.index_dir.glob("*.json"))]

    def generate_all(self, video_id: str | None = None) -> list[QAEntry]:
        """Generate QA pairs for one or all indexed videos."""
        videos = [video_id] if video_id else self.list_indexed_videos()
        entries = []
        for vid in videos:
            index = self.load_index(vid)
            entries.extend(self._generate_for_video(index))
        return entries

    def _generate_for_video(self, index: dict) -> list[QAEntry]:
        """Generate all QA categories for a single video."""
        entries = []
        segments = index.get("segments", [])
        if not segments:
            return entries

        generators = [
            self._generate_scene_classification,
            self._generate_text_extraction,
            self._generate_named_entity,
            self._generate_entity_linking,
            self._generate_cross_annotation,
            self._generate_temporal_reasoning,
            self._generate_visual_description,
            self._generate_shot_level,
            self._generate_relation_qa,
            self._generate_multi_hop_relation,
            self._generate_speaker_qa,
        ]
        for gen in generators:
            entries.extend(gen(index, segments))

        return entries

    def _make_entry(
        self,
        category: str,
        question: str,
        answer: dict,
        video_id: str,
        difficulty: str = "medium",
        tags: list[str] | None = None,
        context: dict | None = None,
    ) -> QAEntry:
        """Helper to create a QAEntry with standard fields."""
        return QAEntry(
            id=self.generate_id(category),
            category=category,
            question=question,
            answer=answer,
            source_project=self.SOURCE_PROJECT,
            source_guid=video_id,
            context=context or {"video_guid": video_id},
            metadata={
                "created_from": "video_index",
                "conversion_date": date.today().isoformat(),
                "generator": "generate_index_qa",
            },
            difficulty=difficulty,
            tags=tags or [],
        )

    # ─── Category 1: Scene Classification ──────────────────────────

    def _generate_scene_classification(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        entries = []
        video_id = index["video_id"]
        by_label = defaultdict(list)
        all_labels_in_video = set()

        for seg in segments:
            label = seg.get("scene_label", "")
            if label:
                by_label[label].append(seg)
                all_labels_in_video.add(label)

        label_pool = SCENE_LABELS_ALL | all_labels_in_video

        # Q: What type of scene at timestamp? (sample every 3rd segment)
        for i, seg in enumerate(segments):
            if i % 3 != 0:
                continue
            label = seg.get("scene_label", "")
            if not label:
                continue
            ts = ms_to_timestamp(seg["start_ms"])

            answer = {
                "scene_label": label,
                "confidence": f"{seg.get('classification', {}).get(label, 0):.2f}",
                "timestamps": [ts],
            }
            if self.mc:
                distractors = pick_distractors(label, label_pool)
                choices, idx = shuffle_choices(label, distractors)
                answer["mc"] = format_mc(choices, idx)

            entries.append(self._make_entry(
                category="scene_classification",
                question=select_question_template("scene_at_time", index=i, timestamp=ts),
                answer=answer,
                video_id=video_id,
                difficulty="easy",
                tags=["scene-at-time", label, "requires_video"] + (["multiple_choice"] if self.mc else []),
                context={
                    "video_guid": video_id,
                    "timestamp": ts,
                    "time_range": {
                        "start": ms_to_timestamp(seg["start_ms"]),
                        "end": ms_to_timestamp(seg["end_ms"]),
                    },
                },
            ))

        # Q: Does this video contain {scene_type}?
        for label, segs in by_label.items():
            sample_ts = [ms_to_timestamp(s["start_ms"]) for s in segs[:5]]
            answer = {"exists": True, "count": len(segs), "scene_label": label, "timestamps": sample_ts}
            if self.mc:
                choices, idx = shuffle_choices("Yes", ["No"])
                answer["mc"] = format_mc(choices, idx)
            entries.append(self._make_entry(
                category="scene_classification",
                question=f"Does this video contain any {label} scenes?",
                answer=answer,
                video_id=video_id,
                difficulty="easy",
                tags=["existence", label, "requires_video"] + (["multiple_choice"] if self.mc else []),
            ))

        # Q: How many {scene_type} scenes?
        for label, segs in by_label.items():
            count = len(segs)
            answer = {"count": count, "scene_label": label}
            if self.mc:
                # Distractors: nearby counts + 0
                distractor_counts = set()
                for d in [max(0, count - 2), max(0, count - 1), count + 1, count + 2, 0]:
                    if d != count:
                        distractor_counts.add(d)
                distractors = [str(d) for d in sorted(distractor_counts)][:3]
                choices, idx = shuffle_choices(str(count), distractors)
                answer["mc"] = format_mc(choices, idx)
            entries.append(self._make_entry(
                category="scene_classification",
                question=f"How many {label} scenes are in this video?",
                answer=answer,
                video_id=video_id,
                difficulty="medium",
                tags=["count", label, "requires_video"] + (["multiple_choice"] if self.mc else []),
            ))

        # Negative existence for missing labels
        found = set(by_label.keys())
        for missing in SCENE_LABELS_ALL - found:
            answer = {"exists": False, "count": 0, "scene_label": missing}
            if self.mc:
                choices, idx = shuffle_choices("No", ["Yes"])
                answer["mc"] = format_mc(choices, idx)
            entries.append(self._make_entry(
                category="scene_classification",
                question=f"Are there {missing} frames in this video?",
                answer=answer,
                video_id=video_id,
                difficulty="easy",
                tags=["existence", "negative", missing, "requires_video"] + (["multiple_choice"] if self.mc else []),
            ))

        # Q: What is the first/last scene type?
        if segments:
            first = segments[0]
            last = segments[-1]
            for which, seg in [("first", first), ("last", last)]:
                label = seg.get("scene_label", "")
                answer = {
                    "scene_label": label,
                    "timestamps": [ms_to_timestamp(seg["start_ms"])],
                }
                if self.mc:
                    distractors = pick_distractors(label, label_pool)
                    choices, idx = shuffle_choices(label, distractors)
                    answer["mc"] = format_mc(choices, idx)
                entries.append(self._make_entry(
                    category="scene_classification",
                    question=f"What type of scene appears {which} in this video?",
                    answer=answer,
                    video_id=video_id,
                    difficulty="easy",
                    tags=["boundary", which, "requires_video"] + (["multiple_choice"] if self.mc else []),
                ))

        return entries

    # ─── Category 2: Text Extraction ──────────────────────────────

    def _generate_text_extraction(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        entries = []
        video_id = index["video_id"]

        # ASR transcript questions
        segments_with_asr = [s for s in segments if is_substantive_transcript(s.get("asr_transcript", ""))]

        # Q: What is said between start and end? (sample)
        for i, seg in enumerate(segments_with_asr):
            if i % 2 != 0:
                continue
            transcript = seg["asr_transcript"].strip()
            start_ts = ms_to_timestamp(seg["start_ms"])
            end_ts = ms_to_timestamp(seg["end_ms"])
            entries.append(self._make_entry(
                category="text_extraction",
                question=f"What is said between {start_ts} and {end_ts}?",
                answer={"text": transcript},
                video_id=video_id,
                difficulty="medium",
                tags=["asr", "transcript", "requires_audio"],
                context={
                    "video_guid": video_id,
                    "time_range": {"start": start_ts, "end": end_ts},
                    "transcript_snippet": transcript[:200],
                },
            ))

        # Q: What text appears on screen at timestamp? (OCR)
        segments_with_ocr = [s for s in segments if s.get("ocr_text", "").strip()]
        for seg in segments_with_ocr:
            ts = ms_to_timestamp(seg["start_ms"])
            entries.append(self._make_entry(
                category="text_extraction",
                question=select_question_template("text_at_time", index=0, timestamp=ts),
                answer={"text": seg["ocr_text"].strip()},
                video_id=video_id,
                difficulty="medium",
                tags=["ocr", "requires_video"],
                context={"video_guid": video_id, "timestamp": ts},
            ))

        # Q: Does this video contain any spoken dialogue?
        if segments_with_asr:
            entries.append(self._make_entry(
                category="text_extraction",
                question="Does this video contain spoken dialogue?",
                answer={
                    "value": True,
                    "explanation": f"Yes, {len(segments_with_asr)} segments contain substantive speech.",
                },
                video_id=video_id,
                difficulty="easy",
                tags=["existence", "asr", "requires_audio"],
            ))

        # Q: Summarize what is discussed in this segment (for rich transcripts)
        rich_segments = [s for s in segments_with_asr if len(s["asr_transcript"].split()) >= 30]
        for seg in rich_segments[:5]:  # cap at 5
            start_ts = ms_to_timestamp(seg["start_ms"])
            end_ts = ms_to_timestamp(seg["end_ms"])
            entries.append(self._make_entry(
                category="text_extraction",
                question=f"Summarize what is discussed between {start_ts} and {end_ts}.",
                answer={"text": seg["asr_transcript"].strip()},
                video_id=video_id,
                difficulty="hard",
                tags=["summarization", "asr", "requires_audio", "open_ended"],
                context={
                    "video_guid": video_id,
                    "time_range": {"start": start_ts, "end": end_ts},
                },
            ))

        return entries

    # ─── Category 3: Named Entity ─────────────────────────────────

    def _generate_named_entity(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        entries = []
        video_id = index["video_id"]

        # Collect all entities across segments
        all_entities = []  # (entity_dict, segment)
        entities_by_type = defaultdict(set)  # type -> set of entity texts

        for seg in segments:
            for ent in seg.get("named_entities", []):
                ent_type = ent.get("type", "")
                ent_text = ent.get("text", "")
                if ent_type in INTERESTING_ENTITY_TYPES and ent_text:
                    all_entities.append((ent, seg))
                    entities_by_type[ent_type].add(ent_text)

        if not all_entities:
            return entries

        # Q: What people are mentioned in this video?
        if entities_by_type.get("PERSON"):
            people = sorted(entities_by_type["PERSON"])
            entries.append(self._make_entry(
                category="named_entity",
                question="What people are mentioned in this video?",
                answer={"people": people, "entities": [{"text": p, "type": "person"} for p in people]},
                video_id=video_id,
                difficulty="medium",
                tags=["person", "aggregation", "requires_audio"],
            ))

        # Q: What organizations are mentioned?
        if entities_by_type.get("ORG"):
            orgs = sorted(entities_by_type["ORG"])
            entries.append(self._make_entry(
                category="named_entity",
                question="What organizations are mentioned in this video?",
                answer={"organizations": orgs, "entities": [{"text": o, "type": "organization"} for o in orgs]},
                video_id=video_id,
                difficulty="medium",
                tags=["organization", "aggregation", "requires_audio"],
            ))

        # Q: What locations are mentioned?
        location_types = {"GPE", "LOC", "FAC"}
        locations = set()
        for t in location_types:
            locations.update(entities_by_type.get(t, set()))
        if locations:
            locs = sorted(locations)
            entries.append(self._make_entry(
                category="named_entity",
                question="What locations are mentioned in this video?",
                answer={"locations": locs, "entities": [{"text": l, "type": "location"} for l in locs]},
                video_id=video_id,
                difficulty="medium",
                tags=["location", "aggregation", "requires_audio"],
            ))

        # Q: What named entities appear in a specific segment?
        segments_with_entities = [(seg, [e for e, s in all_entities if s is seg])
                                  for seg in segments
                                  if any(s is seg for _, s in all_entities)]
        for seg, ents in segments_with_entities[:5]:  # cap at 5
            start_ts = ms_to_timestamp(seg["start_ms"])
            end_ts = ms_to_timestamp(seg["end_ms"])
            entity_list = [{"text": e["text"], "type": NER_TYPE_MAP.get(e["type"], "other")}
                           for e in ents if e["type"] in INTERESTING_ENTITY_TYPES]
            if entity_list:
                entries.append(self._make_entry(
                    category="named_entity",
                    question=f"What named entities are mentioned between {start_ts} and {end_ts}?",
                    answer={"entities": entity_list},
                    video_id=video_id,
                    difficulty="medium",
                    tags=["segment-level", "requires_audio"],
                    context={
                        "video_guid": video_id,
                        "time_range": {"start": start_ts, "end": end_ts},
                    },
                ))

        # Q: Is {person} mentioned in this video? (positive + negative)
        if entities_by_type.get("PERSON"):
            people = list(entities_by_type["PERSON"])
            # Positive: pick up to 3
            for person in people[:3]:
                answer: dict[str, Any] = {"value": True, "explanation": f"Yes, {person} is mentioned."}
                if self.mc:
                    choices, idx = shuffle_choices("Yes", ["No"])
                    answer["mc"] = format_mc(choices, idx)
                entries.append(self._make_entry(
                    category="named_entity",
                    question=f"Is {person} mentioned in this video?",
                    answer=answer,
                    video_id=video_id,
                    difficulty="easy",
                    tags=["existence", "person", "requires_audio"] + (["multiple_choice"] if self.mc else []),
                ))
            # Negative: a clearly fictional name
            neg_answer: dict[str, Any] = {"value": False, "explanation": "No, this person is not mentioned."}
            if self.mc:
                choices, idx = shuffle_choices("No", ["Yes"])
                neg_answer["mc"] = format_mc(choices, idx)
            entries.append(self._make_entry(
                category="named_entity",
                question="Is Dr. Zachary Pemberton mentioned in this video?",
                answer=neg_answer,
                video_id=video_id,
                difficulty="easy",
                tags=["existence", "negative", "person", "requires_audio"] + (["multiple_choice"] if self.mc else []),
            ))

        # Q: Which of these people is mentioned in this video? (MC only)
        if self.mc and entities_by_type.get("PERSON"):
            people = list(entities_by_type["PERSON"])
            if len(people) >= 1:
                correct_person = random.choice(people)
                fake_names = ["Dr. Zachary Pemberton", "Maria Gonzalez-Chen",
                              "Robert Blackwell III", "Sarah Nightingale"]
                distractors = random.sample(fake_names, min(3, len(fake_names)))
                choices, idx = shuffle_choices(correct_person, distractors)
                entries.append(self._make_entry(
                    category="named_entity",
                    question="Which of the following people is mentioned in this video?",
                    answer={
                        "value": correct_person,
                        "mc": format_mc(choices, idx),
                    },
                    video_id=video_id,
                    difficulty="medium",
                    tags=["identification", "person", "requires_audio", "multiple_choice"],
                ))

        return entries

    # ─── Category 4: Entity Linking ───────────────────────────────

    def _generate_entity_linking(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        entries = []
        video_id = index["video_id"]

        # Collect grounded entities (those with Wikidata descriptions)
        grounded = {}  # entity_text -> description
        for seg in segments:
            for ent_text, desc in seg.get("grounded_entities", {}).items():
                if desc and ent_text not in grounded:
                    grounded[ent_text] = desc

        if not grounded:
            return entries

        # Filter to entities with reasonable groundings (not date/time artifacts)
        # Check if the entity appears in named_entities with an interesting type
        interesting_entities = set()
        for seg in segments:
            for ent in seg.get("named_entities", []):
                if ent.get("type") in INTERESTING_ENTITY_TYPES:
                    interesting_entities.add(ent["text"])

        for ent_text, desc in grounded.items():
            if ent_text not in interesting_entities:
                continue

            # Q: What Wikipedia article describes '{entity}'?
            entries.append(self._make_entry(
                category="entity_linking",
                question=f"What do you know about '{ent_text}' as mentioned in this video?",
                answer={
                    "entities": [{"text": ent_text, "type": "other"}],
                    "value": desc[:300],  # truncate long descriptions
                },
                video_id=video_id,
                difficulty="hard",
                tags=["grounding", "wikidata", "requires_audio"],
            ))

        # Q: Which entities in this video have been linked to external knowledge?
        linked_names = sorted(set(grounded.keys()) & interesting_entities)
        if linked_names:
            entries.append(self._make_entry(
                category="entity_linking",
                question="Which entities in this video have been linked to external knowledge bases?",
                answer={
                    "entities": [{"text": n, "type": "other"} for n in linked_names],
                    "value": f"{len(linked_names)} entities linked: {', '.join(linked_names[:10])}",
                },
                video_id=video_id,
                difficulty="hard",
                tags=["aggregation", "grounding"],
            ))

        return entries

    # ─── Category 5: Cross-Annotation ─────────────────────────────

    def _generate_cross_annotation(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Questions requiring multiple annotation layers."""
        entries = []
        video_id = index["video_id"]

        # Q: Who is speaking during {scene_type} segments?
        # Requires: scene_label + named_entities (PERSON) in same segment
        for seg in segments:
            label = seg.get("scene_label", "")
            if "chyron" not in label and "person" not in label:
                continue
            people = [e["text"] for e in seg.get("named_entities", []) if e.get("type") == "PERSON"]
            transcript = seg.get("asr_transcript", "")
            if not people or not is_substantive_transcript(transcript):
                continue

            start_ts = ms_to_timestamp(seg["start_ms"])
            end_ts = ms_to_timestamp(seg["end_ms"])
            entries.append(self._make_entry(
                category="cross_annotation",
                question=f"Who is speaking during the {label} segment at {start_ts}?",
                answer={
                    "entity": ", ".join(people),
                    "matches": [{
                        "guid": video_id,
                        "timestamp": start_ts,
                        "chyron_text": seg.get("ocr_text", ""),
                        "transcript_snippet": transcript[:200],
                        "entity_match": {"text": people[0], "type": "person"},
                    }],
                    "source_annotations": ["scene_classification", "named_entity", "text_extraction"],
                },
                video_id=video_id,
                difficulty="hard",
                tags=["cross-layer", "speaker-identification", "requires_audio", "requires_video"],
                context={
                    "video_guid": video_id,
                    "time_range": {"start": start_ts, "end": end_ts},
                },
            ))

        # Q: What topics are discussed in {scene_type} segments?
        # Combine scene labels with entity aggregation
        by_label = defaultdict(list)
        for seg in segments:
            label = seg.get("scene_label", "")
            if label:
                by_label[label].append(seg)

        for label, segs in by_label.items():
            all_people = set()
            all_orgs = set()
            all_locations = set()
            for seg in segs:
                for ent in seg.get("named_entities", []):
                    if ent["type"] == "PERSON":
                        all_people.add(ent["text"])
                    elif ent["type"] == "ORG":
                        all_orgs.add(ent["text"])
                    elif ent["type"] in ("GPE", "LOC"):
                        all_locations.add(ent["text"])

            total = len(all_people) + len(all_orgs) + len(all_locations)
            if total < 2:
                continue

            entries.append(self._make_entry(
                category="cross_annotation",
                question=f"What people, organizations, and places are discussed during {label} segments?",
                answer={
                    "entity": f"{label} segment entities",
                    "total_count": total,
                    "matches": [
                        *[{"entity_match": {"text": p, "type": "person"}} for p in sorted(all_people)],
                        *[{"entity_match": {"text": o, "type": "organization"}} for o in sorted(all_orgs)],
                        *[{"entity_match": {"text": l, "type": "location"}} for l in sorted(all_locations)],
                    ],
                    "source_annotations": ["scene_classification", "named_entity"],
                },
                video_id=video_id,
                difficulty="hard",
                tags=["cross-layer", "scene-entity", "requires_audio", "requires_video"],
            ))

        return entries

    # ─── Category 6: Temporal Reasoning ───────────────────────────

    def _generate_temporal_reasoning(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Questions about ordering, duration, and temporal relationships."""
        entries = []
        video_id = index["video_id"]

        if len(segments) < 2:
            return entries

        # Collect all labels for distractor pool
        all_labels_in_video = {s.get("scene_label", "") for s in segments} - {""}
        label_pool = SCENE_LABELS_ALL | all_labels_in_video

        # Q: What scene type comes after {label}?
        for i in range(len(segments) - 1):
            curr = segments[i]
            nxt = segments[i + 1]
            curr_label = curr.get("scene_label", "")
            nxt_label = nxt.get("scene_label", "")
            if not curr_label or not nxt_label:
                continue
            if i % 3 != 0:
                continue  # sample
            ts = ms_to_timestamp(curr["start_ms"])
            answer = {
                "value": nxt_label,
                "explanation": f"The {curr_label} at {ts} is followed by {nxt_label} at {ms_to_timestamp(nxt['start_ms'])}.",
            }
            if self.mc:
                distractors = pick_distractors(nxt_label, label_pool)
                choices, idx = shuffle_choices(nxt_label, distractors)
                answer["mc"] = format_mc(choices, idx)
            entries.append(self._make_entry(
                category="cross_annotation",  # temporal fits cross_annotation in schema
                question=f"What type of scene follows the {curr_label} segment at {ts}?",
                answer=answer,
                video_id=video_id,
                difficulty="medium",
                tags=["temporal", "ordering", "requires_video"] + (["multiple_choice"] if self.mc else []),
                context={
                    "video_guid": video_id,
                    "timestamp": ts,
                },
            ))

        # Q: Which is the longest segment? Duration of video?
        if segments:
            longest = max(segments, key=lambda s: s["end_ms"] - s["start_ms"])
            dur = longest["end_ms"] - longest["start_ms"]
            entries.append(self._make_entry(
                category="cross_annotation",
                question="What is the longest segment in this video and how long is it?",
                answer={
                    "value": f"{longest.get('scene_label', 'unknown')} ({ms_to_human(dur)})",
                    "explanation": (
                        f"The longest segment is a {longest.get('scene_label', '')} segment "
                        f"from {ms_to_timestamp(longest['start_ms'])} to {ms_to_timestamp(longest['end_ms'])} "
                        f"({ms_to_human(dur)})."
                    ),
                },
                video_id=video_id,
                difficulty="medium",
                tags=["temporal", "duration", "requires_video"],
            ))

        # Q: Total video duration
        duration_ms = index.get("duration_ms", 0)
        if duration_ms:
            entries.append(self._make_entry(
                category="cross_annotation",
                question="How long is this video?",
                answer={
                    "value": ms_to_human(duration_ms),
                    "explanation": f"The video is {ms_to_human(duration_ms)} ({ms_to_timestamp(duration_ms)}).",
                },
                video_id=video_id,
                difficulty="easy",
                tags=["temporal", "duration"],
            ))

        # Q: Does {person} appear before or after {other_person}?
        people_first_seen = {}
        for seg in segments:
            for ent in seg.get("named_entities", []):
                if ent["type"] == "PERSON" and ent["text"] not in people_first_seen:
                    people_first_seen[ent["text"]] = seg["start_ms"]

        people_list = list(people_first_seen.items())
        if len(people_list) >= 2:
            # Pick first and last person
            people_list.sort(key=lambda x: x[1])
            first_person, first_ms = people_list[0]
            last_person, last_ms = people_list[-1]
            if first_person != last_person:
                answer = {
                    "value": "before",
                    "explanation": (
                        f"{first_person} first appears at {ms_to_timestamp(first_ms)}, "
                        f"while {last_person} first appears at {ms_to_timestamp(last_ms)}."
                    ),
                }
                if self.mc:
                    choices, idx = shuffle_choices("Before", ["After", "At the same time", "Neither is mentioned"])
                    answer["mc"] = format_mc(choices, idx)
                entries.append(self._make_entry(
                    category="cross_annotation",
                    question=f"Is {first_person} mentioned before or after {last_person}?",
                    answer=answer,
                    video_id=video_id,
                    difficulty="medium",
                    tags=["temporal", "ordering", "person", "requires_audio"] + (["multiple_choice"] if self.mc else []),
                ))

        return entries


    # ─── Category 7: Visual Description ─────────────────────────

    def _generate_visual_description(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Questions answerable from visual captions alone (no audio needed)."""
        entries = []
        video_id = index["video_id"]

        segments_with_caption = [s for s in segments if s.get("visual_caption", "").strip()]
        if not segments_with_caption:
            return entries

        # Q: What is visually depicted at timestamp?
        for i, seg in enumerate(segments_with_caption):
            if i % 2 != 0:
                continue  # sample every other
            caption = seg["visual_caption"].strip()
            ts = ms_to_timestamp(seg["start_ms"])
            entries.append(self._make_entry(
                category="scene_classification",
                question=f"Describe what is visually shown at {ts}.",
                answer={"text": caption},
                video_id=video_id,
                difficulty="medium",
                tags=["visual-description", "caption", "requires_video", "visual_only"],
                context={
                    "video_guid": video_id,
                    "timestamp": ts,
                    "time_range": {
                        "start": ms_to_timestamp(seg["start_ms"]),
                        "end": ms_to_timestamp(seg["end_ms"]),
                    },
                },
            ))

        # Q: What type of visual content dominates this video?
        label_counts = defaultdict(int)
        for seg in segments:
            label = seg.get("scene_label", "")
            if label:
                label_counts[label] += 1
        if label_counts:
            dominant = max(label_counts.items(), key=lambda x: x[1])
            entries.append(self._make_entry(
                category="scene_classification",
                question="What type of visual content appears most frequently in this video?",
                answer={
                    "scene_label": dominant[0],
                    "count": dominant[1],
                    "explanation": f"'{dominant[0]}' appears {dominant[1]} times out of {len(segments)} segments.",
                },
                video_id=video_id,
                difficulty="medium",
                tags=["visual-description", "aggregation", "requires_video", "visual_only"],
            ))

        # Q: Does the video contain any on-screen text?
        ocr_segments = [s for s in segments if s.get("ocr_text", "").strip()]
        entries.append(self._make_entry(
            category="text_extraction",
            question="Does this video contain any visible on-screen text?",
            answer={
                "value": len(ocr_segments) > 0,
                "explanation": f"{'Yes' if ocr_segments else 'No'}, {len(ocr_segments)} segments contain OCR text."
                if ocr_segments else "No on-screen text was detected.",
            },
            video_id=video_id,
            difficulty="easy",
            tags=["visual-description", "existence", "ocr", "requires_video", "visual_only"],
        ))

        # Q: Compare visual content across two segments
        if len(segments_with_caption) >= 2:
            seg_a = segments_with_caption[0]
            seg_b = segments_with_caption[len(segments_with_caption) // 2]
            ts_a = ms_to_timestamp(seg_a["start_ms"])
            ts_b = ms_to_timestamp(seg_b["start_ms"])
            entries.append(self._make_entry(
                category="cross_annotation",
                question=f"How does the visual content at {ts_a} differ from {ts_b}?",
                answer={
                    "value": f"At {ts_a}: {seg_a['visual_caption'][:150]}. At {ts_b}: {seg_b['visual_caption'][:150]}.",
                    "explanation": "Comparison of visual descriptions at two different timestamps.",
                },
                video_id=video_id,
                difficulty="hard",
                tags=["visual-description", "comparison", "requires_video", "visual_only"],
                context={
                    "video_guid": video_id,
                    "timestamps": [ts_a, ts_b],
                },
            ))

        # Q: What scene types have visual captions?
        captioned_labels = defaultdict(int)
        for seg in segments_with_caption:
            label = seg.get("scene_label", "")
            if label:
                captioned_labels[label] += 1
        if captioned_labels:
            entries.append(self._make_entry(
                category="cross_annotation",
                question="Which scene types in this video have visual descriptions available?",
                answer={
                    "value": dict(captioned_labels),
                    "explanation": ", ".join(f"{l}: {c}" for l, c in sorted(captioned_labels.items())),
                },
                video_id=video_id,
                difficulty="medium",
                tags=["visual-description", "aggregation", "requires_video", "visual_only"],
            ))

        return entries

    # ─── Category 8: Shot-Level Questions ─────────────────────────

    def _generate_shot_level(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Questions about shot boundaries and shot-level analysis.

        Works with TransNet shot segments or any segment with scene_label='shot'.
        Also generates questions about segment transitions and pacing.
        """
        entries = []
        video_id = index["video_id"]

        # Questions about total segment count and pacing
        duration_ms = index.get("duration_ms", 0)
        n_segments = len(segments)

        if n_segments >= 3:
            # Q: How many distinct segments does this video have?
            entries.append(self._make_entry(
                category="scene_classification",
                question="How many distinct segments or shots does this video contain?",
                answer={"count": n_segments},
                video_id=video_id,
                difficulty="easy",
                tags=["shot-level", "count", "requires_video"],
            ))

            # Q: What is the average segment duration?
            durations = [(s["end_ms"] - s["start_ms"]) for s in segments]
            avg_dur = sum(durations) / len(durations)
            entries.append(self._make_entry(
                category="cross_annotation",
                question="What is the average segment duration in this video?",
                answer={
                    "value": ms_to_human(int(avg_dur)),
                    "explanation": f"Average segment is {ms_to_human(int(avg_dur))} "
                                   f"({n_segments} segments over {ms_to_human(duration_ms) if duration_ms else 'unknown duration'}).",
                },
                video_id=video_id,
                difficulty="medium",
                tags=["shot-level", "duration", "requires_video"],
            ))

            # Q: What is the shortest segment?
            shortest = min(segments, key=lambda s: s["end_ms"] - s["start_ms"])
            shortest_dur = shortest["end_ms"] - shortest["start_ms"]
            entries.append(self._make_entry(
                category="cross_annotation",
                question="What is the shortest segment in this video?",
                answer={
                    "value": f"{shortest.get('scene_label', 'unknown')} ({ms_to_human(shortest_dur)})",
                    "explanation": (
                        f"Shortest segment: {shortest.get('scene_label', '')} from "
                        f"{ms_to_timestamp(shortest['start_ms'])} to {ms_to_timestamp(shortest['end_ms'])} "
                        f"({ms_to_human(shortest_dur)})."
                    ),
                },
                video_id=video_id,
                difficulty="medium",
                tags=["shot-level", "duration", "requires_video"],
            ))

        # Q: How many scene transitions occur?
        transitions = []
        for i in range(len(segments) - 1):
            curr_label = segments[i].get("scene_label", "")
            next_label = segments[i + 1].get("scene_label", "")
            if curr_label and next_label and curr_label != next_label:
                transitions.append((curr_label, next_label, segments[i]["end_ms"]))

        if transitions:
            entries.append(self._make_entry(
                category="cross_annotation",
                question="How many scene type transitions occur in this video?",
                answer={
                    "count": len(transitions),
                    "explanation": f"{len(transitions)} transitions between different scene types.",
                },
                video_id=video_id,
                difficulty="medium",
                tags=["shot-level", "transitions", "requires_video"],
            ))

            # Q: What are the most common transitions?
            transition_counts = defaultdict(int)
            for src, dst, _ in transitions:
                transition_counts[f"{src} → {dst}"] += 1
            top_transitions = sorted(transition_counts.items(), key=lambda x: -x[1])[:5]
            entries.append(self._make_entry(
                category="cross_annotation",
                question="What are the most common scene transitions in this video?",
                answer={
                    "value": {t: c for t, c in top_transitions},
                    "explanation": "; ".join(f"{t}: {c}x" for t, c in top_transitions),
                },
                video_id=video_id,
                difficulty="hard",
                tags=["shot-level", "transitions", "requires_video"],
            ))

        # Q: Segment at specific time (different from scene-at-time — asks about segment boundaries)
        if n_segments >= 5:
            mid_seg = segments[n_segments // 2]
            mid_ts = ms_to_timestamp((mid_seg["start_ms"] + mid_seg["end_ms"]) // 2)
            entries.append(self._make_entry(
                category="scene_classification",
                question=f"What segment is playing at {mid_ts}, and what are its exact boundaries?",
                answer={
                    "scene_label": mid_seg.get("scene_label", "unknown"),
                    "timestamps": [
                        ms_to_timestamp(mid_seg["start_ms"]),
                        ms_to_timestamp(mid_seg["end_ms"]),
                    ],
                    "explanation": (
                        f"Segment: {mid_seg.get('scene_label', '')} from "
                        f"{ms_to_timestamp(mid_seg['start_ms'])} to {ms_to_timestamp(mid_seg['end_ms'])}."
                    ),
                },
                video_id=video_id,
                difficulty="medium",
                tags=["shot-level", "boundaries", "requires_video"],
            ))

        # Q: Content density — which segments have the most annotations?
        richest = max(segments, key=lambda s: (
            (1 if s.get("visual_caption", "").strip() else 0) +
            (1 if s.get("ocr_text", "").strip() else 0) +
            (1 if is_substantive_transcript(s.get("asr_transcript", "")) else 0) +
            len(s.get("named_entities", []))
        ))
        ts = ms_to_timestamp(richest["start_ms"])
        modalities = []
        if richest.get("visual_caption", "").strip():
            modalities.append("visual caption")
        if richest.get("ocr_text", "").strip():
            modalities.append("OCR text")
        if is_substantive_transcript(richest.get("asr_transcript", "")):
            modalities.append("ASR transcript")
        if richest.get("named_entities"):
            modalities.append(f"{len(richest['named_entities'])} entities")

        entries.append(self._make_entry(
            category="cross_annotation",
            question="Which segment in this video has the richest annotation coverage?",
            answer={
                "value": f"{richest.get('scene_label', 'unknown')} at {ts}",
                "explanation": f"Segment at {ts} has: {', '.join(modalities)}.",
            },
            video_id=video_id,
            difficulty="hard",
            tags=["shot-level", "density", "cross-layer", "requires_video"],
        ))

        return entries


    # ------------------------------------------------------------------ #
    # Relation-based QA (from SVO triples)
    # ------------------------------------------------------------------ #

    def _generate_relation_qa(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Generate QA from extracted (subject, predicate, object) triples.

        Question types:
        - What did [subject] do? → predicate + object
        - Who [predicated] [object]? → subject
        - What happened to [object]? → subject + predicate
        - What actions are described involving [entity]? → list of relations
        """
        entries = []
        video_id = index["video_id"]

        # Collect all relations with their segment context
        all_relations = []  # (subj, pred, obj, segment)
        for seg in segments:
            for rel in seg.get("relations", []):
                if len(rel) == 3:
                    subj, pred, obj = rel
                    # Filter out very generic subjects/objects
                    if len(subj) > 3 and len(obj) > 3:
                        all_relations.append((subj, pred, obj, seg))

        if not all_relations:
            return entries

        # Build entity-to-relations index
        entity_rels = defaultdict(list)
        for subj, pred, obj, seg in all_relations:
            entity_rels[subj].append((pred, obj, seg))
            entity_rels[obj].append((pred, subj, seg))

        # --- Q type 1: What did [subject] do? ---
        # Pick entities with named-entity overlap for higher quality
        named_entity_texts = set()
        for seg in segments:
            for ent in seg.get("named_entities", []):
                named_entity_texts.add(ent.get("text", ""))

        # Subjects that are also named entities → highest quality questions
        ne_subjects = [(s, p, o, seg) for s, p, o, seg in all_relations
                       if s in named_entity_texts]

        sampled = random.sample(ne_subjects, min(15, len(ne_subjects))) if ne_subjects else []
        for subj, pred, obj, seg in sampled:
            ts = ms_to_timestamp(seg["start_ms"])
            question = f"What did {subj} do in this video?"
            # Collect all actions for this subject
            actions = [(p, o) for p, o, _ in entity_rels.get(subj, [])
                       if p != pred or o != obj][:5]
            all_actions = [(pred, obj)] + actions

            answer_text = "; ".join(f"{p} {o}" for p, o in all_actions)
            entries.append(self._make_entry(
                category="factual_extraction",
                question=question,
                answer={
                    "value": answer_text,
                    "explanation": f"{subj} appears at {ts} and is described as: {answer_text}",
                },
                video_id=video_id,
                difficulty="medium",
                tags=["relation", "svo", "entity-action"],
                context={"video_guid": video_id, "timestamp": ts},
            ))

        # --- Q type 2: Who [predicated] [object]? ---
        # Find unique (pred, obj) pairs with interesting predicates
        pred_obj_subjects = defaultdict(set)
        for subj, pred, obj, seg in all_relations:
            if subj in named_entity_texts:
                pred_obj_subjects[(pred, obj)].add(subj)

        interesting_preds = [(po, subs) for po, subs in pred_obj_subjects.items()
                             if len(list(subs)[0]) > 3]
        sampled2 = random.sample(interesting_preds, min(10, len(interesting_preds)))
        for (pred, obj), subjects in sampled2:
            subjects_list = list(subjects)
            question = f'Who {pred} {obj[:60]}{"..." if len(obj) > 60 else ""}?'
            entries.append(self._make_entry(
                category="factual_extraction",
                question=question,
                answer={
                    "value": ", ".join(subjects_list),
                    "entities": [{"text": s, "type": "unknown"} for s in subjects_list],
                },
                video_id=video_id,
                difficulty="medium" if len(subjects_list) == 1 else "hard",
                tags=["relation", "svo", "who-did"],
            ))

        # --- Q type 3: What entities are involved in [action]? ---
        pred_counts = defaultdict(list)
        for subj, pred, obj, seg in all_relations:
            pred_counts[pred].append((subj, obj))

        # Pick predicates that appear multiple times (thematic)
        repeated_preds = [(p, pairs) for p, pairs in pred_counts.items()
                          if len(pairs) >= 2 and len(p) > 3]
        sampled3 = random.sample(repeated_preds, min(5, len(repeated_preds)))
        for pred, pairs in sampled3:
            all_entities = set()
            for s, o in pairs:
                all_entities.add(s)
                all_entities.add(o)
            question = f'What entities are involved in "{pred}" actions in this video?'
            entries.append(self._make_entry(
                category="factual_extraction",
                question=question,
                answer={
                    "value": ", ".join(sorted(all_entities)[:10]),
                    "explanation": f'The predicate "{pred}" involves: {", ".join(sorted(all_entities)[:10])}',
                },
                video_id=video_id,
                difficulty="hard",
                tags=["relation", "svo", "predicate-aggregation"],
            ))

        # --- Q type 4: Relation count / summary ---
        if all_relations:
            unique_preds = set(pred for _, pred, _, _ in all_relations)
            entries.append(self._make_entry(
                category="factual_extraction",
                question="How many distinct actions or relationships are described in this video?",
                answer={
                    "count": len(all_relations),
                    "value": f"{len(all_relations)} relations involving {len(unique_preds)} distinct predicates",
                    "explanation": f"Top predicates: {', '.join(list(unique_preds)[:10])}",
                },
                video_id=video_id,
                difficulty="easy",
                tags=["relation", "count", "summary"],
            ))

        return entries

    def _generate_multi_hop_relation(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Generate multi-hop questions requiring reasoning across relations + entities + time.

        These questions can't be answered from a single segment — they require
        connecting information across multiple segments or annotation layers.

        Question types:
        - Entity co-occurrence: "Which people appear in the same segment as [location]?"
        - Temporal entity tracking: "When does [entity] first/last appear?"
        - Cross-modal: "What is being said when [visual element] is shown?"
        - Relation chains: "Who does X, and what else happens in that segment?"
        """
        entries = []
        video_id = index["video_id"]

        # Build cross-segment entity index
        entity_segments = defaultdict(list)  # entity_text → [(seg_idx, seg)]
        for i, seg in enumerate(segments):
            for ent in seg.get("named_entities", []):
                entity_segments[ent["text"]].append((i, seg))

        # --- Multi-segment entity tracking ---
        # Entities appearing in 2+ segments
        recurring = {e: segs for e, segs in entity_segments.items()
                     if len(segs) >= 2 and len(e) > 3}

        sampled = random.sample(list(recurring.items()), min(8, len(recurring)))
        for entity_text, seg_refs in sampled:
            first_seg = seg_refs[0][1]
            last_seg = seg_refs[-1][1]
            first_ts = ms_to_timestamp(first_seg["start_ms"])
            last_ts = ms_to_timestamp(last_seg["start_ms"])

            # Q: When does [entity] first appear?
            entries.append(self._make_entry(
                category="multi_hop",
                question=f'When does "{entity_text}" first appear in this video?',
                answer={
                    "value": first_ts,
                    "explanation": f'"{entity_text}" first appears at {first_ts} '
                                   f'and last at {last_ts} (across {len(seg_refs)} segments).',
                },
                video_id=video_id,
                difficulty="medium",
                tags=["multi-hop", "entity-tracking", "temporal"],
                context={"video_guid": video_id, "timestamp": first_ts},
            ))

            # Q: How many segments mention [entity]?
            entries.append(self._make_entry(
                category="multi_hop",
                question=f'How many segments in this video mention "{entity_text}"?',
                answer={
                    "count": len(seg_refs),
                    "value": f"{len(seg_refs)} segments",
                    "explanation": (
                        f'"{entity_text}" appears in {len(seg_refs)} segments, from '
                        f'{first_ts} to {last_ts}.'
                    ),
                },
                video_id=video_id,
                difficulty="medium",
                tags=["multi-hop", "entity-tracking", "count"],
            ))

        # --- Entity co-occurrence questions ---
        # Find pairs of named entities that appear in the same segment
        cooccurrences = defaultdict(set)
        for seg in segments:
            ents = [e["text"] for e in seg.get("named_entities", [])
                    if e.get("type") in INTERESTING_ENTITY_TYPES and len(e["text"]) > 3]
            for i, e1 in enumerate(ents):
                for e2 in ents[i + 1:]:
                    cooccurrences[(e1, e2)].add(ms_to_timestamp(seg["start_ms"]))

        # Pick entity pairs that co-occur in multiple segments
        multi_cooccur = [((e1, e2), times) for (e1, e2), times in cooccurrences.items()
                         if len(times) >= 2]
        sampled_co = random.sample(multi_cooccur, min(5, len(multi_cooccur)))
        for (e1, e2), times in sampled_co:
            entries.append(self._make_entry(
                category="multi_hop",
                question=f'In how many segments do "{e1}" and "{e2}" appear together?',
                answer={
                    "count": len(times),
                    "value": f"{len(times)} segments",
                    "explanation": f'"{e1}" and "{e2}" co-occur at: {", ".join(sorted(times)[:5])}.',
                },
                video_id=video_id,
                difficulty="hard",
                tags=["multi-hop", "co-occurrence", "entity-pair"],
            ))

        # --- Cross-modal: speech + visual at same time ---
        for seg in random.sample(segments, min(5, len(segments))):
            asr = seg.get("asr_transcript", "").strip()
            caption = seg.get("visual_caption", "").strip()
            rels = seg.get("relations", [])
            if asr and caption and len(asr) > 20 and len(caption) > 20:
                ts = ms_to_timestamp(seg["start_ms"])
                entries.append(self._make_entry(
                    category="cross_modal",
                    question=f"At {ts}, what is being said while the camera shows the scene?",
                    answer={
                        "value": asr[:200],
                        "explanation": f"Speech: {asr[:150]}... Visual: {caption[:150]}...",
                    },
                    video_id=video_id,
                    difficulty="hard",
                    tags=["multi-hop", "cross-modal", "asr-visual"],
                    context={"video_guid": video_id, "timestamp": ts,
                             "transcript_snippet": asr[:200]},
                ))
                break  # One per video

        # --- Relation chain questions ---
        # "What happens in the segment where [entity] is mentioned?"
        ne_with_rels = []
        for seg in segments:
            ents = [e["text"] for e in seg.get("named_entities", [])
                    if e.get("type") == "PERSON" and len(e["text"]) > 3]
            rels = seg.get("relations", [])
            if ents and rels:
                ne_with_rels.append((seg, ents, rels))

        sampled_chains = random.sample(ne_with_rels, min(5, len(ne_with_rels)))
        for seg, ents, rels in sampled_chains:
            entity = random.choice(ents)
            ts = ms_to_timestamp(seg["start_ms"])
            rel_strs = [f"{s} {p} {o}" for s, p, o in rels[:5]]
            entries.append(self._make_entry(
                category="multi_hop",
                question=f'What actions are described in the segment where "{entity}" is mentioned?',
                answer={
                    "value": "; ".join(rel_strs),
                    "explanation": (
                        f'At {ts}, "{entity}" appears alongside these actions: '
                        f'{"; ".join(rel_strs)}'
                    ),
                },
                video_id=video_id,
                difficulty="hard",
                tags=["multi-hop", "relation-chain", "entity-context"],
                context={"video_guid": video_id, "timestamp": ts},
            ))

        return entries

    def _generate_speaker_qa(self, index: dict, segments: list[dict]) -> list[QAEntry]:
        """Generate multi-hop questions combining speaker identity, content, entities, and time.

        These questions require reasoning across speaker diarization, ASR content,
        named entities, and temporal/broadcast context. They can't be answered
        from any single annotation layer alone.

        Question types:
        - What topics did [speaker] discuss? (speaker + NER aggregation)
        - Which speakers mentioned [entity]? (entity + speaker cross-ref)
        - What did [speaker] say about [entity]? (speaker + entity + transcript)
        - Did [speaker] and [other speaker] discuss the same topic? (cross-speaker)
        - What was discussed at [time] and by whom? (temporal + speaker + content)
        """
        entries = []
        video_id = index["video_id"]
        speaker_names = index.get("speaker_names", {})
        broadcast_date = index.get("broadcast_date", "")

        # Skip videos without speaker data
        if not any(s.get("primary_speaker") for s in segments):
            return entries

        # Build speaker → segments map
        speaker_segs = defaultdict(list)
        for seg in segments:
            spk = seg.get("primary_speaker")
            if spk and seg.get("asr_transcript"):
                speaker_segs[spk].append(seg)

        # Build speaker → entities map
        speaker_entities = defaultdict(lambda: defaultdict(int))
        for seg in segments:
            spk = seg.get("primary_speaker")
            if not spk:
                continue
            for ent in seg.get("named_entities", []):
                if ent.get("type") in INTERESTING_ENTITY_TYPES:
                    speaker_entities[spk][ent["text"]] += 1

        named_speakers = {spk: name for spk, name in speaker_names.items() if name}

        # --- Q: What topics did [named speaker] discuss? ---
        for spk, name in named_speakers.items():
            ents = speaker_entities.get(spk, {})
            if len(ents) < 3:
                continue
            top_ents = sorted(ents.items(), key=lambda x: -x[1])[:8]
            top_names = [e for e, _ in top_ents]

            entries.append(self._make_entry(
                category="multi_hop",
                question=f'What topics and entities did {name} discuss in this broadcast?',
                answer={
                    "value": ", ".join(top_names),
                    "explanation": f"{name} mentioned: {', '.join(f'{e} ({c}x)' for e, c in top_ents)}",
                },
                video_id=video_id,
                difficulty="hard",
                tags=["speaker", "multi-hop", "entity-aggregation", "requires_video"],
            ))

        # --- Q: Which speakers mentioned [entity]? ---
        # Find entities mentioned by multiple speakers
        entity_speakers = defaultdict(set)
        for spk, ents in speaker_entities.items():
            display = named_speakers.get(spk, spk)
            for ent_text in ents:
                entity_speakers[ent_text].add(display)

        multi_speaker_ents = [(ent, spks) for ent, spks in entity_speakers.items()
                              if len(spks) >= 2 and len(ent) > 3]
        sampled = random.sample(multi_speaker_ents, min(5, len(multi_speaker_ents)))
        for ent_text, spks in sampled:
            entries.append(self._make_entry(
                category="multi_hop",
                question=f'Which speakers in this broadcast mentioned "{ent_text}"?',
                answer={
                    "value": ", ".join(sorted(spks)),
                    "explanation": f'"{ent_text}" was mentioned by: {", ".join(sorted(spks))}',
                },
                video_id=video_id,
                difficulty="hard",
                tags=["speaker", "multi-hop", "entity-speaker-crossref"],
            ))

        # --- Q: What did [speaker] say about [entity]? ---
        for spk, name in named_speakers.items():
            ents = speaker_entities.get(spk, {})
            # Pick an interesting entity this speaker mentioned
            interesting = [(e, c) for e, c in ents.items()
                           if c >= 2 and len(e) > 4]
            if not interesting:
                continue
            ent_text, count = random.choice(interesting)
            # Find transcript snippets where this entity appears
            snippets = []
            for seg in speaker_segs.get(spk, []):
                asr = seg.get("asr_transcript", "")
                if ent_text.lower() in asr.lower():
                    snippets.append(asr[:150])
            if snippets:
                entries.append(self._make_entry(
                    category="multi_hop",
                    question=f'What did {name} say about {ent_text}?',
                    answer={
                        "value": snippets[0],
                        "explanation": f'{name} mentioned "{ent_text}" in {len(snippets)} segment(s)',
                    },
                    video_id=video_id,
                    difficulty="hard",
                    tags=["speaker", "multi-hop", "speaker-entity-transcript"],
                ))

        # --- Q: Did speakers discuss [topic] — cross-speaker topic overlap ---
        if len(named_speakers) >= 2:
            spk_list = list(named_speakers.items())
            for i in range(min(3, len(spk_list))):
                for j in range(i + 1, min(4, len(spk_list))):
                    spk_a, name_a = spk_list[i]
                    spk_b, name_b = spk_list[j]
                    ents_a = set(speaker_entities.get(spk_a, {}).keys())
                    ents_b = set(speaker_entities.get(spk_b, {}).keys())
                    shared = ents_a & ents_b
                    shared = {e for e in shared if len(e) > 3}
                    if shared:
                        entries.append(self._make_entry(
                            category="multi_hop",
                            question=f'Did {name_a} and {name_b} discuss any of the same topics?',
                            answer={
                                "value": f"Yes, they both mentioned: {', '.join(sorted(shared)[:5])}",
                                "explanation": f"Shared entities: {', '.join(sorted(shared)[:10])}",
                            },
                            video_id=video_id,
                            difficulty="hard",
                            tags=["speaker", "multi-hop", "cross-speaker-overlap"],
                        ))

        # --- Q: Temporal + speaker + content (broadcast date context) ---
        if broadcast_date and named_speakers:
            for spk, name in list(named_speakers.items())[:2]:
                segs = speaker_segs.get(spk, [])
                if not segs:
                    continue
                # Find a segment with interesting entities
                for seg in segs:
                    persons = [e["text"] for e in seg.get("named_entities", [])
                               if e.get("type") == "PERSON" and len(e["text"]) > 4
                               and e["text"] != name]
                    if persons:
                        person = persons[0]
                        ts = ms_to_timestamp(seg["start_ms"])
                        entries.append(self._make_entry(
                            category="multi_hop",
                            question=f'In the {broadcast_date} broadcast, did {name} discuss {person}?',
                            answer={
                                "value": f"Yes, at {ts}",
                                "explanation": f'{name} mentioned {person} at {ts}: "{seg["asr_transcript"][:120]}..."',
                            },
                            video_id=video_id,
                            difficulty="hard",
                            tags=["speaker", "multi-hop", "temporal-speaker-entity", "date-context"],
                            context={"video_guid": video_id, "timestamp": ts},
                        ))
                        break

        return entries


def print_stats(entries: list[QAEntry]):
    """Print statistics about generated QA entries."""
    by_category = defaultdict(int)
    by_difficulty = defaultdict(int)
    by_tag = defaultdict(int)

    for e in entries:
        by_category[e.category] += 1
        if e.difficulty:
            by_difficulty[e.difficulty] += 1
        for tag in e.tags:
            by_tag[tag] += 1

    print(f"\nTotal QA pairs: {len(entries)}")
    print(f"\nBy category:")
    for cat, count in sorted(by_category.items()):
        print(f"  {cat}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in sorted(by_difficulty.items()):
        print(f"  {diff}: {count}")
    print(f"\nTop tags:")
    for tag, count in sorted(by_tag.items(), key=lambda x: -x[1])[:15]:
        print(f"  {tag}: {count}")

    # Modality analysis
    requires_video = sum(1 for e in entries if "requires_video" in e.tags)
    requires_audio = sum(1 for e in entries if "requires_audio" in e.tags)
    both = sum(1 for e in entries if "requires_video" in e.tags and "requires_audio" in e.tags)
    print(f"\nModality requirements:")
    print(f"  requires_video: {requires_video}")
    print(f"  requires_audio: {requires_audio}")
    print(f"  requires_both: {both}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA pairs from video indexes")
    parser.add_argument("--index-dir", default=None,
                        help="Path to video index directory")
    parser.add_argument("--video", default=None,
                        help="Generate for a specific video ID only")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file path")
    parser.add_argument("--mc", action="store_true",
                        help="Generate multiple-choice questions with distractors")
    parser.add_argument("--stats", action="store_true",
                        help="Print statistics only, don't write output")
    args = parser.parse_args()

    # Resolve index directory
    project_root = Path(__file__).parent.parent
    index_dir = Path(args.index_dir) if args.index_dir else project_root / "data" / "video_indexes"

    if not index_dir.exists():
        print(f"Error: Index directory not found: {index_dir}", file=sys.stderr)
        sys.exit(1)

    generator = IndexQAGenerator(index_dir, multiple_choice=args.mc)
    if args.mc:
        print("Mode: Multiple Choice")

    # List available videos
    videos = generator.list_indexed_videos()
    if not videos:
        print("No indexed videos found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} indexed video(s): {', '.join(videos)}")

    # Generate
    entries = generator.generate_all(video_id=args.video)

    print_stats(entries)

    if args.stats:
        return

    # Write output
    default_name = "index_qa_mc.jsonl" if args.mc else "index_qa.jsonl"
    output_path = Path(args.output) if args.output else Path(__file__).parent / "raw" / default_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json() + "\n")

    print(f"\nWrote {len(entries)} entries to {output_path}")

    # Print samples
    print("\n--- Sample entries ---")
    samples = random.sample(entries, min(5, len(entries)))
    for entry in samples:
        print(f"\n[{entry.category}] ({entry.difficulty})")
        print(f"  Q: {entry.question}")
        ans = entry.answer
        if "text" in ans:
            print(f"  A: {ans['text'][:100]}...")
        elif "value" in ans:
            print(f"  A: {ans['value']}")
        elif "entities" in ans:
            print(f"  A: {[e['text'] for e in ans['entities'][:5]]}")
        elif "scene_label" in ans:
            print(f"  A: {ans['scene_label']}")
        elif "count" in ans:
            print(f"  A: count={ans['count']}")
        elif "people" in ans:
            print(f"  A: {ans['people'][:5]}")
        elif "organizations" in ans:
            print(f"  A: {ans['organizations'][:5]}")
        elif "locations" in ans:
            print(f"  A: {ans['locations'][:5]}")
        else:
            print(f"  A: {json.dumps(ans)[:100]}")
        print(f"  Tags: {entry.tags}")


if __name__ == "__main__":
    main()
