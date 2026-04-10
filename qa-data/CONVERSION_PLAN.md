# QA Data Conversion Plan

This document outlines the plan for converting AAPB annotations into question-answering (QA) format for training and evaluating the CLAMS agent.

## Source Data Overview

The `aapb-annotations` repository contains 8,781 annotation files across 10 major projects:

| Project | Annotation Type | Gold Files | Key Data |
|---------|----------------|------------|----------|
| `scene-recognition` | Frame classification | 1000+ | Frame timestamps + scene labels |
| `january-slates` | Slate detection | 468 | Time intervals + metadata |
| `understanding-slates` | Slate transcription | 503 | OCR text + structured parsing |
| `understanding-chyrons` | Chyron understanding | 513 | Text extraction + parsing |
| `newshour-chyron` | Chyron detection | 36 | Time intervals + text |
| `newshour-namedentity` | NER | 30 | Character offsets + entity types |
| `newshour-namedentity-wikipedialink` | Entity linking | 20 | Wikipedia/Wikidata grounding |
| `role-filler-binding` | Key-value extraction | 43 | Structured metadata |
| `role-filler-binding-seqtag` | Sequence tagging | 1 | BIO tags for roles/fillers |
| `newshour-transcript-sync` | Transcript alignment | 10 | Word-level time alignment |

## QA Task Categories

### Category 1: Tool Selection Questions
**Goal**: Train agent to recommend appropriate CLAMS tools for user tasks.

**Question Types**:
- "What tool should I use to detect slates in this video?"
- "I want to extract named entities from news transcripts. What should I use?"
- "How do I identify chyrons in NewsHour footage?"

**Answer Format**: Tool name + brief justification

**Source Data**: Derived from annotation project descriptions (which tools/pipelines produced these annotations)

### Category 2: Scene/Frame Classification Questions
**Goal**: Train agent on temporal video understanding.

**Question Types**:
- "What type of scene appears at timestamp 00:01:12.505?"
- "Is there a slate in the first 30 seconds of this video?"
- "Find all chyron appearances in this video."

**Answer Format**: Scene label + subtype + time range

**Source Data**: `scene-recognition`, `january-slates`, `newshour-chyron`

### Category 3: Text Extraction Questions
**Goal**: Train agent on OCR and text understanding.

**Question Types**:
- "What text appears in the chyron at 00:04:35?"
- "Transcribe the slate at the beginning of this video."
- "What names appear on screen during credits?"

**Answer Format**: Extracted text (verbatim or structured)

**Source Data**: `understanding-slates`, `understanding-chyrons`, `newshour-chyron`

### Category 4: Named Entity Questions
**Goal**: Train agent on entity extraction and linking.

**Question Types**:
- "Who is mentioned in this transcript?"
- "What organizations appear in this news segment?"
- "Link 'Jim Lehrer' to their Wikipedia page."

**Answer Format**: Entity text + type + (optional) Wikipedia QID

**Source Data**: `newshour-namedentity`, `newshour-namedentity-wikipedialink`

### Category 5: Structured Metadata Extraction
**Goal**: Train agent on key-value extraction from media.

**Question Types**:
- "Who directed this program according to the credits?"
- "What is the episode title shown in the slate?"
- "Extract producer names from this credit sequence."

**Answer Format**: Structured dict of role -> filler mappings

**Source Data**: `role-filler-binding`, `role-filler-binding-seqtag`

### Category 6: Pipeline Design Questions
**Goal**: Train agent to chain tools appropriately.

**Question Types**:
- "How do I go from raw video to named entity extraction?"
- "Design a pipeline to extract chyron text and identify people mentioned."
- "What tools need to run before I can do entity linking?"

**Answer Format**: Ordered list of tools with I/O type compatibility

**Source Data**: Derived from annotation task dependencies

### Category 7: Cross-Annotation Questions (Multi-Source Joins)
**Goal**: Train agent on complex queries requiring data from multiple annotation sources.

These are higher-difficulty questions that simulate real user queries spanning multiple analysis types. They require joining gold annotations across projects by GUID.

**Question Types**:

**Chyron + Named Entity (person lookup)**:
- "Find all chyrons that mention Jim Lehrer."
- "Which videos have chyrons identifying George H.W. Bush?"
- "Show me chyrons with organization names."

**Transcript + Named Entity (speech attribution)**:
- "What does George W. Bush say in NewsHour videos?"
- "Find segments where Nancy Pelosi is speaking."
- "What organizations are discussed by Jim Lehrer?"

**Scene Detection + Text Extraction**:
- "What text appears in the slate scenes?"
- "Extract credits text from videos that have credit sequences."
- "Find chyron text in the first 5 minutes of each video."

**Named Entity + Entity Linking (enriched entity queries)**:
- "Find mentions of people with Wikipedia pages about politicians."
- "Which organizations mentioned have Wikidata entries?"
- "Show entities linked to Q-codes starting with Q1*."

**Role-Filler + Scene Detection**:
- "Who directed videos that have digital slates?"
- "Find executive producers from programs with chyrons."

**Answer Format**:
```json
{
  "matches": [
    {
      "guid": "cpb-aacip-507-xxx",
      "chyron_timestamp": "00:04:35.777",
      "chyron_text": "JIM LEHRER\nNewsHour Host",
      "entity_match": {"text": "JIM LEHRER", "type": "person", "qid": "Q931148"}
    }
  ],
  "total_count": 15,
  "source_annotations": ["newshour-chyron", "newshour-namedentity-wikipedialink"]
}
```

**Source Data**: Requires joining multiple gold annotation sets by GUID

**Implementation Notes**:
- Build a GUID index mapping each video to all available annotations
- Create join functions that align annotations by GUID and time overlap
- For speech attribution: use transcript-sync to align NER spans with speaker segments
- Entity name matching should handle variations (e.g., "Jim Lehrer" vs "JAMES LEHRER")
- Consider fuzzy matching for entity text in chyrons vs. transcripts

**Example Join Logic**:
```python
# Find chyrons mentioning a specific person
def find_chyrons_with_entity(entity_name: str) -> list[dict]:
    results = []
    for guid in get_guids_with_both("newshour-chyron", "newshour-namedentity"):
        chyrons = load_chyron_annotations(guid)
        entities = load_ner_annotations(guid)

        # Find entity mentions matching the name
        matching_entities = [e for e in entities if entity_name.lower() in e["text"].lower()]

        for chyron in chyrons:
            # Check if chyron text contains any matching entity
            for entity in matching_entities:
                if entity["text"].lower() in chyron["text"].lower():
                    results.append({
                        "guid": guid,
                        "chyron": chyron,
                        "entity": entity
                    })
    return results
```

## Proposed QA Data Format

### Standard QA Entry (JSON)

```json
{
  "id": "qa-001234",
  "category": "scene_classification",
  "source_project": "scene-recognition",
  "source_guid": "cpb-aacip-507-1v5bc3tf81",
  "question": "What type of scene appears at timestamp 00:01:12.505?",
  "answer": {
    "scene_label": "slate",
    "scene_subtype": "digital",
    "confidence": "gold"
  },
  "context": {
    "video_guid": "cpb-aacip-507-1v5bc3tf81",
    "timestamp": "00:01:12.505",
    "collection": "NewsHour"
  },
  "metadata": {
    "created_from": "scene-recognition/golds/cpb-aacip-507-1v5bc3tf81.csv",
    "annotator_count": 2,
    "conversion_date": "2025-01-16"
  }
}
```

### Tool Selection QA Entry

```json
{
  "id": "qa-tool-001",
  "category": "tool_selection",
  "question": "I have a NewsHour video and want to extract the names of people shown in chyrons. What tools should I use?",
  "answer": {
    "tools": [
      {"name": "scenes-with-text", "purpose": "Detect frames with text"},
      {"name": "app-smolvlm2-captioner", "purpose": "OCR the chyron text"},
      {"name": "spacy-ner", "purpose": "Extract person names from text"}
    ],
    "pipeline_order": ["scenes-with-text", "app-smolvlm2-captioner", "spacy-ner"],
    "reasoning": "First detect text-containing scenes, then OCR the chyrons, then extract named entities from the transcribed text."
  },
  "metadata": {
    "derived_from_annotation_project": "newshour-namedentity"
  }
}
```

### Multi-Turn Conversation QA

```json
{
  "id": "qa-conv-001",
  "category": "multi_turn",
  "conversation": [
    {"role": "user", "content": "What kinds of text appear in broadcast videos?"},
    {"role": "assistant", "content": "Common text types include: slates (production metadata), chyrons (lower-third banners), credits, and on-screen graphics."},
    {"role": "user", "content": "How do I find chyrons specifically?"},
    {"role": "assistant", "content": "Use the scenes-with-text tool to detect frames containing text, then filter for frames labeled as 'chyron'. You can then use a captioner tool to extract the text content."}
  ],
  "source_projects": ["scene-recognition", "newshour-chyron"]
}
```

## Conversion Strategy by Source

### 1. scene-recognition -> QA

**Input**: CSV with `at`, `scene-label`, `scene-subtype-label`, `transitional`

**Conversion**:
```python
# For each row, generate classification question
question = f"What type of scene appears at {row['at']}?"
answer = {
    "scene_label": row["scene-label"],
    "scene_subtype": row["scene-subtype-label"]
}

# Also generate existence questions
question = f"Does this video contain any {label} scenes?"
answer = {"exists": True, "count": N, "timestamps": [...]}
```

**Variants**:
- Point-in-time classification
- Range queries ("What scenes between X and Y?")
- Aggregation queries ("How many slates in this video?")

### 2. january-slates / newshour-chyron -> QA

**Input**: CSV with `start`, `end`, `scene-label`, text fields

**Conversion**:
```python
# Interval-based questions
question = f"When does the {scene_type} appear in this video?"
answer = {"start": start, "end": end, "duration_ms": duration}

# Content questions (if text available)
question = f"What text appears in the {scene_type} at {start}?"
answer = {"text": text_content}
```

### 3. understanding-slates / understanding-chyrons -> QA

**Input**: CSV with OCR text + structured annotations

**Conversion**:
```python
# Transcription questions
question = f"Transcribe the text visible at {timestamp}"
answer = {"raw_text": ocr_text}

# Parsing questions
question = f"What metadata can be extracted from the slate at {timestamp}?"
answer = {"title": "...", "episode": "...", "date": "..."}
```

### 4. newshour-namedentity -> QA

**Input**: TSV with `type`, `start`, `end`, `text`

**Conversion**:
```python
# Entity extraction questions
question = "What named entities appear in this transcript?"
answer = {"entities": [{"text": "Jim Lehrer", "type": "person", "span": [2, 12]}]}

# Type-specific questions
question = "What people are mentioned in this transcript?"
answer = {"people": ["Jim Lehrer", "Judy Woodruff", ...]}
```

### 5. newshour-namedentity-wikipedialink -> QA

**Input**: TSV with entity info + `wiki_url`, `qid`

**Conversion**:
```python
# Linking questions
question = "What Wikipedia article describes 'Jim Lehrer' in this context?"
answer = {
    "wiki_url": "https://en.wikipedia.org/wiki/Jim_Lehrer",
    "wikidata_qid": "Q931148"
}
```

### 6. role-filler-binding -> QA

**Input**: CSV with role-filler pairs or BIO-tagged sequences

**Conversion**:
```python
# Extraction questions
question = "Who is credited as director in this program?"
answer = {"role": "Director", "filler": "Barry Stoner"}

# Full extraction
question = "Extract all credits from this sequence"
answer = {
    "credits": [
        {"role": "Directed By", "name": "Barry Stoner"},
        {"role": "Executive Producer", "name": "Chuck McConnell"}
    ]
}
```

### 7. Cross-Annotation Joins -> QA

**Input**: Multiple gold annotation files sharing the same GUID

**Prerequisites**:
1. Build GUID inventory across all projects
2. Create alignment index for which GUIDs have which annotation types
3. Implement entity name normalization (case, whitespace, abbreviations)

**Join Strategies**:

```python
# Strategy A: GUID-based join (same video, different annotation types)
def join_by_guid(project_a: str, project_b: str) -> dict[str, tuple]:
    """Find GUIDs that have annotations in both projects."""
    guids_a = set(get_guids(project_a))
    guids_b = set(get_guids(project_b))
    common = guids_a & guids_b
    return {guid: (load(project_a, guid), load(project_b, guid)) for guid in common}

# Strategy B: Time-aligned join (overlapping time ranges)
def join_by_time_overlap(chyrons: list, entities: list) -> list[dict]:
    """Match chyrons with entities that appear in same time window."""
    # Requires transcript-sync to map entity character offsets to timestamps
    pass

# Strategy C: Text-based join (entity name appears in text field)
def join_by_text_match(chyrons: list, entities: list) -> list[dict]:
    """Match chyrons containing entity names."""
    matches = []
    for chyron in chyrons:
        chyron_text_lower = chyron["text"].lower()
        for entity in entities:
            if entity["text"].lower() in chyron_text_lower:
                matches.append({"chyron": chyron, "entity": entity})
    return matches
```

**Question Generation from Joins**:
```python
# From chyron + NER join
def generate_entity_chyron_questions(matches: list[dict]) -> list[QAEntry]:
    # Group by entity
    by_entity = defaultdict(list)
    for m in matches:
        by_entity[m["entity"]["text"]].append(m)

    questions = []
    for entity_name, entity_matches in by_entity.items():
        # "Find chyrons mentioning X"
        questions.append(QAEntry(
            question=f"Find all chyrons that mention {entity_name}.",
            answer={
                "entity": entity_name,
                "count": len(entity_matches),
                "matches": [{"guid": m["guid"], "timestamp": m["chyron"]["start"]} for m in entity_matches]
            },
            category="cross_annotation",
            tags=["chyron", "named-entity", "join"]
        ))
    return questions
```

**Key Entities to Target** (high-value for QA):
- Recurring NewsHour anchors: Jim Lehrer, Judy Woodruff, Robert MacNeil
- Politicians with many appearances: Presidents, Speakers of the House
- Frequent organizations: PBS, Congress, White House

## Implementation Phases

### Phase 1: Data Audit & Schema Definition
- [ ] Inventory all gold files with record counts
- [ ] **Build GUID cross-reference index** (which GUIDs have which annotation types)
- [ ] Define final QA JSON schema
- [ ] Create validation scripts
- [ ] Establish train/dev/test split strategy (by GUID)

### Phase 2: Core Converters (Single-Source)
- [ ] `convert_scene_recognition.py` - Frame classification QA
- [ ] `convert_chyron_detection.py` - Interval + text QA
- [ ] `convert_named_entity.py` - NER QA
- [ ] `convert_role_filler.py` - Key-value extraction QA

### Phase 3: Cross-Annotation Converters (Multi-Source Joins)
- [ ] **Build GUID inventory utility** - Map GUIDs to available annotation types
- [ ] **Entity name normalizer** - Handle case, whitespace, abbreviation variants
- [ ] `convert_chyron_entity_join.py` - Chyron + NER questions ("Find chyrons with Jim Lehrer")
- [ ] `convert_transcript_entity_join.py` - Transcript + NER ("What does X say?")
- [ ] `convert_scene_text_join.py` - Scene detection + text extraction
- [ ] **Identify high-value entities** - Build list of frequently-appearing people/orgs

### Phase 4: Derived QA Generation
- [ ] Tool selection questions (from project metadata)
- [ ] Pipeline design questions (from task dependencies)
- [ ] Multi-turn conversations (from related QA pairs)

### Phase 5: Quality & Augmentation
- [ ] Human review of sample QA pairs
- [ ] Question paraphrase generation
- [ ] Negative example generation (incorrect tool recommendations)
- [ ] Difficulty stratification (cross-annotation = harder)

### Phase 6: Integration
- [ ] Load QA data into agent evaluation framework
- [ ] Benchmark agent performance by category
- [ ] **Separate benchmarks for single-source vs. cross-annotation**
- [ ] Identify gaps and iterate

## Output Structure

```
qa-data/
├── CONVERSION_PLAN.md          # This document
├── schema/
│   └── qa_schema.json          # JSON schema for validation
├── converters/
│   ├── base_converter.py       # Shared utilities
│   ├── scene_recognition.py    # Single-source converters
│   ├── chyron_detection.py
│   ├── named_entity.py
│   ├── role_filler.py
│   ├── cross_annotation.py     # Multi-source join converters
│   └── guid_index.py           # GUID cross-reference utilities
├── raw/                        # Intermediate conversion outputs
│   ├── scene_classification.jsonl
│   ├── text_extraction.jsonl
│   ├── named_entity.jsonl
│   ├── tool_selection.jsonl
│   └── cross_annotation/       # Multi-source join outputs
│       ├── chyron_entity.jsonl
│       ├── transcript_entity.jsonl
│       └── scene_text.jsonl
├── indices/                    # Cross-reference data
│   ├── guid_inventory.json     # GUID -> available annotation types
│   └── entity_index.json       # Entity name -> GUIDs where they appear
├── splits/                     # Final train/dev/test splits
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── stats/
    └── conversion_report.md    # Statistics and quality metrics
```

## Estimated QA Counts by Category

| Category | Source Records | Est. QA Pairs | Notes |
|----------|---------------|---------------|-------|
| Scene Classification | ~50,000 frames | 100,000+ | Multiple question types per frame |
| Text Extraction | ~1,500 segments | 5,000+ | With paraphrases |
| Named Entity | ~5,000 entities | 15,000+ | Type-specific variants |
| Entity Linking | ~3,000 linked entities | 6,000+ | |
| Role-Filler | ~2,000 credits | 8,000+ | Single + aggregated |
| **Cross-Annotation** | (joined sources) | **5,000+** | **Multi-source joins, higher difficulty** |
| Tool Selection | (derived) | 500+ | Curated, high-quality |
| Pipeline Design | (derived) | 200+ | Multi-step reasoning |

**Total Estimated**: 135,000+ QA pairs

### Cross-Annotation Breakdown

| Join Type | Example Question | Est. Pairs |
|-----------|-----------------|------------|
| Chyron + NER | "Find chyrons mentioning Jim Lehrer" | 2,000+ |
| Transcript + NER | "What does George W. Bush say?" | 1,500+ |
| Scene + Text | "What text appears in slates?" | 1,000+ |
| Entity + Linking | "Find politicians with Wikipedia pages" | 500+ |

## Open Questions

1. **Difficulty levels**: How to stratify questions by complexity?
2. **Negative examples**: How many incorrect tool recommendations to include?
3. **Context length**: How much video/transcript context to include per question?
4. **Evaluation metrics**: Exact match vs. semantic similarity for answers?
5. **Multi-modal**: Should we include frame images or just timestamps?

## Next Steps

1. Review this plan and provide feedback
2. Create the JSON schema for QA entries
3. Start with `scene-recognition` converter (largest dataset)
4. Validate sample output before full conversion
