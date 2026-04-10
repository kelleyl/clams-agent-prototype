# CLAMS-Agent: Tool-Orchestrated Video Understanding

## Thesis Overview

The goal is to train a **video understanding agent** that autonomously selects and runs CLAMS tools to answer questions about long-form archival broadcast content -- videos it has never seen before, with no pre-existing index or annotations.

## Canonical Evaluation Workflow

For the current V3 setup, the canonical evaluation flow is:

1. Generate prediction artifacts with [run_policy_answerer_eval.py](/Users/kelleylynch/clams/clams-agent-prototype/eval/run_policy_answerer_eval.py)
2. Score those prediction artifacts with [score_predictions.py](/Users/kelleylynch/clams/clams-agent-prototype/eval/score_predictions.py)

This separation matters because:
- `run_policy_answerer_eval.py` is the prediction-generation step
- inline metrics from that script are MC-only previews
- free-text questions are scored post-hoc by `score_predictions.py`

Recommended current files:
- benchmark input: `qa-data/benchmark/v3/val_benchmark_gold.jsonl` or `qa-data/benchmark/v3/test_benchmark_gold.jsonl`
- prediction output: `eval/results/*.jsonl`
- final scoring: `python eval/score_predictions.py --predictions ... --benchmark ...`

### Legacy / Special-Purpose Eval Scripts

These remain in the repo but should not be treated as equivalent to the canonical V3 policy evaluation path:

- `eval/run_sft_eval.py`
  - legacy path for the older SFT-only tool API
  - useful for reproducing older V2 / legacy experiments
- `eval/run_ablation_answerer.py`
  - ablation generator for `random_tools` / `oracle`
  - save predictions, then score them with `score_predictions.py`
- `eval/run_native_tool_eval.py`, `eval/run_react_eval.py`, `eval/run_langgraph_eval.py`
  - alternative experiment paths, not the main V3 reporting pipeline

### The Agent Loop

When given a question about a new video, the trained agent:

1. **Decides what information it needs** ("I need to know what's being said")
2. **Selects and runs a CLAMS tool** (e.g., runs Parakeet ASR on the video)
3. **Processes the tool output** (reads the transcript, identifies relevant segments)
4. **Decides what to do next** ("I found someone discussing unemployment at 15:43 -- who is this person?")
5. **Runs another tool** (e.g., runs OCR on the frame at 15:47 to read the chyron)
6. **Answers the question** ("The guest represents the National Urban League")

Tool outputs are cached automatically -- if another question arrives about the same video, the agent checks the cache before running tools again. But the cache is an optimization, not the point. The agent's core capability is making good tool selection decisions under resource constraints.

### Training Pipeline

1. **Build video indexes** using CLAMS tools on 108 archival videos. This creates a comprehensive knowledge base showing what each tool produces. The index serves as training infrastructure for generating trajectories.

2. **Generate cross-modal QA** from the indexes. Questions require combining information from multiple modalities (speech + visual, OCR + entities) and cannot be answered from any single source.

3. **Reverse-engineer tool execution trajectories** from the QA evidence. Using the provenance chain in the index (which CLAMS app produced each piece of evidence), we construct trajectories showing the sequence of tool execution decisions that would produce the evidence needed to answer each question.

4. **SFT + RL training** on the trajectories. SFT teaches the model the tool calling format and basic strategies. GRPO (RL) teaches efficiency, tool selection optimization, and preference-conditioned behavior.

### Preference-Conditioned Tool Selection

The agent can be conditioned on user preferences for speed vs accuracy vs cost:

- **"Process in 10 minutes"**: Agent runs Parakeet (fast ASR) + SWT (fast text detection) + SpaCy NER. Skips expensive visual captioning.
- **"Maximum accuracy"**: Agent runs Whisper (best ASR) + VLM captioner at 1fps + full OCR + VLM-as-judge verification.
- **"Balanced"**: Agent runs Parakeet + VLM captioner at 0.2fps + OCR on text frames only.

This remains a research direction and design goal. Parts of the current GRPO work already model tool/model cost differences, but the full preference-conditioned training story described here is broader than the current headline experimental path.

### Key Distinction from RAG

The current evaluation uses RAG (context-stuffing from pre-built indexes) as a baseline. On the current V3 full MC test split, this is about 75.0% MC accuracy and assumes the full index already exists. The trained agent approach differs fundamentally:

- **RAG**: Run all tools on the full video -> build complete index -> stuff context into prompt -> answer
- **Agent**: Decide which tools to run -> run only what's needed -> process outputs -> decide if more tools are needed -> answer

RAG is an upper bound on what the index can provide. The agent trades completeness for efficiency -- it should find the right answer with fewer, more targeted tool executions.

## Dataset

- **108 videos** from AAPB (NewsHour, Peabody, NJN, IMLS, FuzzyMemories)
- **107 v2 layered indexes** with Qwen 3.5 captions, NER, entity resolution, SVO relations
- **1 v1 index** (cpb-aacip-225-009w0w1j, no combined MMIF on aristotle)

## Benchmark

### V1 (archived: `qa-data/benchmark/v1/`)

- **1,602 total questions** from 106 videos
- Generated by `qa-data/generate_llm_comprehension.py` using qwen3:30b on Ollama
- Distractors generated by `qa-data/generate_distractors.py`
- Quality review found 65% scored 2/5 or below (single-modality shortcuts, entity naming in questions, hallucinated premises)
- Quality tagging: 276 flagged, 1,326 clean (`benchmark_tagged.jsonl`)

### V2 (current: `qa-data/benchmark/v2/`)

- **809 valid questions** from 97 videos (mean 8.3 per video)
- Generated by `qa-data/generate_llm_comprehension_v2.py` using qwen3:30b (thinking model)
- Evidence-aware pipeline: builds entity-modality matrix first, uses asymmetries as question seeds

**V2 pipeline differences from V1:**
1. **Entity-modality asymmetry seeding**: Builds a matrix of which entities appear in which annotation layers. Visual-only entities (seen but not spoken, e.g., map labels) and OCR-only entities (chyrons not read aloud) become question seeds that naturally require cross-modal reasoning.
2. **Thinking model**: Uses qwen3:30b which reasons about what makes a good cross-modal question before generating.
3. **No entity naming**: Questions use descriptive references ("the guest discussing unemployment") instead of naming entities that appear in chyrons, preventing single-modality OCR lookup shortcuts.
4. **Question refinement step**: Draft questions are refined to be concise (~30 words), natural, and to weave context from both modalities into the question text itself.
5. **Cross-modality validation**: Programmatic check that the answer cannot be reached from ASR or OCR alone. Visual captions and speakers are allowed as single-source answers because the question still requires ASR context to know what to look for.

**Validation decisions:**
- **Fuzzy answer matching** (50% keyword overlap threshold): Prevents false rejections from surface form variation (e.g., "US Navy" vs "United States Navy"). Reduced "answer not found" from 360 to 13.
- **Visual/speaker answers allowed**: If the answer is in visual_captions but the question requires ASR to identify the relevant segment, that's still cross-modal. Only ASR-alone and OCR-alone answers are rejected. This rescued 377 questions that were incorrectly filtered.
- **OCR entity naming check**: Questions that name multi-word proper nouns from chyrons are rejected (24 questions), enforcing descriptive references.

**V2 quality (qwen3:8b + index 3.5):** 62.6% MC accuracy on clean benchmark (385 MC questions).

**V2 issues discovered during manual review:**
1. **Temporal distribution bias**: 75% of questions came from the first 20% of videos, 60% from the first 10%. The entity-modality seed approach found entities from early video content (bars, slates, opening logos) and generated questions about those, while the main program content was underrepresented.
2. **Interstitial content confusion**: 32% of questions (229/721) were about commercials, promos, or bumpers within complete broadcast recordings, treating them as if they were the main program content. Example: "Which venue did the Happy Days cast visit?" for a video that is actually a Barbary Coast episode with a brief Happy Days promo at 33 minutes.
3. **False guest attribution**: The model assumed entities shown on screen were guest affiliations. Example: a Washington Times newspaper shown on screen generated "What organization does the guest represent? Answer: Washington Times" even though no guest from the Washington Times appeared -- the anchor was just discussing a newspaper headline.
4. **ID collisions**: Truncating video names to 20 characters for question IDs caused collisions between videos with similar names (e.g., two ABC News Weekend Reports from different dates), leading to MC options from one question appearing on a different question in the review UI.
5. **Non-English distractors**: The distractor generator (qwen3:8b) occasionally produced Cyrillic text completely unrelated to the video content (5 questions).
6. **Trivial visual questions**: Questions about superficial visual details (hair color, suit color) with no archival research value.
7. **Example contamination**: Specific video content used as examples in the system prompt ("Four Indian Kings' communion silver") was copied verbatim by the model into questions for unrelated videos.

### V3 (current: `qa-data/benchmark/v3/`)

- **1,070 total questions** generated, **925 valid** (86%) from **107 videos**
- Canonical split files on disk:
  - **Train:** `qa-data/raw/qa_v3_train.jsonl` (465 raw) -> `qa-data/benchmark/v3/train_benchmark.jsonl` (464 benchmark = 242 MC + 222 FT)
  - **Val:** `qa-data/raw/qa_v3_val.jsonl` (84 raw) -> `qa-data/benchmark/v3/val_benchmark.jsonl` (80 benchmark = 34 MC + 46 FT)
  - **Test:** `qa-data/raw/qa_v3_test.jsonl` (376 raw) -> `qa-data/benchmark/v3/test_benchmark.jsonl` (373 benchmark = 212 MC + 161 FT)
- Combined benchmark count: **917** questions after distractor generation and formatting
- Generated by `qa-data/generate_qa_v3.py` using **Gemini 2.5 Flash** via Google API
- **Full-context approach**: feeds the entire formatted video index (~33k tokens for a 60-minute broadcast) to Gemini
- Cost: ~$5 for the full dataset (two runs to cover all videos due to API throttling)

**V3 benchmark construction results on the canonical pre-verification benchmark (qwen3:8b + index 3.5):**

| Metric | V1 | V2 | V3 |
|--------|-----|-----|-----|
| MC Accuracy | 62.3% | 62.6% | **75.8%** |
| MC Count | 734 | 385 | 488 |
| Grounding Verified | 53% | 91% | **92%** |
| Multi-modal Citations | -- | 87% | **84%** |
| Blind Correct Rate | 26% | 17% | **26%** (ideal ~25%) |
| Videos | 106 | 98 | 107 |

The +13.5 point MC accuracy improvement from V1/V2 to V3 holds even on hard questions only (blind_wrong): 73.1% vs ~58%. The blind correct rate of 26% is near random chance (25%), confirming distractors are well-calibrated.

**Current local eval results on the full 212-question MC test split:**
- `v3_no_tools_test.jsonl`: **48.1%** (102/212)
- `v3_rag_test.jsonl`: **75.0%** (159/212)
- `v3_random_tools_test.jsonl`: **49.1%** (104/212)
- `v3_policy_answerer_test.jsonl`: **51.9%** (110/212)
- `v3_oracle_test.jsonl`: **97.2%** (206/212)

**Earlier pilot subset results:**
- Earlier 50-question subset runs were used as smoke tests during development.
- They should now be treated as provisional debugging runs rather than the canonical reported results.

**V3 pipeline differences from V2:**
1. **Full-context generation**: The model sees the complete annotated timeline (ASR + OCR + visual descriptions + speakers + chapters), not just isolated entity seeds. This eliminates the front-loading bias (V2: 75% of questions from first 20% of video; V3: questions distributed across full broadcast).
2. **Gemini 2.5 Flash**: 1M token context window easily accommodates full indexes. Stronger instruction following than qwen3:30b, producing consistently valid JSON and well-formed questions.
3. **Chapter-aware**: The model receives the chapter structure and generates questions spanning the full editorial arc.
4. **No seed pipeline**: Eliminates the entity-modality matrix, seed selection, refinement step, and associated complexity. One API call per video replaces ~15 LLM calls.
5. **Natural temporal spread**: Questions distributed across the full broadcast (verified: mean position 21min vs V2's 6min median).

**V3 design decisions:**
- Full-index generation instead of entity-seed generation
- System prompt instructs the model to avoid trivial visual questions, show branding, and use descriptive references instead of naming chyron entities
- Each question includes `source_time` for mapping back to video segments
- Tool-use trajectories are constructed against real indexes rather than simulated evidence
- Canonical benchmark files preserve the unverified V3 split outputs; stricter post-generation quality control can be layered on top of them

**V3 filtering and verification process:**
1. `qa-data/generate_qa_v3.py` generates raw candidate questions from the full annotated broadcast and applies local validation checks for answer presence, source-time locality, premise/evidence consistency, interstitial content, and likely single-source shortcuts.
2. `qa-data/filter_benchmark.py` adds heuristic quality flags such as `trivial_visual`, `single_source_shortcut`, `interstitial_content`, `premise_mismatch`, and `blind_correct`.
3. `qa-data/verify_question_quality.py` runs a stricter accept/reject review against the real index. This was executed on `aristotle` using `qwen3:8b` for the canonical V3 test split.
4. The canonical train/val/test benchmark files remain the source-of-truth split artifacts on disk. Additional local `verified` / `gold` / `rejected` subsets can be materialized for stricter evaluation without overwriting the canonical split files.

**Canonical V3 test artifacts on disk:**
- Raw test split: `qa-data/raw/qa_v3_test.jsonl` (376 questions)
- Benchmark test split with MC/free-text formatting: `qa-data/benchmark/v3/test_benchmark.jsonl` (373 questions)

**Optional stricter local test subsets produced during verification:**
- **332 accepted / 44 rejected** (88.3% acceptance)
- `qa-data/raw/qa_v3_test_verified.jsonl` (376 audit rows with `verification_review`)
- `qa-data/raw/qa_v3_test_gold.jsonl` (332 accepted)
- `qa-data/raw/qa_v3_test_rejected.jsonl` (44 rejected)
- `qa-data/benchmark/v3/test_benchmark_verified.jsonl` (373 audit rows)
- `qa-data/benchmark/v3/test_benchmark_gold.jsonl` (329 accepted benchmark items)
- `qa-data/benchmark/v3/test_benchmark_rejected.jsonl` (44 rejected benchmark items)
- Main reject reasons: `invalid_premise`, `missing_evidence`, `single_source_shortcut`, `low_value`
- Rejects were distributed across many videos rather than concentrated in a single broken source

**Canonical V3 validation artifacts on disk:**
- Raw val split: `qa-data/raw/qa_v3_val.jsonl` (84 questions)
- Benchmark val split with MC/free-text formatting: `qa-data/benchmark/v3/val_benchmark.jsonl` (80 questions)

**Optional stricter local val subsets produced during verification:**
- **78 accepted / 6 rejected** (92.9% acceptance)
- `qa-data/raw/qa_v3_val_verified.jsonl` (84 audit rows with `verification_review`)
- `qa-data/raw/qa_v3_val_gold.jsonl` (78 accepted)
- `qa-data/raw/qa_v3_val_rejected.jsonl` (6 rejected)
- `qa-data/benchmark/v3/val_benchmark_verified.jsonl` (80 audit rows)
- `qa-data/benchmark/v3/val_benchmark_gold.jsonl` (74 accepted benchmark items)
- `qa-data/benchmark/v3/val_benchmark_rejected.jsonl` (6 rejected benchmark items)

**Canonical V3 training artifacts on disk:**
- Raw train split: `qa-data/raw/qa_v3_train.jsonl` (465 questions)
- Benchmark train split with MC/free-text formatting: `qa-data/benchmark/v3/train_benchmark.jsonl` (464 questions)

**Optional stricter local train subsets produced during heuristic filtering:**
- `qa-data/raw/qa_v3_train_tagged.jsonl` (465 rows with `quality_flags`)
- `qa-data/raw/qa_v3_train_filtered.jsonl` (423 retained after hard-fail removal)
- `qa-data/benchmark/v3/train_benchmark_tagged.jsonl` (464 rows with `quality_flags`)
- `qa-data/benchmark/v3/train_benchmark_filtered.jsonl` (404 retained after hard-fail removal)

**Training split filtering policy for the optional local filtered subsets:**
- Train is filtered more permissively than val/test to preserve supervision volume
- Hard-fail items are removed if tagged with:
  - `premise_mismatch`
  - `interstitial_content`
  - `single_source_shortcut`
  - `generator_invalid`
  - `self_flagged`
  - `null_grounding`
- Benchmark-train additionally removes `blind_correct`
- Soft issues such as `trivial_visual` are retained in train but excluded from gold val/test if rejected by verification

### Canonical V3 Question Formats

| Format | Count | File | Scoring |
|--------|-------|------|---------|
| Multiple-choice | 488 | `qa-data/benchmark/v3/benchmark_combined.jsonl` and split benchmark files | Exact match on option letter (A/B/C/D) |
| Free-text | 429 | `qa-data/benchmark/v3/benchmark_combined.jsonl` and split benchmark files | LLM-as-judge (Video-ChatGPT protocol, 5 dimensions) |
| Visual-grounded | 0 | V3 does not use a separate visual-grounded benchmark file | N/A |

### Reasoning types

Questions are tagged by reasoning type (from distractor generation):
- **factual**: Direct information retrieval from a single evidence source
- **cross-modal**: Requires combining information across modalities (e.g., chyron + speech)
- **inferential**: Requires drawing conclusions from evidence
- **comparative**: Requires comparing information across segments/speakers
- **causal**: Requires understanding cause-and-effect relationships

### Visual-grounded questions

V3 does **not** currently maintain a separate `visual_grounded` benchmark file. Visual evidence is represented inside the normal V3 MC/free-text splits through `modalities_required` and the grounding fields. If a dedicated visual-grounded track is reintroduced, it should be documented separately from the canonical V3 split files above.

## Index Versions

| Version | Captions | Location | Notes |
|---------|----------|----------|-------|
| 2.5 baseline | Qwen2.5-VL-3B | `data/video_indexes_25_baseline/` | Preserved for comparison |
| 3.5 (current) | Qwen3.5-9B | `data/video_indexes/` | Production indexes |

Both use the v2 multi-layer schema with independent timeline layers.

### V2 Index Schema

Each index is a JSON file with independent temporal layers. Every item has `id`, `start_ms`, `end_ms` for cross-layer time-overlap queries.

```json
{
  "schema_version": 2,
  "video_id": "cpb-aacip-507-r785h7cp0z",
  "duration_ms": 3549520,
  "layers": {
    "shots":            { "items": [{"id": "shot_0", "start_ms": 0, "end_ms": 47981}] },
    "scenes":           { "items": [{"id": "scene_0", "start_ms": 0, "end_ms": 47981, "label": "Bars"}] },
    "asr":              { "items": [{"id": "asr_0", "start_ms": 48014, "end_ms": 49500, "text": "Good evening..."}] },
    "ocr":              { "items": [{"id": "ocr_0", "start_ms": 947000, "end_ms": 952000,
                                     "text": "JOHN JACOB\nNATIONAL URBAN LEAGUE", "scene_label": "Chyron"}] },
    "visual_captions":  { "items": [{"id": "vcap_0", "start_ms": 0, "end_ms": 47981,
                                     "text": "The video frame shows a test pattern..."}] },
    "speakers":         { "items": [{"id": "spk_0", "start_ms": 69000, "end_ms": 126380,
                                     "speaker_id": "speaker_7", "speaker_name": "Jim Lehrer",
                                     "text": "Good evening from Washington..."}] },
    "chapters":         { "items": [{"id": "ch_0", "start_ms": 0, "end_ms": 434000, "title": "Introduction"},
                                    {"id": "ch_1", "start_ms": 434000, "end_ms": 1745000, "title": "Birth Control Pill"}] },
    "entities":         { "items": [{"id": "ent_0", "start_ms": 69000, "end_ms": 126380,
                                     "text": "Jim Lehrer", "type": "PERSON",
                                     "source_layer": "asr", "grounding": "American journalist..."}] },
    "relations":        { "items": [{"id": "rel_0", "subject": "Jim Lehrer", "predicate": "identified_by",
                                     "object": "chyron", "source_layers": ["speakers", "ocr"]}] },
    "credits":          { "items": [{"id": "cred_0", "entries": [{"name": "Jim Lehrer", "role": "Anchor"}]}] }
  }
}
```

**Key design properties:**
- **Independent temporal boundaries**: Each layer maintains its own timeline. A 3-second chyron is not forced into a 10-second shot boundary.
- **Cross-layer provenance**: Entities track `source_layer` (was it extracted from ASR, OCR, or visual captions?). Relations track `source_layers` (plural) for cross-modal links.
- **Independently rebuildable**: Each layer has its own `source` and `indexed_at`. Rebuilding captions doesn't require re-running ASR.

### Structural Timestamp Encoding

Tool outputs encode temporal position with structural context rather than raw milliseconds:

```
[15:43-16:02 | 73% | Birth Control Pill Discussion (+15s)]
```

Components:
- **Human-readable time**: `MM:SS-MM:SS` for interpretability
- **Video position**: percentage through the broadcast (teaches structural patterns like "credits appear at 95%+")
- **Chapter context**: current topic + offset within chapter (teaches "chyrons appear in the first 30s of a chapter")

This is analogous to positional encoding (like RoPE) for video structure -- the model learns position-dependent patterns:
- 0-5%: Bars, slate, intro
- 5-15%: Headlines summary
- 15-85%: Main segments with chyrons at segment openings
- 85-95%: Final essay/commentary
- 95-100%: Credits, funding acknowledgments

### Agent Tool Interface

The agent has 7 tools for querying video indexes, executed against real index data:

| Tool | Purpose | Returns |
|------|---------|---------|
| `search_asr(video_id, query)` | Find spoken content by keyword | Matching ASR segments with structural timestamps |
| `search_ocr(video_id, query)` | Find on-screen text (chyrons, titles) | Matching OCR items with scene labels |
| `search_entities(video_id, query)` | Find named entities | Entity type, source layer, grounding |
| `get_visual(video_id, start_ms, end_ms)` | Visual descriptions at a time | What's shown on screen |
| `get_speakers(video_id, start_ms, end_ms)` | Speaker attribution at a time | Speaker names and text |
| `get_chapter(video_id, time_ms)` | Topic at a time | Chapter title and duration |
| `browse_timeline(video_id, start_ms, end_ms)` | All layers at a time | Cross-layer snapshot |

The tools are shared across trajectory generation, ReAct eval, and SFT eval. They execute against the real v2 index, not simulated data.

## Pipeline Tools

| Step | Tool | Output |
|------|------|--------|
| Scene detection | app-swt-detection | scenes layer (Bars, Slate, Chyron, Credits, Other-text) |
| Shot detection | app-transnet-wrapper | shots layer (frame-level boundaries) |
| ASR | Parakeet TDT 0.6B | asr layer (sentence-level transcripts) |
| Visual captioning | simple_captioner_35.py (Qwen3.5-9B) | visual_captions layer |
| OCR/transcription | app-qwen3vl-captioner (swt_transcription config) | ocr layer |
| Speaker diarization | pyannote + LLM | speakers + chapters layers |
| Credits extraction | app-credits-ocr (Qwen3-VL) | credits layer |
| NER | SpaCy en_core_web_sm | entities layer |
| Entity resolution | Substring merging, OCR-preferred | entities layer (deduped) |
| SVO relations | SpaCy dependency parsing | relations layer |
| Entity grounding | Wikidata + Wikipedia | entities layer (grounding field) |
| CLIP embeddings | clip-ViT-L-14 via sentence-transformers | data/clip_embeddings/ (FAISS) |

## Evaluation Results

Model: qwen3:8b via Ollama. 1,602 questions (734 MC + 868 free-text). MC exact-match accuracy reported below. Free-text scoring (LLM-as-judge) pending.

### Index vs No-tools

| Condition | MC Accuracy | Delta vs No-tools | Time |
|-----------|-------------|-------------------|------|
| **Index 3.5 (full)** | **62.3%** | **+16.7** | 144 min |
| Index 2.5 (full) | 48.6% | +3.0 | 152 min |
| No-tools baseline | 45.6% | -- | 144 min |

The structured index with Qwen 3.5 captions gives a **+16.7 point** improvement over the blind LLM baseline. The 2.5 captions (which suffered from VLM hallucinations) barely help (+3.0).

### Ablation: Layer Contribution

| Condition | MC Accuracy | Delta vs Full (62.3%) | Interpretation |
|-----------|-------------|----------------------|----------------|
| no_ocr | 60.2% | -2.1 | OCR contributes modestly |
| no_entities | 59.9% | -2.4 | Entities add minimal marginal value |
| no_relations | 59.7% | -2.6 | Relations add minimal marginal value |
| no_visual | 53.8% | -8.5 | Visual captions matter |
| no_asr | 48.0% | -14.3 | ASR is the most valuable layer |
| asr_only | 40.9% | -21.4 | ASR alone underperforms no-tools |
| visual_only | 23.3% | -39.0 | Captions alone far insufficient |

**Key findings:**
- **ASR is dominant**: Removing ASR drops accuracy by 14.3 points, nearly back to the no-tools baseline. Speech carries the primary information in broadcast news.
- **Visual captions contribute significantly**: -8.5 points without them. The Qwen 3.5 upgrade (from 2.5) is what made this layer useful.
- **OCR adds modest value**: Only -2.1 points. On-screen text (chyrons, slates) provides supporting evidence but is rarely the sole answer source.
- **Entities/relations are mostly redundant**: -2.4 and -2.6 respectively, suggesting NER/SVO layers repeat information already present in raw ASR/OCR text.
- **Single-layer isolation hurts**: ASR-only (40.9%) underperforms no-tools (45.6%), and visual-only (23.3%) is worst. Providing incomplete evidence is worse than letting the model reason from parametric knowledge, likely because the model over-anchors on sparse context.
- **Caption quality matters more than having captions**: Index 2.5 (48.6%) vs Index 3.5 (62.3%) is a +13.7 point gap, showing that bad captions hurt more than help.

### Evidence Redundancy

Analysis of 1,284 questions with evidence paths across index layers:

| Answer reachable via | Count | % |
|----------------------|-------|---|
| 1 layer | 340 | 26% |
| 2 layers | 409 | 32% |
| 3 layers | 238 | 19% |
| 4 layers | 188 | 15% |
| 5+ layers | 109 | 8% |

74% of answers are reachable through 2+ independent layers, providing multiple tool-use trajectories per question for SFT training.

### Retrieval Limitations (Current Eval)

The current index eval uses **single-hop keyword retrieval** (`search_index()` in `run_index_qa.py`):
1. Tokenize question, remove stopwords
2. Score each shot by bag-of-words overlap with overlapping layer text
3. Return top-5 shots, stuff all their text into the prompt

This means the 62.3% accuracy is a **lower bound** on what the index can provide. The retrieval cannot:
- **Entity-centric queries**: "What did Jim Lehrer say about X?" requires finding Jim Lehrer in the speakers layer, then pulling all his turns, then filtering for topic X
- **Temporal reasoning**: "What happened after the interview?" requires chapter awareness and sequential navigation
- **Cross-layer joins**: "Who was speaking when this chyron appeared?" requires time-overlap joins across layers
- **Follow-up expansion**: Finding a keyword match, then expanding to gather surrounding context

A multi-hop retrieval strategy (entity expansion, chapter context, cross-layer joins) would likely push accuracy higher. The gap between single-hop and multi-hop retrieval quantifies the value of smarter index querying vs. the full tool-use agent approach.

### Retrieval Quality (scored by `eval/score_retrieval.py`)

Retrieval is scored independently from answer quality using ground truth `source_segment_times`:
- **Hit rate**: did any retrieved segment overlap the ground truth time range?
- **Temporal IoU**: intersection over union of retrieved vs ground truth
- **Precision**: what fraction of retrieved segments are relevant?

| Benchmark | Model | Retrieval Hit Rate | Mean IoU | MC Accuracy |
|-----------|-------|--------------------|----------|-------------|
| V1 | qwen3:8b | 48.2% | 0.464 | 62.3% |
| V1 | qwen3.5:9b | 36.5% | 0.351 | 61.0% |
| V2 | qwen3:8b | 31.9% | 0.248 | 70.1% |

**Key findings:**
- **Retrieval is the bottleneck**: Keyword search only finds the right segment 31-48% of the time. Models answer correctly despite bad retrieval through parametric knowledge or partial matches.
- **V2 questions are harder to retrieve for**: 31.9% vs 48.2% hit rate, because v2 uses descriptive references instead of entity names. Higher answer accuracy (70.1%) despite worse retrieval confirms v2 questions are better designed.
- **This quantifies the value of smarter retrieval**: Going from 48% to 90%+ hit rate (via multi-hop retrieval or tool-use agent) should significantly boost answer accuracy.

### Evaluation Decomposition

Retrieval and answer quality are scored independently, enabling a 2x2 analysis:

|  | Oracle retrieval | System retrieval |
|--|-----------------|-----------------|
| **Answer accuracy** | Upper bound (comprehension only) | Realistic (full pipeline) |
| **Retrieval hit rate** | 100% by definition | 31-48% (keyword search) |

The gap between oracle and system retrieval is the cost of bad retrieval, and the value the trained agent should close.

### Scoring Systems

| Scoring | Evaluates | Dimensions | Script |
|---------|----------|------------|--------|
| **MC accuracy** | Model answers (MC) | Exact letter match | `score_predictions.py --mode mc-exact` |
| **Answer quality** | Model answers (free-text) | correctness, completeness, relevance, cross_modal, specificity | `score_predictions.py --mode llm-judge` |
| **Question quality** | Questions themselves | cognitive_level, naturalness, groundedness, cross_modal_need | `qa-data/score_question_quality.py` |
| **Retrieval quality** | Segment retrieval | hit rate, temporal IoU, precision | `score_retrieval.py` |

### Agent Evaluation Results (V2 Benchmark)

All conditions use qwen3:8b. V2 clean benchmark = 385 MC + 320 free-text unless noted.

| Condition | MC Accuracy | Avg Tools | n (MC) | Notes |
|-----------|-------------|-----------|--------|-------|
| **RAG keyword** | **62.6%** | n/a | 385 | Pre-retrieved top-5 segments stuffed into prompt. Best accuracy. |
| No-tools baseline | 45.6% | 0 | 734 (v1) | Parametric knowledge only, no index access. |
| RAG multihop | 43.6% | n/a | 385 | Entity expansion + cross-layer scoring. Over-retrieval floods context. |
| Prompted ReAct | 35.6% | 1.0 | 385 | Free-text ReAct loop. Model makes 1 tool call then answers. |
| LangGraph agent | 30.9% | 1.5 | 385 | Structured tool calling via LangGraph. 38% of questions get zero tool calls (model sees MC options and guesses). |
| SFT agent (v2, two-stage) | 17.5% | 2.2 | 40 | **Small sample.** Trained on non-interactive trajectories with hallucinated timestamps. Two-stage: SFT model gathers evidence, base model answers MC. Tool selection learned but evidence quality poor. |
| SFT agent v3 | -- | -- | -- | **Training.** Interactive trajectories + structural timestamps. Expected to fix evidence quality. |

**Key findings:**
- **Prompted tool use underperforms RAG** regardless of framework (ReAct or LangGraph). The model doesn't learn multi-step evidence gathering from prompting alone. This gap justifies SFT training.
- **SFT teaches tool selection**: The fine-tuned model uses 2.2 tools on average with diverse tool choice (ASR, OCR, visual, speakers) vs 1.0-1.5 for prompted models. But v2 training data had hallucinated timestamps, causing poor evidence retrieval.
- **V3 trajectories address this**: Interactive generation ensures tool arguments are grounded in real observations. Structural timestamps encode video position (percentage, chapter context) instead of raw milliseconds.

### Policy + Answerer Architecture (Final Results)

The agent is split into two components:
- **Policy model** (SFT Qwen3.5-9B): decides which CLAMS tools to run. Only makes tool selection decisions.
- **Answerer model** (base qwen3:8b via Ollama): given the question + gathered evidence, produces the answer. No fine-tuning needed.

This separation was motivated by the finding that a single model trying to both select tools AND answer questions achieves only 26% MC (random chance). Separating the roles reveals that the answerer is not the bottleneck -- evidence quality is.

**Full ablation on held-out test set (43 videos, never seen during training):**

| Condition | MC Accuracy | Delta vs No-tools | What it measures |
|-----------|------------|-------------------|-----------------|
| No-tools baseline | 48.1% (102/212) | -- | Parametric knowledge only |
| Random tools + Answerer | 49.1% (104/212) | +1.0 | Whether arbitrary evidence helps |
| **Policy + Answerer (SFT-only)** | **51.9% (110/212)** | **+3.8** | **Learned tool selection** |
| RAG baseline | 75.0% (159/212) | +26.9 | Keyword retrieval (top-5 segments) |
| Oracle + Answerer | 97.2% (206/212) | +49.1 | Perfect ground-truth evidence |
| GRPO 1 epoch | 45.8% (97/212) | -2.3 | Early RL with misaligned reward |
| GRPO 3 epochs | 48.1% (102/212) | +0.0 | RL recovered to no-tools level, still below SFT |

**Key findings:**
- **The answerer doesn't need training**: base qwen3:8b achieves 96% with perfect evidence. The comprehension capability is already there.
- **The entire value is in evidence retrieval**: the 48% -> 96% gap (48 points) is purely about what evidence the model receives.
- **Learned tool selection still beats random**: 51.9% vs 49.1% (+2.8 points), but the margin is smaller on the full test set than it looked on the earlier 50-question pilot.
- **Naive GRPO efficiency reward hurt the policy**: both GRPO runs underperformed the SFT policy, showing that reducing tool use is not the same as finding better evidence.
- **Gap to close**: the current SFT policy (51.9%) is still far below RAG (75.0%) and oracle (97.2%). The main bottleneck remains evidence retrieval.
- **The research problem is well-defined**: improve retrieval quality while keeping tool usage efficient; the answerer is not the limiting factor.

**Policy tool usage:**
- Average 1.7 tools per question
- Distribution: run_asr (50), extract_text (14), detect_text_scenes (9), caption_frame (9), identify_speakers (5)
- Policy is conservative -- calling more tools with better targeting would likely improve results

### Eval Progression

| Stage | Accuracy | Approach | Finding |
|-------|----------|----------|---------|
| 1. No-tools | 48.1% | Parametric knowledge | Baseline |
| 2. Prompted ReAct | 35-50% | Free-text tool prompting | Worse than no-tools -- model can't learn tool use from prompts alone |
| 3. LangGraph agent | 30.9% | Structured tool calling | Framework doesn't help without training |
| 4. SFT (single model) | 26% | One model does both tool use + answering | Random chance -- model can't do both |
| 5. SFT (policy + answerer) | **51.9%** | Separate tool selection from comprehension | **+3.8 over no-tools on full test set** |
| 6. GRPO (count/cost pressure) | 45.8-48.1% | RL over tool-use policy | Misaligned reward taught tool avoidance |
| 7. RAG baseline | 75% | Keyword retrieval + same answerer | Upper bound for retrieval approach |
| 8. Oracle | 97.2% | Perfect evidence + same answerer | Ceiling -- almost all questions answerable |

### Remaining Work

| Task | Purpose | Status |
|------|---------|--------|
| Richer index variants (Whisper + multi-model captions) | Give policy real quality differences across tools/models | In progress / running on `aristotle` |
| Trajectory refresh for new tool API | Align training data with model/task-mode-aware tool space | Needed next |
| GRPO with correctness-first reward | Teach policy to find answers before penalizing tool cost | Next major experiment |
| Rejection sampling | SFT on successful rollouts only | Candidate follow-up |
| Process supervision | Score each tool call individually | Designed, not implemented |
| VLM direct baseline | Compare against raw frame approach | Ready, not run |

## Running the Evaluation

```bash
ssh aristotle
cd ~/clams-agent-prototype

# Full eval (all conditions)
bash eval/run_full_eval.sh

# Single condition
python eval/run_index_qa.py --benchmark qa-data/benchmark/benchmark.jsonl \
  --output eval/results/my_test.jsonl --model qwen3:8b

# Ablation
python eval/run_ablation.py --benchmark qa-data/benchmark/benchmark.jsonl \
  --ablation no_asr --output eval/results/ablation_no_asr.jsonl --model qwen3:8b

# Score results
python eval/score_predictions.py --predictions eval/results/my_test.jsonl \
  --benchmark qa-data/benchmark/benchmark.jsonl --mode mc-exact

# Incremental video update (recaption + rebuild + QA)
python scripts/update_video.py <video_id> --captioner simple35
```

## Viewer

```bash
# Local viewer (requires indexes + thumbnails synced locally)
python serve_viewer.py  # runs on port 8769

# Features: timeline, transcript, entities, search, visual search (CLIP), QA
# Visual search requires CLIP embeddings in data/clip_embeddings/
```

## Key Findings

1. **Structured index provides +16.7 point MC accuracy gain** over blind LLM baseline (45.6% to 62.3%), demonstrating that tool-orchestrated indexing enables small LLMs to answer cross-modal questions about long-form video that they cannot answer from parametric knowledge alone.

2. **Caption model quality is critical**: Qwen 3.5 (9B) captions yield +13.7 points over Qwen 2.5 (3B) captions (62.3% vs 48.6%). The 2.5 captions barely outperform the no-tools baseline (+3.0), meaning bad visual descriptions actively confuse the answering model. Caption quality propagates directly to downstream QA.

3. **ASR is the most valuable layer**: Removing ASR drops accuracy by 14.3 points (62.3% to 48.0%), nearly back to the no-tools baseline. For broadcast news, spoken narration carries the primary information.

4. **Visual captions provide the second-largest contribution**: -8.5 points without them (62.3% to 53.8%). This effect only emerged after upgrading to Qwen 3.5 captions.

5. **OCR has modest marginal value**: Only -2.1 points without OCR (62.3% to 60.2%). On-screen text provides supporting evidence but is rarely the sole answer source. This suggests OCR's value is primarily confirmatory (verifying what ASR already provides).

6. **74% of answers have redundant evidence paths**: Most answers can be reached through 2+ independent layers, enabling multi-trajectory SFT training where the agent learns both efficient and exhaustive search strategies.

7. **Multi-layer indexing outperforms flat segmentation**: The v2 schema preserves independent temporal boundaries per tool, enabling cross-layer queries (e.g., "who was speaking when this chyron was shown") that flat shot-aligned indexes cannot express.

## Training Data Pipeline

The indexes and QA pairs are used to generate SFT training data for the agent:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| QA generation | `qa-data/generate_qa_v3.py` | Full video indexes | Questions + answers + grounding (via Gemini 2.5 Flash) |
| Expert trajectories | `training_data/construct_trajectories.py` | QA grounding + indexes | 872 verified ReAct traces with real tool execution |
| LLM trajectories (alt) | `training_data/generate_trajectories_v2.py` | QA pairs + indexes | Interactive LLM-generated traces (lower quality) |
| Evidence paths | `training_data/find_evidence_paths.py` | QA pairs + indexes | Per-question evidence layer mapping |
| SFT data preparation | `training_data/prepare_sft_from_trajectories.py` | Trajectories + benchmark | ShareGPT formatted training examples |
| SFT training | `training_data/run_sft.py` | ShareGPT data | LoRA adapter for Qwen3-8B |

### Trajectory Generation

Three approaches were explored for generating tool-use trajectories:

**1. LLM-interactive (early approach, superseded)**
The LLM (qwen3:8b) iteratively calls tools, receives real observations, and decides next steps. Problem: 30% verify rate, 8+ hours to generate, and the LLM often uses hardcoded timestamps rather than values from observations. The resulting trajectories record the baseline model's behavior, which is what we're trying to improve on -- training on these reinforces poor search patterns.

**2. One-shot LLM generation (v1, superseded)**
The LLM generates the entire trajectory in one call, then we replace observations with real ones. Problem: tool arguments (especially time ranges) are hallucinated since the LLM never sees actual results.

**3. Expert scripted from Gemini grounding (current: `construct_trajectories.py`)**
Uses Gemini's grounding data (speech excerpts, visual references, OCR text) to deterministically construct optimal trajectories. For each question:
1. Extract keywords from the speech excerpt, search ASR
2. If OCR evidence exists, search OCR with those keywords
3. If visual evidence exists, get_visual at the timestamp found in step 1
4. Combine evidence into the answer

This produces *expert* trajectories showing the ideal tool-use pattern, rather than recording a model's search behavior.

| Approach | Verified | Rate | Time | Quality |
|----------|----------|------|------|---------|
| LLM interactive | ~280 | 30% | 8 hours | Records baseline behavior |
| One-shot LLM | 564 | 35% | 10 hours | Hallucinated timestamps |
| **Expert scripted** | **872** | **94%** | **30 seconds** | **Optimal tool selection** |

```
User: What organization does the guest discussing unemployment represent?
Assistant: <think>I need to find information about this topic. Let me search the speech
           transcripts.</think>
           <tool>search_asr(video_id="cpb-aacip-507-r785h7cp0z", query="unemployment")</tool>
Tool:      [15:43-16:02 | 73% | Birth Control Pill Discussion (+15s)] We believe that all 
           the evidence points... unemployment is going down...
Assistant: <think>I found relevant speech content. Now let me check on-screen text around
           that time for identification.</think>
           <tool>search_ocr(video_id="cpb-aacip-507-r785h7cp0z", query="league")</tool>
Tool:      [15:47-15:52 | 73% | Birth Control Pill Discussion (+19s)] (Chyron) JOHN JACOB
           NATIONAL URBAN LEAGUE
Assistant: <think>Based on the evidence: The chyron identifies the speaker as from the
           National Urban League.</think>
           <answer>National Urban League</answer>
```

Key properties:
- **Real tool execution**: All observations come from the actual index
- **Structural timestamps**: Encode broadcast position (%, chapter context) instead of raw milliseconds
- **Expert-level tool selection**: Trajectories show the optimal search pattern derived from ground-truth evidence, not a model's guesswork
- **94% verify rate**: vs 30% for LLM-interactive generation
- **Grounding-driven**: Each tool call is derived from Gemini's evidence mapping, ensuring the trajectory finds the right evidence

### Training Pipeline

Two-phase training following NVIDIA ToolOrchestra (arXiv 2511.21689):

**Phase 1: SFT (Supervised Fine-Tuning)**
Teaches the model the *format* of tool-use interaction.
- **Model**: Qwen3.5-9B with LoRA (r=16, alpha=16), using native tool calling format
- **Data**: 1,728 diverse expert trajectories across 5 strategies (topic-first, chapter-nav, identity-first, speaker-track, entity-lookup)
- **Format**: Qwen3.5 native `<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>` format, NOT custom tags
- **Training**: 1 epoch, batch size 8 (grad accum), lr=1e-4, bf16
- **Duration**: ~31 minutes on one A6000
- **Loss**: 3.27 -> 0.49 (converges quickly in 1 epoch)

**Phase 2: GRPO (Group Relative Policy Optimization)**
Teaches the model *what works* -- efficiency, tool selection, stopping criteria.
- Generate multiple rollouts per question, score with multi-objective reward
- **Reward**: correctness (gates everything) * efficiency (fewer tools = better) + coverage (diverse tools = bonus)
- The model learns from its own successes and failures, not just expert demonstrations
- Following NVIDIA's finding: SFT teaches format, RL teaches when NOT to call a tool

### Preference-Conditioned Tool Selection

The agent can be conditioned on user preferences for speed vs accuracy vs cost. This applies at two levels:

**Level 1: Index query tools** (searching existing indexes)

| Tool | Cost | When to use |
|------|------|-------------|
| search_asr | Low | Fast keyword search in speech |
| search_ocr | Low | Fast keyword search in on-screen text |
| get_chapter | Very low | Single lookup for topic context |
| get_visual | Medium | Returns long captions, use when visual detail needed |
| browse_timeline | High | Cross-layer scan, use sparingly |

**Level 2: CLAMS pipeline tools** (building indexes on new videos)

| Tool + Config | Accuracy | Speed | GPU Cost |
|---------------|----------|-------|----------|
| Parakeet TDT 0.6B (ASR) | 3/5 | 5/5 | Low |
| Whisper Large v3 (ASR) | 4/5 | 3/5 | High |
| SWT detection (scene labels) | 2/5 | 5/5 | Low |
| VLM captioner "is there text?" (scene labels) | 4/5 | 2/5 | High |
| Qwen3.5-9B captioner 1fps | 4/5 | 2/5 | High |
| Qwen3.5-9B captioner 0.2fps | 3/5 | 4/5 | Medium |
| pyannote diarization | 3/5 | 4/5 | Medium |
| SpaCy NER | 3/5 | 5/5 | None |
| Qwen3-VL OCR (chyrons only) | 4/5 | 4/5 | Medium |
| Qwen3-VL OCR (all frames) | 5/5 | 1/5 | Very High |
| VLM-as-judge verification | 5/5 | 1/5 | High (but confirmatory) |

The preference vector `[speed, cost, accuracy]` controls which tools and configurations the agent selects:
- **"Process in 10 minutes"**: Parakeet + SWT + SpaCy. Skip visual captioning.
- **"Maximum accuracy"**: Whisper + VLM captioner 1fps + full OCR + VLM verification.
- **"Balanced"**: Parakeet + VLM captioner 0.2fps + OCR on text frames.

During GRPO training, different preference vectors are sampled per batch. The model learns to adapt its tool selection based on the stated preference. At inference, the user specifies their constraint and the agent selects the appropriate pipeline.

This connects to the existing `evaluation_rag.py` which provides empirical performance data (CER, WER, F1) for each CLAMS tool, and `agent.py`'s GPU-aware tool filtering.

### Future Work: External Knowledge Tools

The current tool set queries pre-built video indexes. Future extensions:
- **query_wikidata(entity, property)**: Look up external facts about entities found in the video
- Questions requiring combining video evidence with external knowledge (e.g., "What legislation did this senator later sponsor?")
- Creates L4-Analyze questions that can't be answered from the video alone
- Entity grounding field in v2 indexes already contains Wikidata descriptions as a starting point

### Question Taxonomy

Questions are categorized along two axes:

**Cognitive levels** (rows) -- correlate with agent workflow complexity:
- **L1 Identify**: Single lookup (e.g., "What is the air date on the slate?")
- **L2 Retrieve**: Find specific content (e.g., "What does the Washington Post guest discuss?")
- **L3 Integrate**: Combine sources (e.g., "What organization does the guest discussing unemployment represent?")
- **L4 Analyze**: Cross-segment reasoning (e.g., "How do the two medical experts differ on risk?")

**Information need types** (columns) -- reflect archival research purposes:
- **Cataloging**: Production metadata (dates, directors, credits)
- **Factual**: Specific content retrieval (names, quotes, events)
- **Subject**: Topic identification and navigation (chapter topics, story sequences)
- **Interpretive**: Contextual understanding (speaker roles, organizational affiliations, editorial framing)

Example questions for each cell are in `qa-data/taxonomy_examples.json`, all grounded in real video index content.

## Comparison Models

The thesis compares the **trained agent** against other video understanding approaches:

| Model | Type | Approach | Script | Status |
|-------|------|----------|--------|--------|
| **Qwen3-VL (direct)** | VLM | Raw frames (1/5/10) +/- ASR | `eval/run_vlm_direct.py` | Ready |
| **VideoAgent** | LLM agent | Iterative frame selection + reasoning | Planned | To implement |
| **LongVideo-R1** | Multi-stage VLM | Reasoning + captioning + QA pipeline | `eval/run_longvideo_r1.py` | Installed on aristotle |
| **Index + VLM combined** | Hybrid | Text retrieval + visual verification | `eval/run_index_vlm_combined.py` | Ready |
| **Deep Video Discovery** | Hybrid | Microsoft paradigm: retrieve then verify | (same as above) | Ready |

### Key comparisons

1. **Trained agent vs. VideoAgent**: Both are agent-based approaches to long-form video. VideoAgent does on-the-fly frame selection; our agent orchestrates structured tool pipelines. Tests whether persistent index construction outperforms iterative frame sampling.

2. **Trained agent vs. LongVideo-R1**: Multi-stage VLM pipeline (reasoning model + caption model + QA model) vs. tool-orchestrated agent. Tests whether explicit tool decomposition outperforms implicit multi-model reasoning.

3. **Trained agent vs. direct VLM**: Can a trained agent with tool access outperform a larger VLM given raw frames? Tests whether tool orchestration compensates for model size.

4. **Agent with vs. without pre-built index**: The critical generalization test. The agent is trained on tool trajectories derived from indexed videos, then evaluated on held-out videos where it must build understanding from scratch using tools.

5. **Modality ablation**: Which pipeline tools contribute most to training signal? Informs tool selection strategy for the agent.

6. **Caption quality ablation**: 2.5 vs 3.5 captions, isolating the contribution of visual description quality to both index QA and downstream trajectory quality.

### Benchmark context

Our benchmark targets **long-form broadcast content (30-60+ min)** requiring cross-modal reasoning. This differs from existing benchmarks:
- **EgoSchema, NExT-QA**: Short clips (3-5 min), single-camera, activity recognition
- **MLVU, LVBench**: Longer videos but primarily entertainment/narrative
- **Video-MME**: Multi-modal but uniform sampling, not tool-orchestrated

Our questions specifically require combining information across modalities (e.g., chyron text identifying a speaker whose words come from ASR), which cannot be answered from any single modality alone.

## File Locations on Aristotle

```
~/clams-agent-prototype/
  data/video_indexes/           # v2 layered indexes (3.5 captions)
  data/video_indexes_25_baseline/  # v2 indexes (2.5 captions, preserved)
  data/combined_outputs/        # Combined MMIF files
  data/clip_embeddings/         # FAISS indexes + metadata
  data/thumbnails_5s/           # Thumbnails every 5s per video
  data/diarization/             # Diarization JSON outputs
  data/recaptioned/             # 2.5 caption MMIFs
  data/recaptioned_35/          # 3.5 caption MMIFs
  qa-data/benchmark/            # Final benchmark (MC + free-text)
  qa-data/raw/                  # Raw QA + baseline backup
  eval/results/                 # Evaluation outputs + timing
  eval/results_old/             # Archived old results
```
