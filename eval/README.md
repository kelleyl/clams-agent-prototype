# CLAMS-Agent: Tool-Orchestrated Video Understanding

## Thesis Overview

The goal is to build and evaluate **video understanding agents** that answer questions about long-form archival broadcast content with explicit evidence provenance. The repo now separates two experimental regimes:

| Regime | Initial State | Main Question |
|--------|---------------|---------------|
| Warm-index archive QA | A multimodal CLAMS-style index already exists. | Can the agent query the right evidence layers, localize support, and answer with grounding? |
| Cold-cache artifact orchestration | The visible cache starts empty, approximating a new raw video. | Can the agent decide which artifacts to create, read, search, reuse, or skip under cost constraints? |

Most current V4.1 scores are warm-index archive QA results. They should not be described as raw-video tool execution from scratch.

## Canonical Evaluation Workflow

For the current V3/V4.1 setup, the canonical evaluation flow is:

1. Generate prediction artifacts with [run_policy_answerer_eval.py](/Users/kelleylynch/clams/clams-agent-prototype/eval/run_policy_answerer_eval.py)
2. Score those prediction artifacts with [score_predictions.py](/Users/kelleylynch/clams/clams-agent-prototype/eval/score_predictions.py)

This separation matters because:
- `run_policy_answerer_eval.py` is the prediction-generation step
- inline metrics from that script are MC-only previews
- free-text questions are scored post-hoc by `score_predictions.py`

Recommended current files:
- benchmark input: `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` for current V4.1 work, or `qa-data/benchmark/v3/test_benchmark_gold.jsonl` for historical V3 comparisons
- prediction output: `eval/results/*.jsonl`
- final scoring: `python eval/score_predictions.py --predictions ... --benchmark ...`

## Current V4.1 Result Snapshot

The canonical run history now lives in [docs/RUN_LEDGER.md](/Users/kelleylynch/clams/clams-agent-prototype/docs/RUN_LEDGER.md). Each run there records status, inputs, model stack, scores, supersession, known issues, and representative errors.

### Current Reportable Results

| Run | Status | File | MC Accuracy | Notes |
|-----|--------|------|-------------|-------|
| Matched-text oracle | current | `eval/results/v4_1_fixed_oracle_matched_text_test.jsonl` | 194/200 = 97.0% | Upper-bound evidence answerability |
| Unified current-tool oracle | current | `eval/results/v4_1_unified_oracle_current_tool_test_v2.jsonl` | 184/200 = 92.0% | Tool-visible oracle after layer-name unification, direct verified-layer reads, and centered truncation |
| Base `Qwen/Qwen3.5-9B` warm-index policy | current baseline | Aristotle `eval/results/v4_1_base_model_shard*.jsonl` | 142/200 = 71.0% | No LoRA adapter; queries the prebuilt index with the same tool schemas and answerer |
| No-tools parametric control | current control | Aristotle `eval/results/v4_1_no_tools_baseline.jsonl` | 117/200 = 58.5% | Correct answers without video evidence; useful benchmark-artifact / grounding control |
| Base `Qwen/Qwen3.5-9B` simulated cold-cache diagnostic | current diagnostic | Aristotle `eval/results/v4_1_base_cold_cache_isolated_test.jsonl` | 135/200 = 67.5% | Uses explicit artifact/cache schema; source index remains a read-only simulator fixture |
| SFT policy, old V4 recovery adapter, max turns 3 | historical baseline | `eval/results/v4_1_sft_recovery_alltools_localbase_test_maxturns3.jsonl` | 133/200 = 66.5% | Valid historical baseline, but worse than base and trained before fixed-context/unified-layer trajectories |
| Unified V4.1 SFT v2 policy | negative SFT result | Aristotle `eval/results/v4_1_unified_v2_policy_shard*.jsonl` | 124/200 = 62.0% | Corrected native-tool SFT still underperformed the base policy |

Superseded runs remain available for provenance:

| Run | File | Old Score | Superseded Because |
|-----|------|-----------|--------------------|
| Old matched-text oracle | `eval/results/v4_1_oracle_matched_text_test.jsonl` | 177/200 = 88.5% | Evidence verifier stored the first 800 chars of long windows and could truncate away the answer-bearing sentence. |
| Old current-tool oracle | `eval/results/v4_1_oracle_current_tool_test.jsonl` | 171/200 = 85.5% | Used pre-fix evidence context and old replay behavior. |
| Fixed current-tool oracle | `eval/results/v4_1_fixed_oracle_current_tool_test.jsonl` | 170/200 = 85.0% | Fixed evidence text, but still had layer mismatch, timestamp drift, and less robust truncation. |
| Unified current-tool oracle v1 | `eval/results/v4_1_unified_oracle_current_tool_test.jsonl` | 180/200 = 90.0% | Replaced by v2 centered truncation. |
| SFT policy, old V4 recovery adapter, max turns 5 | `eval/results/v4_1_sft_recovery_alltools_localbase_test.jsonl` | 132/200 = 66.0% | Max turns 3 was slightly more accurate and used fewer tools. |

### Layer and Oracle Fixes

On 2026-04-20, the video index layer naming convention was unified:

- `visual_captions` became `caption_qwen3vl-8b_general_scene`
- `ocr` became `caption_qwen3vl-8b_text_focus`
- `qwen3vl-8b` means Qwen3-VL-8B-Instruct, distinct from `qwen-8b` / Qwen3.5-9B variant-generation layers

Related code changes:

- `scripts/migrate_layer_names.py` performs the one-time index migration.
- `simulate_tool_output()` no longer silently falls back from missing requested variant layers to canonical `visual_captions` / `ocr` content.
- `eval/run_evidence_span_oracle.py` can read directly from the verified layer/window for oracle diagnostics, then truncate long evidence around answer-bearing terms.
- `utils/layer_utils.py` centralizes layer discovery for OCR/text-focus and visual-caption layers.

The current-tool oracle improved from 85.0% to 92.0% after these fixes. The remaining 5 point gap to the 97.0% matched-text oracle is mostly from multi-item evidence windows, visual/OCR anchoring, and a few QA-quality cases such as near-duplicate distractors.

### Evidence-Context Fix

The verifier now stores matched evidence context centered around the actual matched terms, instead of only the first 800 characters of a verified window.

Fixed benchmark directory: `qa-data/benchmark/v4_1_fixed_context/`

Regeneration stats:

- Train: `439` accepted, `25` rejected
- Val: `78` accepted, `2` rejected
- Test: `350` accepted, `23` rejected
- Combined: `867` accepted, `50` rejected
- Average source-to-stored matched-text recall on test improved from `0.733` to `0.935`
- Low-recall test spans (`<=0.5`) dropped from `216` to `38`

## Tool Execution Modes

`run_policy_answerer_eval.py` separates the source index from the visible tool/cache state.

| Mode | Meaning | Use |
|------|---------|-----|
| `prebuilt_index` | Warm-index archive QA. Tool calls directly delegate to `simulate_tool_output()` over the full read-only index. `search_transcript` and `search_ocr` are valid because those layers already exist. | Current V3/V4.1 policy-answerer accuracy, grounding, and oracle comparisons |
| `simulated_cold_cache` | Cold-cache artifact orchestration. The index is still the read-only fixture, but the policy can only read/search artifacts it has explicitly created in the run cache. | Diagnostic raw-video-style orchestration behavior |

The distinction is critical for efficiency claims. In `prebuilt_index`, a policy that calls `search_transcript` is querying an already-computed ASR layer. In `simulated_cold_cache`, the policy must first create or reuse an ASR artifact before transcript search/read behavior is meaningful.

Example cold-cache smoke run:

```bash
python eval/run_policy_answerer_eval.py \
  --benchmark qa-data/benchmark/v4/val_benchmark.jsonl \
  --index-dir data/video_indexes \
  --output eval/results/v4_cold_cache_smoke.jsonl \
  --policy-adapter training_data/output/qwen35-9b-sft-v4/adapter \
  --tool-execution-mode simulated_cold_cache \
  --clear-artifact-cache \
  --max-questions 5
```

In `simulated_cold_cache`, `data/video_indexes/*.json` remains immutable. The simulator uses it to cheaply materialize artifacts that real CLAMS tools would have created. The disposable registry defaults to `OUTPUT.artifact_registry.json`; clear it between independent experimental conditions.

By default, `--tool-schema-mode auto` exposes the legacy locate-inspect schema in `prebuilt_index` mode and the V5 artifact schema in `simulated_cold_cache` mode. Current caveat: the available SFT/GRPO policies were trained mostly on the legacy schema, so cold-cache mode is still a diagnostic bridge until V5 trajectories teach the policy to create artifacts first (`run_asr`, `run_swt`, `run_ocr`, `run_captioner`) and then read bounded evidence (`read_asr`, artifact search).

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

### Warm-Index Agent Loop

In the current V4.1 warm-index benchmark, the video index already contains ASR, OCR/text-focus, captions, and related layers. A policy can therefore:

1. **Decide what evidence modality is likely needed** ("This is asking about speech, a chyron, or a visible object")
2. **Query an existing layer** (e.g., `search_transcript("unemployment")` or `search_ocr("director")`)
3. **Inspect bounded evidence** (e.g., read ASR around a hit or OCR/caption a relevant timestamp)
4. **Stop when evidence is sufficient**
5. **Pass the gathered evidence to the answerer**

This evaluates evidence retrieval, localization, and provenance over a precomputed archive index. It does not measure whether the model would have known to run ASR/OCR before those layers existed.

### Cold-Cache Agent Loop

When given a question about a new video or an empty cache, the agent must instead:

1. **Decides what information it needs** ("I need to know what's being said")
2. **Selects and runs an artifact-creating tool** (e.g., runs Parakeet ASR on the video, or runs SWT/OCR for title cards and credits)
3. **Processes or registers the artifact** (reads a bounded transcript range, searches OCR, or records embedding/cache metadata)
4. **Decides what to do next** ("I found someone discussing unemployment at 15:43 -- who is this person?")
5. **Runs another tool** (e.g., runs OCR on the frame at 15:47 to read the chyron)
6. **Answers the question** ("The guest represents the National Urban League")

Tool outputs are cached automatically -- if another question arrives about the same video, the agent checks the cache before running tools again. But the cache is an optimization, not the point. The agent's core capability is making good tool selection decisions under resource constraints.

Concrete examples:

- "Who directed this program?" should usually use text-scene/OCR evidence from slates, credits, or title cards; ASR is optional corroboration, not the default first move.
- "What did the guest say about unemployment?" usually justifies ASR, then bounded transcript reads and possibly a chyron/OCR check.
- "Are these two names the same guest or an OCR misspelling?" may require OCR conflict detection, ASR corroboration, edit distance, and eventually face/speaker embeddings.
- "Show frames like the title cards across this collection" requires cached embeddings or an external FAISS-style index, not one-off per-question transcript reads.

### Training Pipeline

1. **Build video indexes** using CLAMS tools on 108 archival videos. This creates a comprehensive knowledge base showing what each tool produces. The index serves as training infrastructure for generating trajectories.

2. **Generate cross-modal QA** from the indexes. Questions require combining information from multiple modalities (speech + visual, OCR + entities) and cannot be answered from any single source.

3. **Reverse-engineer tool execution trajectories** from the QA evidence. Using the provenance chain in the index (which CLAMS app produced each piece of evidence), we construct trajectories showing the sequence of tool execution decisions that would produce the evidence needed to answer each question.

4. **Base-model, SFT, and RL comparisons** on the trajectories. Current V4.1 results show that base `Qwen/Qwen3.5-9B` already performs strong native tool use in the warm-index setting, so SFT is not assumed to be necessary. GRPO/RL should be justified as improving groundedness, localization, efficiency, or preference-conditioned behavior beyond the base policy.

### Preference-Conditioned Tool Selection

The agent can be conditioned on user preferences for speed vs accuracy vs cost:

- **"Process in 10 minutes"**: Agent runs Parakeet (fast ASR) + SWT (fast text detection) + SpaCy NER. Skips expensive visual captioning.
- **"Maximum accuracy"**: Agent runs Whisper (best ASR) + VLM captioner at 1fps + full OCR + VLM-as-judge verification.
- **"Balanced"**: Agent runs Parakeet + VLM captioner at 0.2fps + OCR on text frames only.

This remains a research direction and design goal. Parts of the current GRPO work already model tool/model cost differences, but the full preference-conditioned training story described here is broader than the current headline experimental path.

### Key Distinction from RAG

RAG/context-stuffing baselines assume a prebuilt index and retrieve or pack context directly into the answer prompt. The warm-index agent uses the same kind of existing archive substrate, but the policy must make explicit tool calls and produces an auditable evidence trace before the answerer responds.

The cold-cache agent is a different problem: it must decide which artifacts to create before retrieval is even possible.

| Approach | Assumption | Main Measurement |
|----------|------------|------------------|
| No tools | No video evidence; answer from priors/question text only. | Parametric guessing and MC artifact control. |
| RAG/context stuffing | Prebuilt index already exists. | How much answer accuracy is possible from broad retrieval/context. |
| Warm-index agent | Prebuilt index already exists, but evidence must be gathered through explicit tools. | Tool-selection, localization, grounding, and provenance. |
| Cold-cache agent | Visible cache starts empty. | Artifact creation/reuse decisions, cost, and downstream evidence quality. |

The agent contribution should therefore be reported with both answer accuracy and grounded evidence quality. A correct answer without supporting evidence is not equivalent to a grounded archival answer.

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

### V4 (current: `qa-data/benchmark/v4/`)

- Same questions as V3 -- **only MC distractors were repaired**
- Questions and correct answers are unchanged
- **917 total** (488 MC + 429 FT), same video-level splits
- Canonical split files:
  - **Train:** `qa-data/benchmark/v4/train_benchmark.jsonl` (464 = 242 MC + 222 FT)
  - **Val:** `qa-data/benchmark/v4/val_benchmark.jsonl` (80 = 34 MC + 46 FT)
  - **Test:** `qa-data/benchmark/v4/test_benchmark.jsonl` (373 = 212 MC + 161 FT)
  - **Combined:** `qa-data/benchmark/v4/benchmark_combined.jsonl` (917)

**V4 distractor repair process:**
- Generated by `qa-data/repair_distractors.py` using Gemini 2.5 Flash
- Each question receives a compact **video entity summary** (people, organizations, places, topics from the index) so Gemini can pick video-grounded distractors
- Type consistency enforced: all distractors must match the correct answer type
- 488/488 MC questions repaired, 0 failures
- Original distractors preserved in `mc_options_original` field
- Cost: ~$1-2 total

**Why V4 exists:**
V3 distractors (generated by qwen3:8b) had systematic quality issues:
- Type mismatches (event question with place/time distractors)
- Trivially wrong options
- Non-video-grounded choices
- 93% of V3 MC questions had identifiable distractor quality issues

**Impact on scores:** MC accuracy is expected to decrease on V4 because weak distractors that could be eliminated without evidence are replaced with harder alternatives. V4 scores are more trustworthy.

**Clarification: V4 is distractor repair only, not question regeneration.**
`generate_qa_v4.py` exists as a prototype for dense-evidence question generation but is NOT used for the V4 benchmark. The V4 benchmark uses V3 questions with V3 correct answers -- only the wrong MC options were regenerated. Full V4 question generation (with validation, chapter-aware generation, and dense paraphrasing) is future work.

**Known issues in V4 tooling (not blocking current experiments):**
- `generate_qa_v4.py` lacks validation (no `validate_question()` call) and uses single-pass generation instead of chapter-local -- needs fixing before use
- `eval/run_ablation_answerer.py` still uses Ollama directly instead of the shared `utils/answerer_utils.py` -- needs patching for consistency
- GRPO question lookup uses raw question text as key instead of row ID -- fragile for paraphrased/regenerated datasets
- A small number of inherited V3 questions have grounding text that does not align with the actual indexed ASR at `source_segment_times`. A quick V4 test MC audit found 209 questions with `speech_excerpt`; 11 had <=0.75 token recall against nearby ASR, 9 had <=0.5, and one had 0.0. These are dataset issues, not model errors, because the policy cannot retrieve quoted speech that is absent from the tool-visible index.

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
| 3.0 (canonical) | Qwen3-VL captioner | `data/video_indexes/` | Production indexes, canonical visual_captions and ocr layers |
| 3.5+ (enriched) | Multi-model variants | `data/video_indexes/` (additional layers) | SmolVLM2, Qwen3.5-0.8B, Qwen3.5-9B caption variants + Whisper ASR |

The canonical indexes use the v2 multi-layer schema. The enriched layers (caption variants, Whisper ASR) are added alongside the canonical layers, not replacing them. Each variant layer tracks provenance (model, task_mode, segment_source).

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

The agent follows a **locate-then-inspect** pattern: first discover relevant moments in the video using search or browsing, then inspect specific timestamps with targeted tools. video_id is bound at the system level, not exposed as a parameter.

**Discovery tools** (find where to look):

| Tool | Purpose | Cost |
|------|---------|------|
| `search_transcript(query, top_k)` | Keyword search across all ASR segments. Returns matching segments with timestamps. | 0.0 |
| `search_ocr(query, top_k)` | Keyword search across on-screen text (chyrons, titles). | 0.0 |
| `browse_timeline(start_time, end_time)` | Compact summary of a time range: topics, speakers, text, notable timestamps. | 0.0 |
| `detect_text_scenes()` | Find all frames containing text (chyrons, slates, credits). | 0.1 |

**Inspection tools** (examine specific moments):

| Tool | Models | Task Modes | Cost |
|------|--------|------------|------|
| `run_asr(start_time, end_time, model)` | parakeet, whisper | -- | 0.2-0.5 |
| `extract_text(timestamp, model)` | qwen-small (Qwen3.5-0.8B), qwen-8b (Qwen3.5-9B) | -- | 0.2-0.4 |
| `caption_frame(timestamp, model, task_mode)` | smolvlm (SmolVLM2-2.2B), qwen-small, qwen-8b | general_scene, text_focus | 0.3-0.5 |
| `identify_speakers(start_time, end_time)` | -- | -- | 0.3 |

**Dual-mode ASR:**
- `run_asr()` with no time range: runs full-video ASR (makes transcript searchable)
- `run_asr(start, end)`: returns transcript for a specific range
- `search_transcript(query)` requires transcript to be available (pre-indexed or after `run_asr()`)

**Caption lookup:**
- Returns all captions within +/- 15 seconds of the requested timestamp (up to 3, closest first)
- Pre-computed per (model, task_mode) and stored as separate index layers (e.g., `caption_qwen-small_text_focus`)
- General_scene captions are from TransNet shots, text_focus captions are from SWT text scenes

All tool execution logic lives in `construct_tool_trajectories.py:simulate_tool_output()` as a single source of truth. Both the GRPO environment and eval delegate to it.

## Pipeline Tools

### Canonical layers (all 107 videos)

| Step | Tool | Layer | Notes |
|------|------|-------|-------|
| Scene detection | app-swt-detection | scenes | Bars, Slate, Chyron, Credits, Other-text |
| Shot detection | app-transnet-wrapper | shots | Frame-level visual boundaries |
| ASR (Parakeet) | Parakeet TDT 0.6B | asr | Fast CTC-based, 98/107 videos |
| ASR (Whisper) | Whisper large-v3 | asr_whisper | Higher quality on noisy audio, 107/107 videos |
| Visual captioning (canonical) | Qwen3-VL captioner | visual_captions | One caption per TransNet shot |
| OCR | Qwen3-VL (swt_transcription) | ocr | Text from SWT-detected text scenes |
| Speaker diarization | pyannote + LLM | speakers, chapters | Speaker names + chapter structure |
| Credits extraction | app-credits-ocr (Qwen3-VL) | credits | Structured credit extraction |
| NER | SpaCy en_core_web_sm | entities | Named entities from ASR/OCR |
| Entity resolution | Substring merging | entities (deduped) | Cross-layer entity linking |
| SVO relations | SpaCy dependency parsing | relations | Subject-verb-object triples |

### Multi-model caption variants (in progress)

Generated by `training_data/generate_caption_variants.py`. Each (model, task_mode) combination produces a separate layer:

| Model | Actual Model | general_scene (TransNet shots) | text_focus (SWT scenes) |
|-------|-------------|-------------------------------|------------------------|
| smolvlm | SmolVLM2-2.2B | In progress | In progress |
| qwen-small | Qwen3.5-0.8B | Done (107/107) | Done (107/107) |
| qwen-8b | Qwen3.5-9B | In progress | In progress |
| qwen-30b | Qwen3.5-27B | Deferred | Deferred |

Quality note: Qwen3.5-0.8B produces surprisingly good structured OCR output (JSON with bounding boxes), sometimes outperforming larger models on text_focus tasks. Model quality is not monotonically "bigger = better."

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
| **Grounded MC accuracy** | Correct MC answers with supporting gathered evidence | answer correctness + evidence support / provenance | planned analysis over prediction tool traces |
| **Answer quality** | Model answers (free-text) | correctness, completeness, relevance, cross_modal, specificity | `score_predictions.py --mode llm-judge` |
| **Question quality** | Questions themselves | cognitive_level, naturalness, groundedness, cross_modal_need | `qa-data/score_question_quality.py` |
| **Retrieval quality** | Segment retrieval | hit rate, temporal IoU, precision | `score_retrieval.py` |

Because the V4.1 no-tools control reaches 58.5% MC, raw MC accuracy must be reported alongside grounding/provenance checks. A correct answer without answer-bearing evidence in the tool trace is a parametric or choice-artifact success, not a grounded archival QA success.

### Agent Evaluation Results (V2 Benchmark)

This section is historical. It predates the V4.1 base-model comparison and should not be used to justify SFT as the default current path.

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

**Historical findings:**
- **Prompted tool use underperformed RAG in the older V2 setup** regardless of framework (ReAct or LangGraph). This motivated SFT experiments at the time, but the current V4.1 Qwen3.5 base-policy result shows that SFT is not automatically justified.
- **SFT teaches tool selection**: The fine-tuned model uses 2.2 tools on average with diverse tool choice (ASR, OCR, visual, speakers) vs 1.0-1.5 for prompted models. But v2 training data had hallucinated timestamps, causing poor evidence retrieval.
- **V3 trajectories address this**: Interactive generation ensures tool arguments are grounded in real observations. Structural timestamps encode video position (percentage, chapter context) instead of raw milliseconds.

### Policy + Answerer Architecture (Final Results)

This section records historical V3 results. The current V4.1 baseline is base `Qwen/Qwen3.5-9B` warm-index policy at 71.0% MC, with no-tools control at 58.5% MC and current-tool oracle at 92.0% MC.

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

### V4 Distractor Repair

V3 MC distractors were generated by qwen3:8b and had systematic quality issues: type mismatches (event question with place/time distractors), trivially wrong options, and non-video-grounded choices.

V4 repairs distractors using Gemini 2.5 Flash with:
- **Video entity grounding**: compact entity summary from the video index (people, organizations, places, topics) is provided so Gemini can use real entities from the broadcast as wrong answers
- **Type consistency enforcement**: all distractors must match the correct answer type (all events, all people, etc.)
- **Quality tracking**: each repair logs what issues were fixed and whether distractors came from the video or were generated

Script: `qa-data/repair_distractors.py`

**Expected impact on scores**: MC accuracy should **decrease** on the repaired benchmark because weak distractors that could be eliminated without evidence are replaced with harder, video-grounded alternatives. The resulting scores are more trustworthy. Models that previously benefited from ruling out obviously wrong options will show their true retrieval quality.

**V4 benchmark files:**
- `qa-data/benchmark/v4/benchmark_mc_repaired_grounded.jsonl` -- questions with repaired distractors
- Original distractors preserved in `mc_options_original` field for comparison

### Remaining Work

| Task | Purpose | Status |
|------|---------|--------|
| V4 distractor repair | Harder, video-grounded MC distractors | Running (93% success so far) |
| Re-eval all models on V4 benchmark | Trustworthy comparison | After repair completes |
| Caption variant generation (SmolVLM, Qwen3.5-0.8B, 9B) | Real model-specific outputs for training | Done (107/107 each) |
| GRPO correctness-only | Can RL improve retrieval? | Done (70.3% on V3) |
| GRPO answerer-in-loop | Aligned train/eval reward | Done (64.2% on V3, underperformed) |
| GRPO weak-cost | Duration-aware cost penalty | Not started |
| person_identity + action_event task modes | Broader task mode coverage | Deferred |
| Human-in-the-loop process scoring | Route low-confidence VLM outputs to review | Future work |
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

| Step | Script | Input | Output |
|------|--------|-------|--------|
| QA generation | `qa-data/generate_qa_v3.py` | Full video indexes | 925 questions + answers + grounding (via Gemini 2.5 Flash) |
| Caption variants | `training_data/generate_caption_variants.py` | Videos + indexes | Per-(model, task_mode) caption layers in indexes |
| Whisper ASR | `training_data/run_whisper_asr.py` | Videos | asr_whisper layer in indexes |
| Expert trajectories | `training_data/construct_tool_trajectories.py` | QA grounding + enriched indexes | 465 trajectories with (tool, model, task_mode) selections |
| SFT training | `training_data/run_native_sft.py` | Trajectories (includes TOOL_SCHEMAS) | LoRA adapter for Qwen3.5-9B |
| GRPO training | `training_data/run_grpo_env.py` | SFT adapter + train questions | Refined LoRA adapter |
| Eval | `eval/run_policy_answerer_eval.py` | Adapter + test benchmark | Prediction JSONL |

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
Assistant: <think>Let me search the transcript for discussion about
           unemployment.</think>
           <tool_call>search_transcript(query="unemployment")</tool_call>
Tool:      Found 3 matching segments (showing top 3):
           [15:43 | 73%] We believe that all the evidence points...
           unemployment is going down...
Assistant: <think>Found relevant speech at 15:43. Let me check for a name
           chyron near that timestamp.</think>
           <tool_call>caption_frame(timestamp="15:47", model="qwen-small",
                                    task_mode="text_focus")</tool_call>
Tool:      [15:47] JOHN JACOB / NATIONAL URBAN LEAGUE
Assistant: The chyron identifies the speaker as representing the
           National Urban League.
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
Teaches the model *what works* through its own rollouts against the simulated environment.
- TRL `environment_factory` enables real multi-turn tool execution during RL
- Generate 4 rollouts per question, score with reward function, update policy
- Three reward variants implemented (`--reward` flag):
  - **correctness_only**: 1.0 correct, 0.0 wrong. No cost penalty. (Current default)
  - **weak_cost**: 1.0 - 0.1 * total_cost for correct. Mild efficiency incentive.
  - **full_cost**: Efficiency decay function gated on correctness. (Original, shown to cause tool avoidance.)
- **Key finding from first round**: count-based efficiency reward (full_cost) degraded the policy below SFT (45.8% vs 51.9%). The model learned to skip tools rather than select better ones. This motivates the correctness-first experiment sequence.
- Rollout logs capture full (tool, model, task_mode) per call for behavioral analysis

### Preference-Conditioned Tool Selection

The (tool, model, task_mode) architecture naturally supports preference conditioning. The agent makes cost/quality tradeoffs at three levels:

**Tool selection**: which capability to use (ASR, OCR, captioning, diarization)
**Model selection**: which quality tier (smolvlm/qwen-small/qwen-8b/qwen-30b for VLM, parakeet/whisper for ASR)
**Task mode**: what information to extract (general_scene vs text_focus)

Cost table reflecting actual CLAMS app resource usage:

| Tool + Model | Cost | Quality |
|-------------|------|---------|
| search_transcript | 0.0 | Index lookup |
| search_ocr | 0.0 | Index lookup |
| browse_timeline | 0.0 | Index summary |
| detect_text_scenes (SWT) | 0.1 | Fast CNN |
| run_asr(parakeet) | 0.2 | Fast CTC |
| run_asr(whisper) | 0.5 | Accurate on noisy audio |
| extract_text(qwen-small) | 0.2 | Fast OCR |
| extract_text(qwen-8b) | 0.4 | Good OCR |
| caption_frame(smolvlm) | 0.3 | Brief descriptions |
| caption_frame(qwen-small) | 0.3 | Good structured output |
| caption_frame(qwen-8b) | 0.5 | Detailed descriptions |
| identify_speakers | 0.3 | Pyannote diarization |

The current iteration trains the agent to select models and modes based on question type. Future work: explicit preference vector conditioning where users specify speed/accuracy/cost constraints and the agent adapts its pipeline accordingly.

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
