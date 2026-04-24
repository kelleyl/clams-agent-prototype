# VNext Dataset Plan

## Purpose

This document plans the next benchmark/dataset release after
`v4_1_fixed_context`.

The immediate motivation is that manual review is still surfacing data quality
problems even after evidence verification:

- weak or implausible distractors
- answer-bearing evidence mixed with noisy evidence
- ASR artifacts such as repeated short filler segments like `Thank you.`
- questions that are too easy from parametric/world knowledge alone
- questions whose required modality is overstated or unclear
- rows where the visible tool output is technically present but poorly packaged
  for answering

The next version should improve question quality, answer quality, and evidence
quality separately, rather than treating them as one combined problem.

## Versioning Rule

Use the version number to signal the amount of change.

### Patch release: `v4.2`

Use `v4.2` if the work is primarily:

- removing bad rows
- correcting answers
- replacing or tightening evidence spans
- repairing distractors
- updating metadata/review fields
- keeping the same basic question families

### Major release: `v5`

Use `v5` if the work includes any of:

- generating a substantial number of new questions from scratch
- introducing new task families, especially canonical field/cataloging tasks
- changing the question schema materially
- changing the evidence representation materially
- separating benchmark tracks in a way that changes the benchmark identity

Current expectation:

- `v4.2` = cleanup/review release on the current benchmark family
- `v5` = new-generation release with improved task design

## Current Benchmark Provenance

This should be explicit, because the current benchmark is the result of several
different stages rather than one clean generation pass.

### What `v4_1_fixed_context` actually is

`v4_1_fixed_context` is not a fresh question-generation dataset.

It is a cleaned benchmark freeze built from:

1. V3-generated question stems
2. V4 distractor repair
3. V4.1 evidence verification and span repair
4. a fixed-context packaging pass for the accepted V4.1 rows

So when we discuss the current benchmark, we need to distinguish:

- original question generation
- multiple-choice distractor generation/repair
- deterministic evidence verification and rejection
- later context packaging fixes

### Stage 1: V3 question generation

Source file:
- `qa-data/generate_qa_v3.py`

Documented in:
- `qa-data/V3_DATASET_MANIFEST.md`

Model used:
- `Gemini 2.5 Flash`

Primary input:
- `107` v2 layered video indexes
- full-video formatted index documents
- source indexes included layered ASR, OCR, Qwen 3.5 captions, chapters, and
  related metadata

Generation style:
- full-context generation over the formatted per-video index
- roughly `~33k` tokens per video according to the V3 manifest

Filtering / validation used at this stage:
- answer-in-index check
- cross-modal validity check
- interstitial / low-value filtering
- video-level split with seed `42`

Important note:
- this stage created the underlying question stems for the current benchmark
- it is the main reason the current benchmark should be described as
  V3-derived rather than as a fresh V4/V5 generation set

### Stage 2: V3 multiple-choice distractor generation

Documented in:
- `qa-data/V3_DATASET_MANIFEST.md`

Model used:
- `qwen3:8b` via Ollama

Primary input:
- V3 question stems and answers
- adversarial multiple-choice generation flow

Filtering used at this stage:
- adversarial distractor filter

Important note:
- this is one of the main known weak points in the historical benchmark lineage
- many later benchmark quality concerns come from this answer-choice stage, not
  from the question stems themselves

### Stage 3: V4 distractor repair

Source file:
- `qa-data/repair_distractors.py`

Model used:
- `Gemini 2.5 Flash`

Primary input:
- existing V3 benchmark rows
- question text
- correct answer
- grounding evidence fields (`speech_excerpt`, `visual_reference`,
  `ocr_reference`, reasoning text)
- compact entity summaries extracted from the same video index

Filtering / constraints used at this stage:
- type consistency requirement
- same-video distractors preferred over invented distractors
- plausibility and distinctness constraints

Important note:
- this stage repaired distractors only
- it did not regenerate question stems from scratch

### Stage 4: V4.1 evidence verification and repair

Source file:
- `qa-data/verify_and_repair_evidence_spans.py`

Model used:
- none; this step is deterministic

Primary input:
- repaired benchmark rows
- local tool-visible video indexes

Verification / filtering behavior:
- verify speech grounding against ASR layers
- verify OCR grounding against OCR / text-focus layers
- verify visual grounding against caption / visual layers
- repair mistimed spans when the evidence exists elsewhere in the video
- reject rows when the claimed grounding is not visible in the index
- write accepted rows, rejected rows, and `verification_report.json`

Important note:
- this stage produced the V4.1-style evidence-verified benchmark candidate
- it introduced typed `evidence_spans`
- it is a filtering and repair pass, not an LLM generation step

### Stage 5: `v4_1_fixed_context`

Current benchmark files:
- `qa-data/benchmark/v4_1_fixed_context/*.jsonl`

Model used:
- none for the packaging fix itself

Primary input:
- accepted V4.1 rows
- repaired evidence context packaging

Important note:
- this stage keeps the same benchmark family
- it does not represent a new generation model or a new benchmark philosophy
- it is a better-packaged freeze of the accepted evidence-verified rows

## What Must Be Documented For VNext

For the next release, we should record provenance in the same document that
defines the release, not leave it implicit in code and conversation history.

Minimum required fields:

- source benchmark or source raw dataset
- source index snapshot used for generation
- question-generation script
- question-generation model and backend
- dense-paraphrasing script and model, if used
- distractor-generation or distractor-repair script and model
- deterministic filtering scripts used
- human review tools used
- rejected-row counts by issue type
- accepted-row counts by format and task family
- exact split files produced

## Core Goals

1. Increase grounding quality.
2. Make distractors consistently strong and type-matched.
3. Remove rows whose answer is not well supported by tool-visible evidence.
4. Reduce benchmark rows that can be answered correctly for the wrong reason.
5. Separate question review from answer-choice review.
6. Preserve enough provenance to explain why each row was kept, edited, or
   rejected.

## Non-Goals

The next dataset version should not try to solve all model-training problems at
once.

Non-goals:

- redesigning the entire tool interface
- mixing warm-index and cold-cache evaluation semantics into the dataset itself
- introducing many new experimental task families before the core QA set is
  stable
- preserving backward comparability at the cost of keeping clearly bad rows

## Current Problem Areas

## 1. Question Quality

Observed issues:

- some questions are underspecified or ambiguous
- some cross-modal questions overstate how many modalities are truly needed
- some questions rely more on generic knowledge than on video evidence
- some questions are phrased around review-time descriptions rather than the
  actual answer-bearing evidence

Desired fix:

- each question should have a clear evidence path
- each question should justify why the video is needed
- modality requirements should reflect actual necessity, not aspiration

## 2. Distractor Quality

Observed issues:

- distractors with the wrong answer type
- distractors that are obviously weak
- near-duplicate distractors that make the item noisy rather than challenging
- multiple-choice sets where the correct answer stands out for superficial
  reasons

Desired fix:

- review question text and answer choices separately
- require type consistency for all distractors
- store distractor provenance and repair rationale

## 3. Evidence Quality

Observed issues:

- evidence spans can be technically valid but too broad
- visible tool output sometimes contains the answer plus a large amount of noise
- some ASR contains likely recognition artifacts, including repeated `Thank you.`
  filler segments
- OCR and caption evidence can include the wrong nearby person/frame/text region

Desired fix:

- prefer specific answer-bearing tool outputs over broad windows
- retain multiple evidence spans when needed, but keep each span precise
- record which layer/tool actually supports the answer
- flag noisy ASR/OCR/caption regions explicitly during review

## 4. Benchmark Leakage / Parametric Answerability

Observed issues:

- some rows appear answerable without grounded retrieval
- some rows have very common entities, institutions, or world facts

Desired fix:

- add a per-row assessment of whether the item is genuinely video-dependent
- rerun and preserve the no-tools baseline on the next benchmark freeze
- demote or remove rows that are mostly testing parametric recall

## 5. Dataset Structure

Observed issues:

- the current benchmark mixes repaired legacy QA with emerging new ideas
- canonical metadata/cataloging tasks are important but not yet separated cleanly

Desired fix:

- keep the core QA benchmark coherent
- introduce canonical field questions as a clearly labeled track, not as ad hoc
  additions

## Proposed Release Structure

## Track A: `v4.2` Cleanup Release

This is the conservative path and should likely happen first.

Scope:

- start from `v4_1_fixed_context`
- use review outputs to accept, reject, or correct rows
- repair answer choices where the question is worth keeping
- tighten evidence spans and tool-visible support
- add clearer issue tags and provenance

Deliverables:

- `qa-data/benchmark/v4_2/`
- `qa-data/benchmark/v4_2_fixed_context/` if we keep the naming pattern
- rejected-row files
- review report
- quality summary

Rows to remove or demote in `v4.2`:

- unsupported or weakly supported rows
- rows with noisy ASR as the only support
- rows with unfixable distractors
- rows that are mostly parametric recall

## Track B: `v5` Major Release

This is the new-generation path.

Scope:

- generate new questions from source evidence, not from prior question wording
- add canonical field/cataloging tasks
- preserve stronger provenance for question generation and answer generation
- separate question generation from distractor generation

New question families to consider:

- field existence:
  - does this video contain a director credit?
  - does this video identify a guest?
- field value:
  - who is the director?
  - what organization is named in the chyron?
- evidence-aware reporting:
  - if present, what is the value and what evidence supports it?
  - if conflicting evidence exists, should the answer be uncertain?

## Source Material For New Generation

For `v5`, generation should start from the actual source evidence shown to the
model, not from prior benchmark phrasing.

Preferred inputs:

- verified ASR segments
- verified OCR items
- verified visual captions
- typed evidence spans

Dense paraphrasing should be used to enrich source evidence when helpful, but it
should not become the only source of truth. The benchmark row must still point
back to original evidence spans and source layers.

## Concrete `v5` Generation Spec

The new-generation path should use a window-first candidate generation pipeline.

The key idea is:

- generate candidate questions from bounded windows of evidence
- verify and rate them in a separate pass
- keep only the questions that survive evidence-aware filtering
- store the final row against the precise supporting evidence, not against the
  broad generation window

### Why window-first generation is preferred

Compared with full-video generation, window-first generation should improve:

- distribution across the video
- question specificity
- evidence locality
- coverage of field reports and other non-studio segments in long news programs

The window is a candidate source, not the final evidence unit. The kept row
should still point to the actual answer-bearing ASR/OCR/caption items.

### Multi-scale candidate windows

Do not rely on only one window size.

Use several candidate sources per video:

- local windows:
  - `30-90s`
  - used for specific ASR/OCR/visual evidence and tightly grounded questions
- medium windows:
  - `3-5 min`
  - used for short packages, interviews, and topic-contained segments
- large windows:
  - `8-10 min`
  - used when the answer depends on a broader discussion arc
- chapter or package windows:
  - preferred when real chapter/topic boundaries exist
- whole-video windows:
  - reserved for canonical metadata or whole-program questions

Generation should oversample from these windows and then filter aggressively.

### Window sampling policy

Use stratified sampling rather than a literal fixed rule like one question per
minute.

Recommended policy:

- sample candidate windows at multiple scales
- ensure coverage across the video timeline
- ensure coverage across segment roles
- cap oversampling from obvious low-value regions such as:
  - opening title sequences
  - generic studio tosses with no substantive content
  - credits and purchase/transcript bumpers
  - repetitive promos or sponsor/funding cards

For videos without reliable chapters, derive pseudo-segments from windows and
role labels instead of collapsing the entire program to `full_video`.

### Segment-role labels

Every candidate window should receive a coarse segment-role label before or
during generation.

Recommended labels:

- `studio_anchor`
- `field_report`
- `interview`
- `panel_or_analysis`
- `broll_or_montage`
- `credits_or_promo`
- `other`

These labels can come from:

- existing chapters when they are useful
- heuristics over ASR/OCR/caption evidence
- an LLM classification pass over the candidate window context

### NewsHour-specific coverage rule

This is especially important for PBS NewsHour / MacNeil-Lehrer style videos,
where the archival value is often in the field packages rather than in the
anchor intro.

Because many NewsHour videos currently lack usable local chapter annotations,
the generation pipeline should impose role-aware coverage explicitly.

Minimum goals for long-form NewsHour-style programs:

- at least `1` accepted question from each substantial `field_report` window
- at least `1` accepted question from each substantial `interview` window
- at least `2-3` accepted questions from non-studio segments in a typical
  60-minute program
- do not let studio-anchor or credits questions dominate the accepted set

This should be treated as a distribution target, not a hard guarantee if the
evidence quality is poor.

### Generation stages

The pipeline should separate question creation, verification, and distractor
creation.

### Thinking-mode comparison

Thinking mode should be treated as an explicit generation parameter rather than
as an always-on default.

Recommended parameters:

- `generation_thinking = on | off`
- `verification_thinking = on | off`

Default assumption for the main pipeline:

- generation: `off`
- verification: `off`

Use thinking mode only if a small calibration study shows a meaningful quality
gain that justifies the additional latency and token cost.

#### Small comparison study

Before large-scale `v5` generation, run a focused comparison on `3` deliberately
different videos:

- one long-form NewsHour-style video with field segments
- one interview or talk-heavy video
- one OCR / credits / metadata-heavy or otherwise structurally different video

Compare at least:

- `generation_thinking = off`
- `generation_thinking = on`

with all other settings held fixed:

- same model
- same windows
- same prompts
- same verifier
- same acceptance thresholds

#### Repeat each setting twice

Run each setting at least `2` times.

Reason:

- even if acceptance rates are similar, different settings may produce
  different "flavors" of questions
- repeated runs help measure whether a setting yields:
  - more varied question families
  - more field-oriented questions
  - more canonical-field questions
  - more duplicates or near-duplicates

This comparison should therefore capture both:

- quality differences
- distribution / flavor differences

#### What to measure

For each condition, record:

- accepted questions per video
- duplicate rate
- usefulness
- ambiguity
- video dependence
- evidence precision
- question-family distribution
- segment-role distribution
- latency
- token usage

If possible, also do a small human comparison on the accepted questions to see
whether one setting produces noticeably more useful or better distributed
questions even when automatic scores are similar.

#### Stage 1: Generate candidate questions

Input:

- bounded window context
- nearby ASR/OCR/caption evidence
- optional neighboring windows for context
- video metadata
- segment-role label

Output per candidate:

- question text
- provisional answer
- suggested task family
- suggested modality requirement
- suggested supporting evidence items
- short rationale

At this stage the system should generate more candidates than will be kept.

#### Stage 2: Verify, answer, and rate

Use a separate model call from the generator.

Input:

- candidate question
- candidate answer
- bounded window evidence only

Output:

- verified answer
- answerability judgment
- cited support span or spans
- structured quality scores
- pass or fail decision

This stage can combine answering and rating in one structured call, but it
should remain separate from the initial generator so that the generator is not
grading its own work.

#### Stage 2 model policy

The verifier / rater does not need to be the same size as the generator.

Preferred setup:

- use a stronger model for candidate generation
- use a smaller, cheaper model for verification and rating
- evaluate one candidate at a time rather than scoring a large bundle in one
  prompt

Rationale:

- one-at-a-time verification reduces order effects
- one-at-a-time verification is easier to audit and debug
- this stage is primarily about precision and consistency, not creativity

Practical target:

- a mid-size judge model is acceptable here, for example something in the
  `12B-27B` range if it is sufficiently reliable on the calibration set
- if Qwen-family consistency is important, a Qwen 3.5-family judge is a
  reasonable default

The verifier should return structured outputs only, such as:

- `answerable_from_evidence`
- `best_answer`
- `support_span`
- `usefulness`
- `ambiguity`
- `video_dependence`
- `pass_fail`
- short rationale

#### Stage 3: Multiple-choice distractor generation

Only run this stage for rows that already passed Stage 2.

Input:

- verified question
- verified answer
- same-video entity and evidence context
- answer type

Output:

- distractor set
- distractor type labels
- distractor rationale

Distractor generation should remain separate from question generation.

### Candidate scoring fields

Each candidate should be scored on explicit dimensions before acceptance.

Recommended fields:

- `answer_recoverable_from_evidence`
  - boolean
- `usefulness`
  - `1-5`
- `ambiguity`
  - `1-5`
- `video_dependence`
  - `1-5`
- `evidence_precision`
  - `1-5`
- `parametric_answerability_risk`
  - `1-5`
- `novelty_within_video`
  - `1-5`
- `modality_fit`
  - `1-5`
- `segment_role_fit`
  - `1-5`

Suggested interpretation:

- usefulness:
  - is this worth asking in an archival or retrieval setting?
- ambiguity:
  - could a reasonable reviewer read the evidence and still disagree?
- video dependence:
  - does the video actually need to be consulted?
- evidence precision:
  - is the answer supported by specific evidence rather than a noisy dump?
- parametric risk:
  - is the question too answerable from world knowledge alone?
- novelty within video:
  - does it add something different from existing accepted questions?

### Acceptance rules

Default acceptance should require all of:

- `answer_recoverable_from_evidence = true`
- clear cited support span or spans
- usefulness above a minimum threshold
- ambiguity below a maximum threshold
- acceptable parametric-answerability risk
- no near-duplicate accepted question in the same video

Rows should be rejected or demoted if they:

- rely on noisy ASR artifacts
- are mostly answerable from generic world knowledge
- duplicate an already accepted question from a nearby window
- use a broad generation window when the final answer-bearing evidence cannot be
  localized

### Final evidence packaging rule

The accepted row should not keep the broad generation window as its only
support.

Instead, store:

- the generation window metadata
- the final supporting evidence item or items
- the final answer-bearing span or spans

This preserves the benefit of broad exploration while keeping the benchmark
itself tightly grounded.

## Required Schema Additions

The next version should store more review structure per row.

Recommended fields:

- `question_review`
  - decision
  - issue_tags
  - reviewer_notes
- `answer_review`
  - decision
  - distractor_issue_tags
  - repair_rationale
- `evidence_review`
  - decision
  - supporting_items
  - noisy_items
  - preferred_span
- `video_dependence`
  - `high`, `medium`, or `low`
- `task_family`
  - `core_qa`, `canonical_field`, `cataloging`, etc.
- `question_source`
  - legacy carryover, regenerated, human-authored, repaired

## Review Workflow

The review process should be split into explicit stages.

### Stage 1: Automated triage

- evidence verification
- answer-type checking for distractors
- duplicate/near-duplicate distractor detection
- ASR noise heuristics
- no-tools risk heuristics

### Stage 2: Human question review

Use the annotation reviewer for:

- correct answer confirmation
- bad question marking
- flagging rows for deeper review

### Stage 2A: Human calibration of the verifier

Before relying heavily on the smaller verifier / rater, compare it against a
human-annotated calibration set.

The calibration sample should be stratified rather than purely random.

Include:

- high-confidence passes
- borderline cases
- obvious fails
- different segment roles such as:
  - `field_report`
  - `studio_anchor`
  - `interview`
  - `panel_or_analysis`

Measure:

- pass/fail agreement
- agreement on answerability from evidence
- agreement on ambiguity band
- agreement on usefulness band
- false-positive rate on accepted questions

The main requirement is not perfect agreement.

The key requirement is that the verifier has high precision on accepted rows
and that its failure modes are documented.

### Stage 3: Human evidence review

Use the evidence browser for:

- replacing noisy support with better support
- selecting more precise tool-visible items
- identifying ASR/OCR/caption artifacts

### Stage 4: Distractor review

Distractors should be reviewed separately from the question stem.

This can use:

- LLM-judge majority vote for type consistency and plausibility
- human spot review for high-risk categories
- explicit rejection of near-duplicate distractor sets

### Stage 5: Freeze and validate

Before release:

- run matched-text oracle
- run current-tool oracle
- run no-tools baseline
- inspect failures by modality and question family

## Gating Criteria

The next version should not be declared current until:

1. Oracle results and artifacts are preserved.
2. Rejected rows are preserved and categorized.
3. Review outputs are versioned and reproducible.
4. The no-tools control is rerun and documented.
5. Each kept row has a clear answer-bearing evidence path.

## Acceptance Targets

These are directional targets, not hard guarantees.

- fewer rows with weak distractors
- fewer rows with noisy evidence packaging
- materially cleaner answer-bearing evidence than `v4_1_fixed_context`
- lower ungrounded success under no-tools control
- clearer separation between core QA and canonical field tasks

## Immediate Next Steps

1. Continue review in the fixed-context annotation app and evidence browser.
2. Add explicit issue tags for:
   - weak distractor
   - parametric-only risk
   - noisy ASR
   - noisy OCR
   - ambiguous visual reference
   - bad evidence packaging
3. Export a review summary grouped by issue type and by video.
4. Decide whether the next release is a `v4.2` cleanup or a `v5` regeneration.
5. If `v4.2`:
   - apply review corrections to the current rows
   - regenerate rejected/accepted manifests
6. If `v5`:
   - build a source-evidence-first generation pipeline
   - keep canonical field tasks as a separate labeled family

## Recommendation

The best near-term path is:

- do `v4.2` first as a cleanup release
- use that to stabilize review tooling, evidence packaging, and distractor
  quality
- then do `v5` as the genuinely new benchmark generation pass

That keeps the next release tractable while still leaving room for a more
ambitious benchmark redesign.
