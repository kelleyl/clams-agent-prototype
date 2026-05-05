# CLAMS-Agent: Agent-Orchestrated Video Indexing and QA

CLAMS-Agent is a research codebase for training and evaluating LLM-based agents that answer questions about long-form archival broadcast video by selecting evidence tools over structured video indexes and, in cold-cache experiments, deciding which derived artifacts to create.

The current pipeline is built around:
- layered video indexes in `data/video_indexes/`
- V3 benchmark generation and curation in `qa-data/`
- trajectory construction and policy training in `training_data/`
- policy + answerer evaluation in `eval/`

## Experimental Regimes

The project now separates two related but different tasks:

| Regime | Initial State | What Is Measured | Current Role |
|--------|---------------|------------------|--------------|
| Warm-index archive QA | ASR, OCR, captions, shots, and other layers already exist in `data/video_indexes/`. | Can the policy query the right existing layers, gather answer-supporting evidence, and give the answerer provenance? | Main V4.1 accuracy/grounding benchmark. |
| Cold-cache artifact orchestration | The visible run cache starts empty; the source index is only an immutable simulator fixture. | Can the policy decide which CLAMS-style artifacts to create, reuse, read, and search under cost constraints? | Diagnostic bridge toward raw-video deployment. |

Warm-index results should not be described as raw-video orchestration. For example, `search_transcript` is a valid warm-index lookup only because ASR already exists. In a cold-cache setting, the agent must first create or reuse an ASR artifact before transcript search/read operations are meaningful.

Example tool-planning distinctions:

- A speech-content question should usually use transcript evidence: locate relevant terms, read a bounded ASR window, then optionally inspect visual evidence.
- A credits/slate question such as "Who directed this program?" should usually prefer text-scene/OCR evidence and should not run ASR unless speech corroboration is useful.
- A guest-identity question may require OCR/chyrons, ASR, or both depending on whether the video names guests visually, verbally, or inconsistently.
- A future collection-search question such as "Show frames like these title cards across the collection" requires cached visual/text embeddings or a FAISS-style external index, not just per-question transcript search.

## Current Experimental Flow

The current workflow is:

1. **Build layered video indexes** from AAPB videos.
   Indexes include ASR, OCR, visual captions, speakers, chapters, entities, and increasingly model-specific variant layers such as Whisper ASR or caption variants by model/task mode.

2. **Generate and curate V3 QA** from the indexes.
   The current benchmark generation flow is full-context QA generation from the annotated broadcast plus post-generation filtering and verification. Canonical split files live under `qa-data/raw/` and `qa-data/benchmark/v3/`.

   For current V4 benchmark work, the direction is to insert a **dense paraphrasing / semantic enrichment** step before final question and MC construction. Rather than only paraphrasing the generated question surface form, the pipeline rewrites the source evidence presentation into a denser textual form that makes implicit semantics more explicit:
   - reference resolution
   - speaker / participant roles
   - event structure
   - temporal relations
   - cross-modal alignment between speech, OCR, and visual evidence

   The goal is to improve question generation, grounding, and distractor quality by generating from a richer semantic substrate rather than trying to repair underspecified questions after the fact.

   A separate V4 subtrack now targets **canonical-field QA** for archival metadata and participant fields. This is not general question generation. It asks paired tasks for fields such as `director`, `air_date`, `program_title`, `host`, `producer`, and `guests`:
   - existence: "Does this video identify a director?"
   - value extraction: "Who is credited or identified as a director in this video?"

   The generator is evidence-first and index-backed: it extracts candidate field values from OCR, slate/credit text, chyrons, title cards, and speaker labels, then emits QA rows with exact evidence provenance. These rows are labeled `candidate` by default because OCR and caption layers can be noisy, and negative existence answers are index-relative until reviewed.

3. **Construct tool-use trajectories** against real indexes.
   Trajectories are derived from the evidence chain for benchmark questions and then converted into SFT / policy-training data.

4. **Train or evaluate tool-selection policies** with base-model prompting, SFT, and GRPO.
   The current warm-index setup uses index-backed simulation rather than live CLAMS execution at training time. The policy learns to choose tools, models, and task modes over existing index layers. Cold-cache experiments wrap the same read-only source indexes with an explicit disposable artifact registry.

5. **Evaluate with a policy + answerer architecture**.
   The policy gathers evidence through tool calls; a separate answerer model answers from the gathered evidence. Final metrics are computed post-hoc from saved prediction artifacts.

## Current Warm-Index Baselines and Target

The current accuracy-focused path is **warm-index archive QA**: the source video indexes are treated as existing archive infrastructure, and the policy is evaluated on whether it can retrieve answer-relevant evidence from those layers. This run intentionally does **not** claim raw-video cold-start orchestration.

Recent V4.1 results show that the base `Qwen/Qwen3.5-9B` policy is a strong warm-index baseline without SFT. SFT is therefore no longer assumed to be necessary; any future training run should be justified against the base-model and no-tools controls.

Current reportable warm-index references:

| Condition | Status | MC Accuracy | Avg Tools | Interpretation |
|-----------|--------|-------------|-----------|----------------|
| Unified current-tool oracle | current oracle | 184/200 = 92.0% | 2.1 | Tool-visible ceiling when the verified evidence layer/window is known. |
| Base `Qwen/Qwen3.5-9B` + tools | current warm-index policy baseline | 142/200 = 71.0% | 2.75 | Zero-shot policy over existing index layers; Aristotle shard artifacts pending local sync. |
| No-tools parametric control | current control | 117/200 = 58.5% | 0 | Shows how much MC can be answered from priors/choice artifacts without grounding. |
| Unified V4.1 SFT v2 + tools | negative SFT result | 124/200 = 62.0% | 3.15 | Corrected-format SFT underperformed the base policy. |
| Old V4 recovery SFT + tools | historical adapter baseline | 133/200 = 66.5% | 2.43 | Valid historical baseline, trained before fixed-context/unified-layer trajectories. |

The desired warm-index behavior is targeted evidence acquisition: use transcript search when speech is relevant, OCR/text-scene tools for slates/chyrons/credits, visual captioning only when visual evidence is needed, and timeline browsing only when coarse localization is useful. The desired cold-cache behavior is different: create or reuse only the artifacts needed for the question, then read/search those artifacts with provenance and cost accounting.

Important SFT lineage:

- Old V4 recovery adapter: `training_data/output/qwen35-9b-sft-v4-oracle-locate-targeted-recovery-x3-r1-i1/adapter`
- Old V4 recovery training data: `training_data/output/trajectories_v4_oracle_locate_nobrowse_x3_retry_x1_interleaved_x1.jsonl`
- Note: `nobrowse` is a historical filename token. This adapter was trained on the older V4 recovery trajectories, not the new V4.1 evidence-span trajectories.
- Old trajectory mix: `2,109` total = `1,395` standard, `386` consecutive retry-recovery, `328` interleaved recovery.

The SFT results should be treated as negative/diagnostic until they beat the base warm-index policy. The first SFT result showed a trajectory-design problem rather than a tool-format problem: the policy called tools, but often inspected the wrong time range or frame and then abstained or answered from weak evidence. The next trajectory iteration added explicit recovery traces:

- consecutive same-tool retry, such as `run_asr -> run_asr` or `caption_frame -> caption_frame`
- interleaved recovery, such as `run_asr(wrong segment) -> caption_frame/search_ocr/extract_text -> run_asr(new segment)`

The first V4.1 evidence-span trajectory file is now superseded for future definitive training:

- Superseded file: `training_data/output/trajectories_v4_1_evidence_spans_x1_retry_x1_interleaved_x1.jsonl`
- Reason: it predates the latest fixed-context / unified-layer oracle fixes
- Provenance: `1,279` total = `439` standard, `439` consecutive retry-recovery, `401` interleaved recovery

Any next SFT run should first explain why training is expected to improve on the base `Qwen/Qwen3.5-9B` warm-index baseline. If using `training_data/output/trajectories_v4_1_unified_x1_r1_i1.jsonl`, sync and verify that artifact first because it is not present locally in this checkout.

## Current V4.1 Results

Use [docs/RUN_LEDGER.md](/Users/kelleylynch/clams/clams-agent-prototype/docs/RUN_LEDGER.md) as the source of truth for run status, scores, supersession, and representative errors. README files should only summarize the current state.

Current reportable references:

| Reference | Status | File | MC Accuracy | Notes |
|-----------|--------|------|-------------|-------|
| Matched-text oracle | current | `eval/results/v4_1_fixed_oracle_matched_text_test.jsonl` | 194/200 = 97.0% | Upper-bound evidence answerability; fixed evidence-context split |
| Unified current-tool oracle | current | `eval/results/v4_1_unified_oracle_current_tool_test_v2.jsonl` | 184/200 = 92.0% | Direct verified-layer/window oracle after layer unification and centered truncation |
| Base `Qwen/Qwen3.5-9B` warm-index policy | current baseline | Aristotle shards `v4_1_base_model_shard*.jsonl` | 142/200 = 71.0% | No LoRA adapter; queries the prebuilt index |
| No-tools parametric control | current control | Aristotle `v4_1_no_tools_baseline.jsonl` | 117/200 = 58.5% | Correct answers without grounding; benchmark-quality control |
| Base `Qwen/Qwen3.5-9B` simulated cold-cache diagnostic | current diagnostic | Aristotle `v4_1_base_cold_cache_isolated_test.jsonl` | 135/200 = 67.5% | Explicit artifact/cache schema; source index remains a simulator fixture |
| Unified V4.1 SFT v2 policy | negative SFT result | Aristotle shards `v4_1_unified_v2_policy_shard*.jsonl` | 124/200 = 62.0% | Corrected native-tool SFT underperformed base |
| SFT policy, old V4 recovery adapter, max turns 3 | historical baseline | `eval/results/v4_1_sft_recovery_alltools_localbase_test_maxturns3.jsonl` | 133/200 = 66.5% | Valid historical baseline, but worse than base and trained before fixed-context/unified-layer trajectories |

Superseded results are preserved in the run ledger rather than deleted. In particular, the earlier `85.0%` fixed current-tool oracle is no longer the current tool-visible ceiling because the layer naming and oracle evidence-window behavior changed.

Current interpretation:

- The fixed evidence-context split shows the reviewed V4.1 questions are mostly answerable when the verified evidence text is visible.
- The unified current-tool oracle raised the tool-visible ceiling from 85.0% to 92.0%.
- The base model already performs strong zero-shot warm-index tool use at 71.0%, while no-tools performance is 58.5%; future claims should distinguish raw answer accuracy from grounded accuracy.
- The remaining warm-index policy gap is retrieval/localization and evidence quality: base policy is 21 points below the current-tool oracle.
- The current SFT adapters are negative results relative to the base policy and should not be used as the default training story without a targeted failure analysis.

## Canonical Paths

### Benchmark and QA
- Canonical V3 split files: `qa-data/benchmark/v3/`
- Gold / verified subsets for stricter evaluation: `qa-data/benchmark/v3/*_gold.jsonl`, `*_verified.jsonl`
- Raw split artifacts: `qa-data/raw/qa_v3_*.jsonl`
- Canonical-field candidate QA generator: `qa-data/generate_canonical_field_qa.py`
- Current canonical-field candidate artifact: `qa-data/raw/canonical_field_qa.jsonl`

### Training
- Current training code: `training_data/`
- Current environment-based GRPO path: `training_data/run_grpo_env.py`
- Current trajectory construction: `training_data/construct_tool_trajectories.py`

### Evaluation
- Canonical prediction-generation step: `eval/run_policy_answerer_eval.py`
- Canonical scoring step: `eval/score_predictions.py`
- Evaluation docs and benchmark history: `eval/README.md`

## Repo Structure

| Path | Purpose |
|------|---------|
| `data/video_indexes/` | Layered video indexes used for QA and tool simulation |
| `qa-data/` | Benchmark generation, filtering, verification, and split artifacts |
| `training_data/` | Trajectory generation, SFT prep, GRPO environment training |
| `eval/` | Prediction generation, ablations, scoring, and eval documentation |
| `utils/` | CLAMS execution, MMIF/index handling, supporting utilities |
| `clams_haystack/` | Chainlit/Haystack cataloging app and CLAMS service endpoint integration |

## Current Evaluation Pattern

For the current V4.1 experiments, evaluation should be treated as a two-step process:

1. Generate predictions:

```bash
python eval/run_policy_answerer_eval.py \
  --benchmark qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl \
  --index-dir data/video_indexes \
  --output eval/results/my_policy_test.jsonl \
  --policy-adapter training_data/output/my_adapter \
  --policy-base Qwen/Qwen3.5-9B \
  --answerer-backend local-base
```

2. Score predictions:

```bash
python eval/score_predictions.py \
  --predictions eval/results/my_policy_test.jsonl \
  --benchmark qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl
```

`run_policy_answerer_eval.py` is the prediction-generation step. Final MC / free-text metrics should come from `score_predictions.py`.
Use `--answerer-backend local-base` when GPU pressure matters: it reuses the loaded Qwen3.5 policy base model with the LoRA policy adapter disabled for answer synthesis, avoiding a second Qwen3.5 HTTP server process.

Omit `--policy-adapter` to run the base-model warm-index baseline. This should be run before interpreting any SFT/GRPO adapter result.

## Current Training Pattern

The active policy training path is warm-index simulation rather than live CLAMS execution:

- the agent calls tools in the GRPO environment
- tool outputs are simulated from real index layers
- model and task-mode differences are represented through indexed layer variants and simulation logic

This lets the policy learn tool selection without paying live inference cost during training.

For the current V4.1 warm-index runs, the index should be treated as the ground-truth evidence substrate for training/evaluation, not as a mutable cache experiment. Cache state, artifact storage, FAISS/vector-index management, and live CLAMS execution are separate cold-cache / collection-indexing experiments and should be reported separately.

## Earlier / Legacy Components

This repo still contains earlier systems and experiments. They are retained for provenance and comparison.

### Earlier LangGraph-Orchestrator Framing

The project originally emphasized a LangGraph-based CLAMS pipeline orchestrator for dynamic workflow construction. That code still exists in modules like:
- `agent.py`
- `utils/clams_tools.py`
- `utils/clams_executor.py`
- `utils/evaluation_rag.py`

That system remains useful context and infrastructure, but it is no longer the best summary of the current benchmark/training/eval workflow.

### Earlier Synthetic Training Pipeline

The repo also contains an older synthetic ToolOrchestra-style training pipeline under `training_data/` built around:
- synthetic tasks
- synthetic tool trajectories
- generic CLAMS pipeline orchestration examples

That pipeline is still documented and preserved, but the current research path relies much more heavily on:
- real video indexes
- benchmark-derived trajectories
- V3 benchmark curation
- GRPO over index-backed tool simulation

### Earlier Eval Paths

Several evaluation scripts remain for older experiments or special-purpose baselines, including:
- `eval/run_sft_eval.py`
- `eval/run_ablation_answerer.py`
- `eval/run_native_tool_eval.py`
- `eval/run_react_eval.py`
- `eval/run_langgraph_eval.py`

These are not all equivalent to the current canonical V3 policy evaluation path. See `eval/README.md` for the current interpretation.

## Dataset Positioning

This benchmark overlaps with prior VideoQA work, but it is not targeting the same underlying problem.

- **DramaQA** is the closest conceptual influence for the cognitive hierarchy. It is built for character-centered story understanding in a fictional drama world, with recurring characters, emotions, intentions, and causal narrative structure across shots and scenes.
- **TVQA** is relevant as a subtitle-heavy TV benchmark, but it is still centered on short entertainment clips rather than long-form archival retrieval.
- **How2QA** focuses on instructional and procedural reasoning over short clips, which is only a partial match for archival broadcast material.
- **STAR** emphasizes short situated real-world reasoning with scene-graph structure, not long-form evidence discovery across a full program.

The CLAMS-Agent benchmark differs in both **content ontology** and **evaluation target**:

- The videos are primarily **news reports, interviews, performances, documentaries, and other archival broadcast segments**, not fictional narratives with stable character arcs.
- The central unit of understanding is usually **segment-centered informational understanding** rather than character-centered story understanding.
- Questions are often about **speakers, topics, events, on-screen text, identities, chronology, and cross-modal evidence alignment**.
- The benchmark is designed for **tool-using agents over structured indexes**, not only passive end-to-end VideoQA models.

This makes DramaQA most useful as a model of **cognitive difficulty structuring**, while the archival benchmark contributes a different axis centered on **archival information needs** and **long-video multimodal retrieval**.

## Documentation

- [eval/README.md](/Users/kelleylynch/clams/clams-agent-prototype/eval/README.md): benchmark history, split artifacts, canonical eval flow, results
- [training_data/README.md](/Users/kelleylynch/clams/clams-agent-prototype/training_data/README.md): current training pipeline and legacy synthetic pipeline
- [docs/RUN_LEDGER.md](/Users/kelleylynch/clams/clams-agent-prototype/docs/RUN_LEDGER.md): run records, scores, supersession status, known issues, and error examples
- [qa-data/V3_DATASET_MANIFEST.md](/Users/kelleylynch/clams/clams-agent-prototype/qa-data/V3_DATASET_MANIFEST.md): V3 artifacts, known benchmark issues, and current V4 dense-paraphrasing / MC-repair direction
- [CLAMS_AGENT_SPEC.md](/Users/kelleylynch/clams/clams-agent-prototype/CLAMS_AGENT_SPEC.md): broader system vision

## Related

- [CLAMS Project](https://clams.ai/)
- [CLAMS Apps Directory](https://apps.clams.ai/)
- [MMIF Specification](https://mmif.clams.ai/)
- [American Archive of Public Broadcasting](https://americanarchive.org/)
