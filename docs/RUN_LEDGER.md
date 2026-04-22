# Run Ledger

This file is the source of truth for reported training and evaluation runs. The
README files should summarize the current state and link here instead of
duplicating full result histories.

## Status Labels

| Status | Meaning |
|--------|---------|
| `current` | Best current result for this experiment type. Use in active reporting. |
| `active-baseline` | Valid baseline, but not the current best result. |
| `superseded` | Valid historical artifact, but replaced by a later run after a bug fix, data fix, or architecture change. Do not report as current. |
| `invalidated` | Known bug makes the result misleading. Keep only for provenance. |
| `pending` | Run or artifact is reported/planned but not yet complete or not synced locally. |

## Run Record Template

Each new run should record:

| Field | What to Record |
|-------|----------------|
| Run ID | Stable short name, usually matching the output filename. |
| Status | One of the labels above. |
| Date | Date run completed or was last updated. |
| Goal | What question this run answers. |
| Input data | Benchmark split, trajectory file, and index version. |
| Model stack | Policy base, adapter, answerer, and backend. |
| Tool mode | Tool schema and execution mode. |
| Output artifact | JSONL prediction file, adapter directory, logs. |
| Score | MC accuracy, rows completed, free-text status, average tool count when relevant. |
| Supersedes / superseded by | Explicit lineage. |
| Known issues | Bugs, architecture caveats, or data caveats. |
| Error examples | A few representative failures with question id, expected answer, predicted answer, and likely failure mode. |

## Experiment Regimes

Runs should declare which regime they evaluate:

| Regime | Initial State | Report As |
|--------|---------------|-----------|
| `no_tools` | No video evidence is shown. | Parametric guessing / MC artifact control. |
| `warm_index` | ASR, OCR, captions, and other layers already exist in the read-only video index. | Archive-index evidence retrieval and grounding. |
| `current_tool_oracle` | The verified evidence layer/window is known and read directly. | Tool-visible ceiling, not normal policy execution. |
| `simulated_cold_cache` | The visible artifact cache starts empty; source indexes are immutable simulator fixtures. | Diagnostic artifact orchestration / raw-video-style planning. |

Warm-index runs can use `search_transcript` because ASR already exists. Cold-cache runs must create or reuse an ASR artifact before transcript search/read operations are meaningful.

## Current Reportable Runs

### RUN-V4.1-MATCHED-TEXT-ORACLE-FIXED-CONTEXT

| Field | Value |
|-------|-------|
| Status | `current` oracle upper-bound reference |
| Date | 2026-04-20 |
| Goal | Measure whether the reviewed V4.1 questions are answerable when the verified evidence text is shown directly. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Index version | Fixed evidence-context split generated from unified local/Aristotle indexes. |
| Model stack | `Qwen/Qwen3.5-9B` answerer |
| Tool mode | `matched-text` evidence-span oracle; no policy retrieval. |
| Output artifact | `eval/results/v4_1_fixed_oracle_matched_text_test.jsonl` |
| Score | 350 rows, 200 MC, 194/200 = 97.0% MC, 150 free-text rows unscored, avg evidence items 2.07 |
| Supersedes | `RUN-V4.1-MATCHED-TEXT-ORACLE-OLD-CONTEXT` |
| Known issues | Free-text rows still need post-hoc scoring. The last 3% MC gap may include answerer error, ambiguous questions, or residual evidence noise. |

### RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V2

| Field | Value |
|-------|-------|
| Status | `current` tool-visible oracle reference |
| Date | 2026-04-20 |
| Goal | Measure the oracle ceiling when evidence is read from the verified tool-visible layer/window after layer-name unification and centered truncation. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Index version | Unified layer naming: `visual_captions -> caption_qwen3vl-8b_general_scene`, `ocr -> caption_qwen3vl-8b_text_focus` |
| Model stack | `Qwen/Qwen3.5-9B` answerer |
| Tool mode | Evidence-span `current-tool` oracle with direct verified-layer reads, full evidence window bounds, and answer-term-centered truncation. |
| Output artifact | `eval/results/v4_1_unified_oracle_current_tool_test_v2.jsonl` |
| Score | 350 rows, 200 MC, 184/200 = 92.0% MC, 150 free-text rows unscored, avg evidence items 2.07 |
| Supersedes | `RUN-V4.1-FIXED-CURRENT-TOOL-ORACLE`, `RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V1` |
| Known issues | This is an oracle diagnostic, not normal policy execution. It can read the verified layer/window directly. Remaining gap to matched-text oracle is mostly multi-item visual/OCR window fidelity and truncation/anchoring. |

Representative errors:

| Question ID | Expected | Predicted | Likely Failure Mode |
|-------------|----------|-----------|---------------------|
| `v3-1279b012-000` | `B`: Large, gold-rimmed glasses | `C`: Large, black-rimmed glasses | Visual evidence window still includes the wrong or ambiguous nearby person/caption item. |
| `v3-1279b012-001` | `B`: A full, grey/white beard | `D`: A clean-shaven face | Cross-modal person localization failure inside a long segment. |
| `v3-0aecf2ea-004` | `D`: Oysters and Crabs | `A`: Blue Crabs and American Oysters | Near-duplicate distractor / OCR wording ambiguity. Flag for QA review. |

### RUN-V4.1-BASE-QWEN35-WARM-INDEX

| Field | Value |
|-------|-------|
| Status | `current` warm-index policy baseline |
| Date | 2026-04-21 |
| Goal | Measure zero-shot Qwen3.5 policy performance over the prebuilt V4.1 video indexes with the same tool schemas and answerer as adapter runs. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Model stack | Policy `Qwen/Qwen3.5-9B` without LoRA adapter; answerer `Qwen/Qwen3.5-9B` via `--answerer-backend local-base` |
| Tool mode | `warm_index` / `prebuilt_index`, all legacy evidence tools exposed, max turns 5 |
| Output artifact | Aristotle `eval/results/v4_1_base_model_shard0.jsonl` through `v4_1_base_model_shard3.jsonl` |
| Score | 350 rows, 200 MC, 142/200 = 71.0% MC, avg tool calls 2.75, zero-tool rows 5 |
| Supersedes | SFT adapters as the current warm-index policy baseline unless a future trained policy beats it. |
| Known issues | Not synced locally at the time of this update. This is warm-index retrieval, not cold-start raw-video execution. MC accuracy does not by itself prove grounding quality. |

### RUN-V4.1-NO-TOOLS-PARAMETRIC-CONTROL

| Field | Value |
|-------|-------|
| Status | `current` control baseline |
| Date | 2026-04-21 |
| Goal | Measure how many V4.1 MC questions can be answered from question text, answer choices, and parametric knowledge without video evidence. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Model stack | `Qwen/Qwen3.5-9B` answer-only baseline |
| Tool mode | `no_tools` |
| Output artifact | Aristotle `eval/results/v4_1_no_tools_baseline.jsonl` |
| Score | 350 rows, 200 MC, 117/200 = 58.5% MC, avg tool calls 0 |
| Known issues | Correct answers are ungrounded by design. This is a benchmark-quality and grounding control, not a useful archival QA system. |

### RUN-V4.1-BASE-QWEN35-SIMULATED-COLD-CACHE

| Field | Value |
|-------|-------|
| Status | `current` cold-cache diagnostic baseline |
| Date | 2026-04-21 |
| Goal | Measure whether base Qwen3.5 can use the explicit artifact/cache schema when the visible run cache starts empty. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Model stack | Policy `Qwen/Qwen3.5-9B` without LoRA adapter; answerer `Qwen/Qwen3.5-9B` via `--answerer-backend local-base` |
| Tool mode | `simulated_cold_cache`; source index is immutable fixture, visible artifacts are isolated to the run cache |
| Output artifact | Aristotle `eval/results/v4_1_base_cold_cache_isolated_test.jsonl` and `v4_1_base_cold_cache_isolated_test.artifact_registry.json` |
| Score | 350 rows, 200 MC, 135/200 = 67.5% MC, avg tool calls 4.39 |
| Known issues | Diagnostic bridge only. It simulates artifact creation from existing indexes and does not run live CLAMS tools. Review traces before treating as a headline result. |

### RUN-V4.1-UNIFIED-SFT-V2-WARM-INDEX

| Field | Value |
|-------|-------|
| Status | `active-baseline` negative SFT result |
| Date | 2026-04-21 |
| Goal | Evaluate corrected native-tool SFT on unified-layer V4.1 trajectories against the V4.1 fixed-context test set. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Training trajectories | `training_data/output/trajectories_v4_1_unified_x1_r1_i1.jsonl` |
| Model stack | Policy `Qwen/Qwen3.5-9B` + unified V4.1 SFT v2 adapter; answerer `Qwen/Qwen3.5-9B` via `--answerer-backend local-base` |
| Tool mode | `warm_index` / `prebuilt_index`, all tools exposed |
| Output artifact | Aristotle `eval/results/v4_1_unified_v2_policy_shard0.jsonl` through `v4_1_unified_v2_policy_shard3.jsonl` |
| Score | 350 rows, 200 MC, 124/200 = 62.0% MC, avg tool calls 3.15 |
| Superseded by | `RUN-V4.1-BASE-QWEN35-WARM-INDEX` as the stronger current warm-index policy baseline. |
| Known issues | Underperforms the base policy and the no-adapter old V4 recovery comparison. Do not use as evidence that SFT improves tool use without further failure analysis. |

### RUN-V4.1-SFT-RECOVERY-OLD-ADAPTER-MAXTURNS3

| Field | Value |
|-------|-------|
| Status | `active-baseline` historical adapter baseline |
| Date | 2026-04-20 |
| Goal | Evaluate the older V4 recovery SFT policy against V4.1 fixed-context questions. |
| Input data | `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl` |
| Training trajectories | `training_data/output/trajectories_v4_oracle_locate_nobrowse_x3_retry_x1_interleaved_x1.jsonl` |
| Model stack | Policy `Qwen/Qwen3.5-9B` + adapter `training_data/output/qwen35-9b-sft-v4-oracle-locate-targeted-recovery-x3-r1-i1/adapter`; answerer `Qwen/Qwen3.5-9B` via `--answerer-backend local-base` |
| Tool mode | Normal policy retrieval over prebuilt index, all tools exposed, max turns 3 |
| Output artifact | `eval/results/v4_1_sft_recovery_alltools_localbase_test_maxturns3.jsonl` |
| Score | 350 rows, 200 MC, 133/200 = 66.5% MC, 150 free-text rows unscored, avg tool calls 2.43 |
| Supersedes | `RUN-V4.1-SFT-RECOVERY-OLD-ADAPTER-MAXTURNS5` as the more efficient setting with essentially tied accuracy. |
| Superseded by | `RUN-V4.1-BASE-QWEN35-WARM-INDEX` as the current stronger warm-index policy baseline. |
| Known issues | Adapter was trained on older V4 recovery trajectories, not fixed-context V4.1 evidence spans and not unified-layer trajectories. Use as historical adapter baseline, not final model result. |

Representative errors:

| Question ID | Expected | Predicted | Likely Failure Mode |
|-------------|----------|-----------|---------------------|
| `v3-301701ac-006` | `A`: Her Creole | `C`: Her Cadillac | Correct region found, but ASR/evidence wording and answer synthesis favored wrong distractor. |
| `v3-1279b012-000` | `B`: Large, gold-rimmed glasses | `C`: Large, black-rimmed glasses | Same visual localization ambiguity as the oracle failure. |
| `v3-b04a9248-000` | `A`: Barbara Grutter | `C`: Dennis Shields | OCR/text evidence retrieval failure. |

## Superseded Runs

### RUN-V4.1-MATCHED-TEXT-ORACLE-OLD-CONTEXT

| Field | Value |
|-------|-------|
| Status | `superseded` |
| Date | 2026-04-20 |
| Output artifact | `eval/results/v4_1_oracle_matched_text_test.jsonl` |
| Score | 350 rows, 200 MC, 177/200 = 88.5% MC |
| Superseded by | `RUN-V4.1-MATCHED-TEXT-ORACLE-FIXED-CONTEXT` |
| Reason | Evidence verifier scored full windows but stored only the first 800 characters. Some answer-bearing sentences were truncated out of the oracle prompt. |

### RUN-V4.1-CURRENT-TOOL-ORACLE-OLD-CONTEXT

| Field | Value |
|-------|-------|
| Status | `superseded` |
| Date | 2026-04-20 |
| Output artifact | `eval/results/v4_1_oracle_current_tool_test.jsonl` |
| Score | 350 rows, 200 MC, 171/200 = 85.5% MC |
| Superseded by | `RUN-V4.1-FIXED-CURRENT-TOOL-ORACLE`, then `RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V2` |
| Reason | Used pre-fix evidence context and older layer/replay behavior. |

### RUN-V4.1-FIXED-CURRENT-TOOL-ORACLE

| Field | Value |
|-------|-------|
| Status | `superseded` |
| Date | 2026-04-20 |
| Output artifact | `eval/results/v4_1_fixed_oracle_current_tool_test.jsonl` |
| Score | 350 rows, 200 MC, 170/200 = 85.0% MC |
| Superseded by | `RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V2` |
| Reason | Fixed evidence context but still had layer-name mismatch, timestamp drift, and less robust evidence truncation. |

### RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V1

| Field | Value |
|-------|-------|
| Status | `superseded` |
| Date | 2026-04-20 |
| Output artifact | `eval/results/v4_1_unified_oracle_current_tool_test.jsonl` |
| Score | 350 rows, 200 MC, 180/200 = 90.0% MC |
| Superseded by | `RUN-V4.1-UNIFIED-CURRENT-TOOL-ORACLE-V2` |
| Reason | V2 added better centered truncation of long direct-layer evidence windows. |

### RUN-V4.1-SFT-RECOVERY-OLD-ADAPTER-MAXTURNS5

| Field | Value |
|-------|-------|
| Status | `superseded` |
| Date | 2026-04-20 |
| Output artifact | `eval/results/v4_1_sft_recovery_alltools_localbase_test.jsonl` |
| Score | 350 rows, 200 MC, 132/200 = 66.0% MC, avg tool calls 2.62 |
| Superseded by | `RUN-V4.1-SFT-RECOVERY-OLD-ADAPTER-MAXTURNS3` |
| Reason | Max turns 3 was slightly more accurate and used fewer tools. Both remain older-adapter baselines. |

## Superseded / Unsynced Training Artifacts

### RUN-V4.1-UNIFIED-SFT

| Field | Value |
|-------|-------|
| Status | `superseded` by completed v2 eval |
| Date | 2026-04-20 |
| Goal | Train a Qwen3.5-9B SFT policy on unified-layer V4.1 trajectories. |
| Planned / reported adapter | `training_data/output/qwen35-9b-sft-v4_1-unified/` |
| Planned / reported trajectories | `training_data/output/trajectories_v4_1_unified_x1_r1_i1.jsonl` |
| Local artifact status | Not present locally at the time this ledger was written. Aristotle v2 eval is recorded above as `RUN-V4.1-UNIFIED-SFT-V2-WARM-INDEX`. |
| Required eval | No longer pending for the current comparison; use the recorded v2 result unless retraining a new SFT variant. |

## Current Interpretation

The current reportable ceiling is 97.0% matched-text oracle and 92.0%
unified current-tool oracle. The current reportable warm-index policy baseline
is the base `Qwen/Qwen3.5-9B` policy at 71.0% MC. The no-tools control is
58.5% MC, showing that raw MC accuracy must be separated from grounded
accuracy.

The next model-training result should not be compared only against the older
85.0% current-tool oracle or the older SFT adapter baseline, because both are
superseded for current interpretation. It should be compared against:

| Reference | Score |
|-----------|-------|
| No-tools parametric control | 58.5% MC |
| Base Qwen3.5 warm-index policy | 71.0% MC |
| Unified current-tool oracle | 92.0% MC |
| Matched-text oracle ceiling | 97.0% MC |

Interpretation: the base model already knows the native tool format well enough
for strong warm-index evidence retrieval. Future SFT/GRPO should be framed as
targeted improvement in groundedness, localization, efficiency, or
preference-conditioned artifact planning, not as basic tool-format teaching.
