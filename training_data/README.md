# Training Data and Policy Training Pipeline

This directory contains both:

- the **current** index-backed trajectory / SFT / GRPO workflow used for V3 benchmark experiments
- **earlier synthetic pipeline code** inspired by NVIDIA ToolOrchestra, retained for provenance and comparison

The current research path is no longer purely synthetic. It is centered on real video indexes, benchmark-derived questions, and tool-use policies evaluated in two regimes: warm-index archive QA and simulated cold-cache artifact orchestration.

| Regime | Training/Eval Substrate | What The Policy Learns |
|--------|-------------------------|------------------------|
| Warm-index archive QA | Existing ASR/OCR/caption/speaker layers in `data/video_indexes/` | Query and inspect the right evidence layers with provenance. |
| Simulated cold-cache orchestration | Read-only source indexes plus a disposable artifact registry | Decide which artifacts to create, reuse, read, or search before answering. |

## Current Pipeline

### 1. Inputs

The current training flow starts from:
- layered video indexes in `data/video_indexes/`
- curated V3/V4.1 question splits in `qa-data/raw/`, `qa-data/benchmark/v3/`, and `qa-data/benchmark/v4_1/`

Important current split files:
- `qa-data/raw/qa_v3_train.jsonl`
- `qa-data/raw/qa_v3_val.jsonl`
- `qa-data/raw/qa_v3_test.jsonl`
- `qa-data/benchmark/v3/train_benchmark.jsonl`
- `qa-data/benchmark/v3/val_benchmark.jsonl`
- `qa-data/benchmark/v3/test_benchmark.jsonl`
- `qa-data/benchmark/v4_1/train_benchmark.jsonl` (superseded by fixed-context split for current training)
- `qa-data/benchmark/v4_1/val_benchmark.jsonl` (superseded by fixed-context split for current training)
- `qa-data/benchmark/v4_1/test_benchmark.jsonl` (superseded by fixed-context split for current evaluation)
- `qa-data/benchmark/v4_1_fixed_context/train_benchmark.jsonl` (current preferred V4.1 training input)
- `qa-data/benchmark/v4_1_fixed_context/val_benchmark.jsonl`
- `qa-data/benchmark/v4_1_fixed_context/test_benchmark.jsonl`

Optional stricter subsets also exist, such as:
- `*_gold.jsonl`
- `*_verified.jsonl`
- `*_filtered.jsonl`

### 2. Tool Simulation Substrate

Training does not currently call live CLAMS tools during GRPO rollouts.

Instead:
- tools are defined in the environment
- tool outputs are simulated from real index layers
- model and task-mode differences are represented through the index and shared simulation logic

In warm-index runs, tools such as `search_transcript` and `search_ocr` are index lookups over layers that already exist. In cold-cache runs, those searches should only be available after the corresponding artifact has been created or reused.

The core current files are:
- [construct_tool_trajectories.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/construct_tool_trajectories.py)
- [run_grpo_env.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/run_grpo_env.py)
- [prompt_configs.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/prompt_configs.py)

Current all-tools evidence space includes:
- `search_transcript(query=...)`
- `search_ocr(query=...)`
- `run_asr(model=...)`
- `browse_timeline(start_time=..., end_time=...)`
- `extract_text(timestamp, model=...)`
- `caption_frame(timestamp, model=..., task_mode=...)`
- `identify_speakers()`
- `detect_text_scenes()`

For the current accuracy-focused SFT path, all tools are exposed. The trajectory design still teaches explicit evidence-location behavior instead of relying on broad, answer-like summaries.

### 3. Current Trajectory Construction

The current trajectory path is evidence-driven rather than purely synthetic.

Benchmark questions contain grounding and evidence timing information. For V4.1, the preferred supervision is `evidence_spans`; legacy fixed 30-second `source_segment_times` should not be treated as the evidence target.

Current trajectory construction lives primarily in:
- [construct_tool_trajectories.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/construct_tool_trajectories.py)

This is the main source of policy-oriented training data for the current setup.

### 4. Current SFT Preparation

The repo contains several SFT preparation paths because the project evolved through multiple tool APIs.

For current work, the important distinction is:
- older SFT formatting code exists for legacy tool APIs
- current policy work is moving toward native tool-calling with the same tool semantics used by the GRPO environment and canonical evaluation path

Relevant files include:
- `run_native_sft.py` (current SFT entry point, uses Qwen3.5 native tool format)
- `prepare_native_tool_sft.py` (legacy data prep, superseded by trajectory-embedded tool schemas)
- `prepare_sft_from_trajectories.py` (legacy data prep)
- `run_sft.py` (legacy, uses older tool format)

These files should be interpreted in light of the current tool API and benchmark split you are using.

### 5. Current GRPO Training

The main active RL training path is:
- [run_grpo_env.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/run_grpo_env.py)

This script:
- exposes environment methods as tools
- simulates tool outputs from index layers
- computes reward from answer correctness plus cost/efficiency terms
- logs tool usage and rollout behavior

The policy is learning:
- which tool to call
- which model variant to use
- which task mode to use for captioning
- when to stop gathering evidence

### 6. Index Variant Generation

As the index expands, training can simulate richer tool differences without live inference.

Current helper scripts include:
- `run_whisper_asr.py`
- `generate_caption_variants.py`

These populate additional index layers such as:
- `asr_whisper`
- `caption_<model>_<task_mode>`

This is the mechanism by which the policy can learn real differences between:
- Parakeet vs Whisper
- SmolVLM vs Qwen3.5 variants
- caption task modes such as `general_scene`, `text_focus`, `person_identity`, `action_event`

## Current Recommended Flow

The practical current flow is:

1. Build / update index layers
2. Generate or refresh benchmark-derived trajectories
3. Run no-tools and base-model controls before training adapters
4. Prepare SFT data aligned with the current tool API if a targeted SFT hypothesis remains
5. Train SFT/GRPO only if the run is expected to improve on a concrete base-policy failure mode
6. Evaluate with `eval/run_policy_answerer_eval.py`
7. Score with `eval/score_predictions.py`

## Current V4.1 Training State

Run status, scores, supersession, and representative errors are tracked in [docs/RUN_LEDGER.md](/Users/kelleylynch/clams/clams-agent-prototype/docs/RUN_LEDGER.md). This README only summarizes the training implications.

Current reportable warm-index baselines:

| Condition | Status | File | Rows | MC Accuracy | Avg Tools |
|-----------|--------|------|------|-------------|-----------|
| Base `Qwen/Qwen3.5-9B` + tools | current baseline | Aristotle `eval/results/v4_1_base_model_shard*.jsonl` | 350/350 | 142/200 = 71.0% | 2.75 |
| No-tools parametric control | current control | Aristotle `eval/results/v4_1_no_tools_baseline.jsonl` | 350/350 | 117/200 = 58.5% | 0 |
| Unified V4.1 SFT v2 + tools | negative SFT result | Aristotle `eval/results/v4_1_unified_v2_policy_shard*.jsonl` | 350/350 | 124/200 = 62.0% | 3.15 |
| SFT policy, old V4 recovery adapter, max turns 3 | historical baseline | `eval/results/v4_1_sft_recovery_alltools_localbase_test_maxturns3.jsonl` | 350/350 | 133/200 = 66.5% | 2.43 |

The base-model result means SFT is not the default next step. Any future SFT run should target a diagnosed failure mode, such as evidence-span localization, OCR/ASR conflict resolution, or policy recovery after an unhelpful tool observation.

Adapter lineage for that baseline:

- Base model: `Qwen/Qwen3.5-9B`
- Adapter: `training_data/output/qwen35-9b-sft-v4-oracle-locate-targeted-recovery-x3-r1-i1/adapter`
- Training file: `training_data/output/trajectories_v4_oracle_locate_nobrowse_x3_retry_x1_interleaved_x1.jsonl`
- Training mix: `2,109` trajectories = `1,395` standard, `386` consecutive retry-recovery, `328` interleaved recovery
- Status: valid historical adapter baseline, but not the current best warm-index policy because base `Qwen/Qwen3.5-9B` performs better without the adapter

Current oracle references for training targets:

| Reference | Status | MC Accuracy | Meaning |
|-----------|--------|-------------|---------|
| Matched-text oracle | current | 194/200 = 97.0% | Questions are mostly answerable when verified evidence text is shown directly. |
| Unified current-tool oracle | current | 184/200 = 92.0% | Tool-visible oracle after unified layer names, direct verified-layer reads, and centered truncation. |
| Old fixed current-tool oracle | superseded | 170/200 = 85.0% | Do not use as the current ceiling; superseded by layer/oracle fixes. |

If another SFT run is attempted, it should use fixed-context, unified-layer trajectories rather than the earlier `qa-data/benchmark/v4_1/train_benchmark.jsonl` trajectories, and it should be evaluated against the base-model warm-index baseline.

Fixed benchmark split:

```text
qa-data/benchmark/v4_1_fixed_context/
```

Fixed verification stats:

- Train: `439` accepted, `25` rejected
- Val: `78` accepted, `2` rejected
- Test: `350` accepted, `23` rejected
- Combined: `867` accepted, `50` rejected
- Average source-to-stored matched-text recall on test improved from `0.733` to `0.935`
- Low-recall test spans (`<=0.5`) dropped from `216` to `38`

Superseded trajectory artifact:

```text
training_data/output/trajectories_v4_1_evidence_spans_x1_retry_x1_interleaved_x1.jsonl
```

This file is useful provenance, but it was generated before the latest layer unification and oracle-window fixes. It contains `1,279` trajectories from `439` train questions: `439` standard, `439` consecutive retry-recovery, and `401` interleaved recovery.

Pending unified SFT artifacts reported in docs but not present locally at the time of this update:

```text
training_data/output/trajectories_v4_1_unified_x1_r1_i1.jsonl
training_data/output/qwen35-9b-sft-v4_1-unified/
```

Sync these from Aristotle before reporting them as complete or training/evaluating from them locally.

## V4 Historical All-Tools SFT Run

This historical V4 run prioritized answer accuracy before introducing cache/artifact-management experiments.

Trajectory generation:

```bash
python3 training_data/construct_tool_trajectories.py \
  --questions qa-data/raw/qa_v3_train.jsonl \
  --index-dir data/video_indexes \
  --output training_data/output/trajectories_v4_oracle_locate_targeted_x3.jsonl \
  --no-noise \
  --trajectories-per-question 3
```

Expected trajectory properties:

- 1,395 training examples from 465 train questions
- Full tool schema exposed; trajectories still emphasize targeted evidence gathering
- Evidence-derived tool calls over the read-only prebuilt index
- Qwen3.5 native tool-calling format

Training target:

```bash
CUDA_VISIBLE_DEVICES=1 python3 training_data/run_native_sft.py \
  --data training_data/output/trajectories_v4_oracle_locate_targeted_x3.jsonl \
  --model Qwen/Qwen3.5-9B \
  --output training_data/output/qwen35-9b-sft-v4-oracle-locate-targeted-x3 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 8e-5 \
  --max-seq-length 4096 \
  --lora-r 16 \
  --lora-alpha 16
```

Post-training evaluation should use the same Qwen3.5 family for consistency. On a single GPU, prefer `--answerer-backend local-base`; it reuses the loaded policy base model with the LoRA adapter disabled for answer synthesis, avoiding a second Qwen3.5 server process.

```bash
CUDA_VISIBLE_DEVICES=2 \
python3 eval/run_policy_answerer_eval.py \
  --benchmark qa-data/benchmark/v4/test_benchmark.jsonl \
  --index-dir data/video_indexes \
  --output eval/results/v4_sft_oracle_locate_targeted_x3_test.jsonl \
  --policy-adapter training_data/output/qwen35-9b-sft-v4-oracle-locate-targeted-x3/adapter \
  --policy-base Qwen/Qwen3.5-9B \
  --answerer-backend local-base \
  --max-turns 3
```

This run treats the source index as read-only simulation data. It does not test cold-cache behavior, artifact persistence, FAISS/vector-index cache management, or live CLAMS execution.
On Aristotle specifically, CUDA device order differs from `nvidia-smi` order: `CUDA_VISIBLE_DEVICES=2` maps to physical GPU3 in the current environment.

### Retry-Recovery Augmentation

The all-tools SFT run exposed a trajectory-design issue: standard evidence-derived traces often jump from a locator tool to the hidden gold timestamp. That teaches tool format, but not recovery from inspecting the wrong region.

Observed eval behavior from the first SFT run:

- The clean eval was valid after restarting the Qwen3.5 answerer server and rerunning with the current tool contract.
- Partial accuracy settled around the low-to-mid 60s during the run, below the previous correctness-only GRPO result and below the RAG baseline.
- Wrong answers were usually not malformed tool calls. They were evidence-location failures: failed search hits, ASR over the wrong range, visual inspection of the wrong frame, or abstaining because the gathered evidence did not support any option.
- This suggests the model learned the tool API but did not learn a robust recovery policy for wrong locations.

For the next SFT iteration, add retry-recovery trajectories. Consecutive retry trajectories teach direct same-tool recovery:

```bash
python3 training_data/construct_tool_trajectories.py \
  --questions qa-data/raw/qa_v3_train.jsonl \
  --index-dir data/video_indexes \
  --output training_data/output/trajectories_v4_oracle_locate_targeted_x3_retry_x1.jsonl \
  --no-noise \
  --trajectories-per-question 3 \
  --retry-trajectories-per-question 1
```

This generates the standard evidence-targeted trajectories plus additional examples where the policy:

- locates a plausible but wrong time region
- calls the same inspection tool/model on that wrong region
- observes that the output is not sufficient
- retries the same tool/model on a different region
- answers from the recovered evidence

These traces are intended to teach recovery behavior such as `run_asr -> run_asr`, `caption_frame -> caption_frame`, and `extract_text -> extract_text`. They also align with the cost-aware reward design: a correct rollout that inspects wrong locations first should score lower than a correct rollout that finds the evidence directly, because the extra inspection calls increase total tool cost.

Interleaved retry trajectories teach non-consecutive recovery, where another modality changes the next location choice:

```bash
python3 training_data/construct_tool_trajectories.py \
  --questions qa-data/raw/qa_v3_train.jsonl \
  --index-dir data/video_indexes \
  --output training_data/output/trajectories_v4_oracle_locate_targeted_x3_retry_x1_interleaved_x1.jsonl \
  --no-noise \
  --trajectories-per-question 3 \
  --retry-trajectories-per-question 1 \
  --interleaved-retry-trajectories-per-question 1
```

The interleaved pattern is:

- `search_transcript`
- `run_asr` on a plausible but wrong segment
- use an intervening modality such as `caption_frame`, `search_ocr -> extract_text`, or `detect_text_scenes -> extract_text`
- `run_asr` again on a different segment suggested by the intervening evidence

Aristotle generation stats for the current recovery dataset:

- Total trajectories: `2,109`
- Standard evidence-targeted trajectories: `1,395`
- Consecutive retry-recovery trajectories: `386`
- Interleaved recovery trajectories: `328`
- Legacy timeline-summary schema/tool-call/string mentions: `0`
- Interleaved patterns include `run_asr -> caption_frame -> run_asr`, `run_asr -> search_ocr -> extract_text -> run_asr`, and `run_asr -> detect_text_scenes -> extract_text -> run_asr`.

Some existing Aristotle artifact filenames still contain the historical `nobrowse` token from the earlier wording. New artifacts should use `targeted` naming.

## Key Current Files

| File | Role |
|------|------|
| `construct_tool_trajectories.py` | Shared tool simulation and evidence-driven trajectory construction |
| `run_grpo_env.py` | Main GRPO environment-based training path |
| `prompt_configs.py` | Model/task-mode prompt templates for indexed variant generation and future live integration |
| `generate_caption_variants.py` | Populate caption-model/task-mode index layers |
| `run_whisper_asr.py` | Populate Whisper ASR layers |
| `run_native_sft.py` | SFT training entry point (native Qwen3.5 tool format) |
| `prepare_native_tool_sft.py` | Legacy SFT data prep (superseded by trajectory-embedded schemas) |
| `run_sft.py` | Legacy SFT training (older tool format) |

## Earlier / Legacy Pipeline

This directory also contains an earlier synthetic training-data pipeline.

That older pipeline generated:
1. synthetic multimedia tasks
2. synthetic tool-calling trajectories
3. synthetic SFT examples
4. optional DPO-style preference pairs

This work was based on the ToolOrchestra-style idea that a small orchestration model can be bootstrapped from simulated tool-use traces.

That pipeline is still useful historically, and some of its files remain:
- `generate_tasks.py`
- `prepare_sft_data.py`
- `run_pipeline.py`
- `tools.json`

However, it should now be read as an **earlier version** of the training story, not the best description of the current V3 benchmark-driven workflow.

## Earlier Synthetic Pipeline Details

The legacy synthetic flow worked roughly as follows:

### Synthetic tasks
- generated broad CLAMS-style multimedia analysis requests
- categories included slate extraction, chyron extraction, credits, transcription, indexing, entity extraction, and quality control

### Synthetic tool trajectories
- formatted step-by-step reasoning and tool calls
- used simulated tool outputs

### Synthetic SFT formatting
- built conversation-style training examples with reasoning blocks and tool-call markup

### Optional DPO-style preference pairs
- generated chosen/rejected completion pairs for preference learning

This material remains in the repo because:
- it influenced the current policy-training design
- some scripts are still useful utilities
- it preserves the project’s evolution

## Current Caveat

Not every file in `training_data/` has been fully migrated to the newest tool API yet.

In particular, when refreshing trajectories or SFT data, it is important to verify that:
- the tool schemas match the current GRPO environment
- model/task-mode parameters are represented where expected
- outputs are aligned with the current canonical evaluation path

The repo currently contains both modernized and legacy paths, so check the specific script before using it in a new training run.

## References

- [NVIDIA ToolOrchestra](https://github.com/NVlabs/ToolOrchestra)
- [CLAMS Platform](https://clams.ai/)
- [eval/README.md](/Users/kelleylynch/clams/clams-agent-prototype/eval/README.md)
