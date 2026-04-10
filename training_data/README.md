# Training Data and Policy Training Pipeline

This directory contains both:

- the **current** index-backed trajectory / SFT / GRPO workflow used for V3 benchmark experiments
- **earlier synthetic pipeline code** inspired by NVIDIA ToolOrchestra, retained for provenance and comparison

The current research path is no longer purely synthetic. It is centered on real video indexes, benchmark-derived questions, and tool-use policies trained over simulated index-backed tools.

## Current Pipeline

### 1. Inputs

The current training flow starts from:
- layered video indexes in `data/video_indexes/`
- curated V3 question splits in `qa-data/raw/` and `qa-data/benchmark/v3/`

Important current split files:
- `qa-data/raw/qa_v3_train.jsonl`
- `qa-data/raw/qa_v3_val.jsonl`
- `qa-data/raw/qa_v3_test.jsonl`
- `qa-data/benchmark/v3/train_benchmark.jsonl`
- `qa-data/benchmark/v3/val_benchmark.jsonl`
- `qa-data/benchmark/v3/test_benchmark.jsonl`

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

The core current files are:
- [construct_tool_trajectories.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/construct_tool_trajectories.py)
- [run_grpo_env.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/run_grpo_env.py)
- [prompt_configs.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/prompt_configs.py)

Current tool space includes:
- `run_asr(model=...)`
- `extract_text(timestamp, model=...)`
- `caption_frame(timestamp, model=..., task_mode=...)`
- `identify_speakers()`
- `detect_text_scenes()`
- `browse_timeline(...)`

### 3. Current Trajectory Construction

The current trajectory path is evidence-driven rather than purely synthetic.

Benchmark questions already contain grounding and source-time information. That information is used to construct tool-use traces that are consistent with the index and with the evidence needed to answer the question.

Current trajectory construction lives primarily in:
- [construct_tool_trajectories.py](/Users/kelleylynch/clams/clams-agent-prototype/training_data/construct_tool_trajectories.py)

This is the main source of policy-oriented training data for the current setup.

### 4. Current SFT Preparation

The repo contains several SFT preparation paths because the project evolved through multiple tool APIs.

For current work, the important distinction is:
- older SFT formatting code exists for legacy tool APIs
- current policy work is moving toward native tool-calling with the same tool semantics used by the GRPO environment and canonical evaluation path

Relevant files include:
- `prepare_native_tool_sft.py`
- `prepare_sft_from_trajectories.py`
- `run_sft.py`

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
3. Prepare SFT data aligned with the current tool API
4. Train SFT baseline
5. Run GRPO on the same tool semantics
6. Evaluate with `eval/run_policy_answerer_eval.py`
7. Score with `eval/score_predictions.py`

## Key Current Files

| File | Role |
|------|------|
| `construct_tool_trajectories.py` | Shared tool simulation and evidence-driven trajectory construction |
| `run_grpo_env.py` | Main GRPO environment-based training path |
| `prompt_configs.py` | Model/task-mode prompt templates for indexed variant generation and future live integration |
| `generate_caption_variants.py` | Populate caption-model/task-mode index layers |
| `run_whisper_asr.py` | Populate Whisper ASR layers |
| `prepare_native_tool_sft.py` | Native tool-calling SFT formatting path |
| `run_sft.py` | SFT training entry point |

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
