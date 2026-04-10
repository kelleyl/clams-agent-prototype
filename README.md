# CLAMS-Agent: Agent-Orchestrated Video Indexing and QA

CLAMS-Agent is a research codebase for training and evaluating LLM-based agents that answer questions about long-form archival broadcast video by selecting tools over structured video indexes.

The current pipeline is built around:
- layered video indexes in `data/video_indexes/`
- V3 benchmark generation and curation in `qa-data/`
- trajectory construction and policy training in `training_data/`
- policy + answerer evaluation in `eval/`

## Current Experimental Flow

The current workflow is:

1. **Build layered video indexes** from AAPB videos.
   Indexes include ASR, OCR, visual captions, speakers, chapters, entities, and increasingly model-specific variant layers such as Whisper ASR or caption variants by model/task mode.

2. **Generate and curate V3 QA** from the indexes.
   The current benchmark generation flow is full-context QA generation from the annotated broadcast plus post-generation filtering and verification. Canonical split files live under `qa-data/raw/` and `qa-data/benchmark/v3/`.

3. **Construct tool-use trajectories** against real indexes.
   Trajectories are derived from the evidence chain for benchmark questions and then converted into SFT / policy-training data.

4. **Train tool-selection policies** with SFT and GRPO.
   The current GRPO setup uses index-backed simulation rather than live CLAMS execution at training time. The policy learns to choose tools, models, and task modes over cached index layers.

5. **Evaluate with a policy + answerer architecture**.
   The policy gathers evidence through tool calls; a separate answerer model answers from the gathered evidence. Final metrics are computed post-hoc from saved prediction artifacts.

## Canonical Paths

### Benchmark and QA
- Canonical V3 split files: `qa-data/benchmark/v3/`
- Gold / verified subsets for stricter evaluation: `qa-data/benchmark/v3/*_gold.jsonl`, `*_verified.jsonl`
- Raw split artifacts: `qa-data/raw/qa_v3_*.jsonl`

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

## Current Evaluation Pattern

For the current V3 experiments, evaluation should be treated as a two-step process:

1. Generate predictions:

```bash
python eval/run_policy_answerer_eval.py \
  --benchmark qa-data/benchmark/v3/test_benchmark_gold.jsonl \
  --index-dir data/video_indexes \
  --output eval/results/my_policy_test.jsonl \
  --policy-adapter training_data/output/my_adapter \
  --answerer-model qwen3:8b
```

2. Score predictions:

```bash
python eval/score_predictions.py \
  --predictions eval/results/my_policy_test.jsonl \
  --benchmark qa-data/benchmark/v3/test_benchmark_gold.jsonl
```

`run_policy_answerer_eval.py` is the prediction-generation step. Final MC / free-text metrics should come from `score_predictions.py`.

## Current Training Pattern

The active policy training path is index-backed rather than live CLAMS execution:

- the agent calls tools in the GRPO environment
- tool outputs are simulated from real index layers
- model and task-mode differences are represented through indexed layer variants and simulation logic

This lets the policy learn tool selection without paying live inference cost during training.

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

## Documentation

- [eval/README.md](/Users/kelleylynch/clams/clams-agent-prototype/eval/README.md): benchmark history, split artifacts, canonical eval flow, results
- [training_data/README.md](/Users/kelleylynch/clams/clams-agent-prototype/training_data/README.md): current training pipeline and legacy synthetic pipeline
- [CLAMS_AGENT_SPEC.md](/Users/kelleylynch/clams/clams-agent-prototype/CLAMS_AGENT_SPEC.md): broader system vision

## Related

- [CLAMS Project](https://clams.ai/)
- [CLAMS Apps Directory](https://apps.clams.ai/)
- [MMIF Specification](https://mmif.clams.ai/)
- [American Archive of Public Broadcasting](https://americanarchive.org/)
