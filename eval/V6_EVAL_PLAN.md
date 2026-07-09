# v6 Evaluation Plan (5 days, 2026-07-09 .. 2026-07-14)

Decisions (2026-07-09, with KL): v6 is the primary benchmark with grounded accuracy as
the headline metric and one v4.1 MC continuity row per model; model comparison swaps the
FULL agent stack per model plus one answerer-only ablation; VLM-direct baseline runs on
the visual subset with evidence-window frames.

## Benchmark

`qa-data/benchmark/v6/benchmark_combined.jsonl` (release candidate, 796 rows: 603 speech
free-text, 160 visual free-text, 33 exploration retrieval-set). Human ratification in
progress; prediction runs use the RC and scoring re-filters to the final row set by id.
Continuity: `qa-data/benchmark/v4_1_fixed_context/` test MC (350 rows).

## Configurations

Levels (v6, all models unless noted):
1. no-tools        - parametric floor; expect ~0 on the robust stratum (benchmark headline)
2. RAG             - keyword retrieval over the index, top segments stuffed in prompt
3. CLAMS-Agent     - warm-index policy + answerer
4. oracle          - matched evidence provided directly (ceiling; one model suffices)

Models (all on aristotle):
- Qwen3.5-9B (vLLM, current stack)     - primary
- qwen3.6:35b-a3b (ollama)             - the "vs newer/larger" comparison
- gemma3:27b-it-qat (ollama)
- llama3.3:latest 70B (ollama)
Ablation: best policy + swapped answerers (one row per answerer) to separate tool
selection from comprehension.
VLM-direct (visual subset only, 160 q): qwen2.5vl:7b or Qwen3-VL-8B; frames sampled from
the question's evidence window via data/thumbnails_5s (idx = ms//5000+1); no index text.

## Metrics

- Raw accuracy: free text via utils/answer_match.decide_match (matcher + LLM judge);
  exploration via set F1 (>=0.5 counts correct for aggregate rows; report mean F1 too).
- GROUNDED accuracy (primary): answer counts only if retrieved evidence supports it
  (adapt eval/score_grounding.py). Raw is reported alongside.
- Strata cuts: blind stratum (robust / leaky-to-few / leaky-to-some), family
  (speech/visual/exploration), 5W role, cell. The robust stratum is the headline: raw ~=
  grounded there by construction. The leaky-to-large slice tests grounded retrieval vs
  parametric recall (thesis claim).
- v4.1 continuity: MC exact match, one row per model, warm-index config only.

## Day plan

- D1 (07-09/10): harness adaptation. v6 loader in run_policy_answerer_eval.py, free-text
  scoring path, set-F1 scorer for exploration, grounding instrumentation. Smoke run
  (20 q) per config on Qwen3.5-9B. Stand up vLLM 9B; verify ollama models respond.
- D2 (07-10/11): core runs on v6 with Qwen3.5-9B: no-tools, RAG, agent, oracle.
  v4.1 continuity row. Grounding scoring.
- D3 (07-11/12): model sweep: full-stack agent + no-tools floor for qwen3.6:35b-a3b,
  gemma3:27b, llama3.3:70b (sharded on aristotle, resumable).
- D4 (07-12/13): answerer-only ablation; VLM-direct on visual subset; exploration
  scoring; strata analysis.
- D5 (07-13/14): buffer, reruns, tables (comparative table + grounded-accuracy figure)
  into dissertation Ch5; fill or descope remaining placeholders.

## Outputs

- eval/results/v6_<config>_<model>.jsonl (+ .summary.json)
- eval/results/v6_comparison_table.md (generated)
- Manuscript: fills "Comparative evaluation results table" and "Updated results
  discussion" placeholders; VLM-direct placeholder resolved.
