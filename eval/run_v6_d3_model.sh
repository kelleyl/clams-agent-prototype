#!/bin/bash
# D3 model-comparison chain for one ollama model on the v6.0 test split.
# Runs: full-stack agent (two-stage, prebuilt_index) -> score -> no_tools floor -> score.
#
# Usage: bash eval/run_v6_d3_model.sh <ollama_model> <output_tag> <done_marker>
# Example: bash eval/run_v6_d3_model.sh gemma3:27b-it-qat gemma3-27b GEMMA3
set -euo pipefail

MODEL="$1"
TAG="$2"
MARKER="$3"

cd "$(dirname "$0")/.."
BM=qa-data/benchmark/v6/test.jsonl
JUDGE_URL=http://localhost:11434
JUDGE_MODEL=llama3.3:latest

echo "=== [$(date)] agent_warm ${MODEL} ==="
python3 eval/run_policy_answerer_eval.py \
    --benchmark "$BM" \
    --output "eval/results/v6_agent_warm_${TAG}.jsonl" \
    --agent-mode two-stage \
    --policy-backend ollama --policy-ollama-model "$MODEL" \
    --answerer-backend ollama --answerer-model "$MODEL" \
    --tool-execution-mode prebuilt_index \
    --max-turns 20 \
    --resume

echo "=== [$(date)] score agent_warm ${MODEL} ==="
python3 eval/score_predictions.py \
    --predictions "eval/results/v6_agent_warm_${TAG}.jsonl" \
    --benchmark "$BM" --mode auto \
    --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL"

echo "=== [$(date)] no_tools ${MODEL} ==="
python3 eval/run_policy_answerer_eval.py \
    --benchmark "$BM" \
    --output "eval/results/v6_no_tools_${TAG}.jsonl" \
    --agent-mode single --max-turns 0 \
    --policy-backend ollama --policy-ollama-model "$MODEL" \
    --resume

echo "=== [$(date)] score no_tools ${MODEL} ==="
python3 eval/score_predictions.py \
    --predictions "eval/results/v6_no_tools_${TAG}.jsonl" \
    --benchmark "$BM" --mode auto \
    --judge-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL"

echo "D3_${MARKER}_DONE"
