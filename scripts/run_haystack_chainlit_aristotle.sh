#!/usr/bin/env bash
set -euo pipefail

export CLAMS_HAYSTACK_PROVIDER="${CLAMS_HAYSTACK_PROVIDER:-vllm}"
export CLAMS_HAYSTACK_MODEL="${CLAMS_HAYSTACK_MODEL:-Qwen/Qwen3.5-9B}"
export CLAMS_HAYSTACK_VLLM_URL="${CLAMS_HAYSTACK_VLLM_URL:-http://localhost:8890/v1}"
export CLAMS_HAYSTACK_CATALOG_ROOT="${CLAMS_HAYSTACK_CATALOG_ROOT:-data/haystack_catalog}"
export CLAMS_HAYSTACK_MMIF_ROOTS="${CLAMS_HAYSTACK_MMIF_ROOTS:-data/chicago:data/enriched:data/combined_outputs:data/pipeline_outputs:data/scene_summaries_smoke:data/qwen_outputs}"
export CLAMS_HAYSTACK_TOOL_SERVICES="${CLAMS_HAYSTACK_TOOL_SERVICES:-data/tool_services.json}"
export CLAMS_HAYSTACK_EXECUTE_SERVICES="${CLAMS_HAYSTACK_EXECUTE_SERVICES:-0}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export PYTHONPATH=".:${PYTHONPATH:-}"

PORT="${PORT:-8784}"
HOST="${HOST:-0.0.0.0}"

"${CHAINLIT_BIN:-.venv/bin/chainlit}" run clams_haystack/chainlit_app.py --host "$HOST" --port "$PORT"
