#!/bin/bash
# Arm comparison for the "serialized-index-as-prose" experiment. Hold the
# generator (gemma3:27b) and every gate fixed; vary ONLY the evidence the
# generator reads: raw ASR slice (arm A) vs DP video-description window (arm B1).
# densify + round-trip are identical across arms (round-trip verifies against the
# RAW index -- the faithfulness guard), so the only variable is generation evidence.
cd ~/clams_apps/clams-agent-prototype || exit 1
PY=.venv/bin/python
OLL=http://localhost:11434/v1
GEN=gemma3:27b-it-qat
VIDS="cpb-aacip-507-1v5bc3tf81 cpb-aacip-507-5x2599zp0m cpb-aacip-507-3x83j39n00 cpb-aacip-507-f47gq6rq94 cpb-aacip-507-251fj2b03t cpb-aacip-507-ns0ks6jx8k"
PANEL="llama3.1:8b,qwen3:8b,gemma3:27b-it-qat,llama3.3:latest"
VERIFY=llama3.3:latest

ARMS=(
  "slice|A-slice"
  "description|B1-desc"
)

for spec in "${ARMS[@]}"; do
  MODE="${spec%%|*}"; SLUG="${spec##*|}"
  echo "===== ARM $MODE -> $SLUG  $(date +%H:%M:%S) ====="
  for V in $VIDS; do
    $PY scripts/generate_qa_needdown.py --all --only-video "$V" \
        --vllm-url "$OLL" --model "$GEN" --evidence-mode "$MODE" >/dev/null 2>&1
    $PY scripts/densify_questions.py --video "$V" --dp-url "$OLL" --dp-model "$GEN" >/dev/null 2>&1
  done
  echo "  gen+densify done $(date +%H:%M:%S)"
  $PY scripts/blind_panel.py --video $VIDS --models "$PANEL" 2>&1 | tail -14
  for V in $VIDS; do
    $PY scripts/roundtrip_check.py --video "$V" --rt-url $OLL --rt-model $VERIFY --judge-model $VERIFY >/dev/null 2>&1
  done
  echo "  gates done $(date +%H:%M:%S)"
  mkdir -p "data/gen_compare/$SLUG"
  cp data/qa_needdown/*.json "data/gen_compare/$SLUG/"
  echo "===== $SLUG archived $(date +%H:%M:%S) ====="
done
echo "=== ALL DONE $(date +%H:%M:%S) ==="
