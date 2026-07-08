#!/bin/bash
# v6 multi-video sweep: rebuild salience -> generate -> densify (gemma-4 on vLLM),
# then the multi-model blind panel across ALL videos (each panel model loaded once).
cd ~/clams_apps/clams-agent-prototype || exit 1
PY=.venv/bin/python
GENURL=http://localhost:8202/v1
GEN=gemma-4-26b
VIDS="cpb-aacip-507-1v5bc3tf81 cpb-aacip-507-5x2599zp0m cpb-aacip-507-3x83j39n00 cpb-aacip-507-f47gq6rq94 cpb-aacip-507-251fj2b03t cpb-aacip-507-ns0ks6jx8k"

echo "=== PHASE 1: salience + generate + densify (gemma-4) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  $PY scripts/build_salience_map.py --only-video "$V" >/dev/null 2>&1
  $PY scripts/generate_qa_needdown.py --all --only-video "$V" --vllm-url $GENURL --model "$GEN" >/dev/null 2>&1
  $PY scripts/densify_questions.py --video "$V" --dp-url $GENURL --dp-model "$GEN" >/dev/null 2>&1
  echo "  gen+densify done: $V $(date +%H:%M:%S)"
done

echo "=== PHASE 2: blind panel across all videos $(date +%H:%M:%S) ==="
$PY scripts/blind_panel.py --video $VIDS
echo "=== DONE $(date +%H:%M:%S) ==="
