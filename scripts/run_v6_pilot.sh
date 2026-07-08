#!/bin/bash
# v6 pilot sweep over a few NewsHour videos. Two phases to avoid model swaps:
# phase 1 generate+densify (gemma), phase 2 blind+round-trip (llama3.3).
cd ~/clams_apps/clams-agent-prototype || exit 1
PY=.venv/bin/python
URL=http://localhost:11434/v1
GEN=gemma3:27b-it-qat
VERIFY=llama3.3:latest
VIDS="cpb-aacip-507-1v5bc3tf81 cpb-aacip-507-5x2599zp0m cpb-aacip-507-3x83j39n00 cpb-aacip-507-f47gq6rq94 cpb-aacip-507-251fj2b03t cpb-aacip-507-ns0ks6jx8k"

echo "=== PHASE 1: generate + densify (gemma) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  echo "[gen+densify] $V $(date +%H:%M:%S)"
  $PY scripts/generate_qa_needdown.py --all --only-video "$V" --vllm-url $URL --model "$GEN" >/dev/null 2>&1
  $PY scripts/densify_questions.py --video "$V" --dp-url $URL --dp-model "$GEN" >/dev/null 2>&1
done

echo "=== PHASE 2: blind + round-trip (llama3.3) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  echo "[verify] $V $(date +%H:%M:%S)"
  $PY scripts/blind_check.py --video "$V" --blind-url $URL --blind-model "$VERIFY" --judge-model "$VERIFY" >/dev/null 2>&1
  $PY scripts/roundtrip_check.py --video "$V" --rt-url $URL --rt-model "$VERIFY" --judge-model "$VERIFY" >/dev/null 2>&1
done
echo "=== DONE $(date +%H:%M:%S) ==="
