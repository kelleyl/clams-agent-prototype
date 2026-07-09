#!/bin/bash
# v6 FULL corpus run (aristotle). Resumable: every stage skips completed work,
# so re-running this script after a crash continues where it left off.
#
#   nohup bash scripts/run_v6_full.sh > /tmp/v6_full.log 2>&1 &
#
# Stages (ordered to minimize ollama model swaps):
#   1. need-down generate + densify   (gemma3:27b)     per video
#   2. exploration program + corpus   (gemma + llama)  per video / once
#   3. blind panel                    (7 ollama models, each loads once)
#   4. round-trip gate                (llama3.3)       per video
#   5. run stats
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
URL=http://localhost:11434/v1
GEN=gemma3:27b-it-qat
VERIFY=llama3.3:latest
PANEL7="llama3.1:8b,qwen3:8b,gemma3:12b-it-qat,gemma3:27b-it-qat,qwen3:30b,llama3.1:70b,llama3.3:latest"

# in-scope videos: salience maps minus the explicit exclusion list
VIDS=$($PY - <<'PYEOF'
import json, pathlib
excl = {e["video_id"] for e in json.load(open("data/v6_excluded_videos.json"))["excluded"]}
vids = [p.stem for p in sorted(pathlib.Path("data/salience_maps").glob("*.json"))
        if p.stem not in excl]
print("\n".join(vids))
PYEOF
)
N=$(echo "$VIDS" | wc -l)
echo "=== v6 full run: $N in-scope videos | git $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | $(date) ==="

echo "=== STAGE 1: need-down generate + densify ($GEN) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  $PY scripts/generate_qa_needdown.py --all --only-video "$V" --skip-done \
      --two-hop-cap 0 --vllm-url $URL --model "$GEN" 2>&1 | tail -1
  $PY scripts/densify_questions.py --video "$V" --dp-url $URL --dp-model "$GEN" 2>&1 | tail -1
  echo "[stage1] done: $V $(date +%H:%M:%S)"
done

echo "=== STAGE 1b: visual VQA (text_focus + corroborated scene captions) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  $PY scripts/generate_qa_visual.py --only-video "$V" --skip-done \
      --gen-url $URL --gen-model "$GEN" --verify-url $URL --verify-model "$VERIFY" 2>&1 | tail -1
done

echo "=== STAGE 2: exploration (program + corpus) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  $PY scripts/generate_qa_exploration.py --scope program --only-video "$V" --skip-done \
      --gen-url $URL --gen-model "$GEN" --verify-url $URL --verify-model "$VERIFY" 2>&1 | tail -1
done
$PY scripts/generate_qa_exploration.py --scope corpus --limit-questions 40 \
    --gen-url $URL --gen-model "$GEN" --verify-url $URL --verify-model "$VERIFY" 2>&1 | tail -3

echo "=== STAGE 3: blind panel (7 models) $(date +%H:%M:%S) ==="
$PY scripts/blind_panel.py --models "$PANEL7" --video $VIDS
echo "=== STAGE 3b: blind panel over visual rows $(date +%H:%M:%S) ==="
$PY scripts/blind_panel.py --qa-dir data/qa_visual --models "$PANEL7" --video $VIDS

echo "=== STAGE 4: round-trip gate ($VERIFY) $(date +%H:%M:%S) ==="
for V in $VIDS; do
  $PY scripts/roundtrip_check.py --video "$V" --rt-url $URL --rt-model "$VERIFY" \
      --judge-model "$VERIFY" 2>&1 | tail -2
done

echo "=== STAGE 5: run stats $(date +%H:%M:%S) ==="
$PY scripts/summarize_v6_run.py

echo "=== DONE $(date) ==="
