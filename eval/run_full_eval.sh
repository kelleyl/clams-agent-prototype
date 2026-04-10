#!/bin/bash
# Full evaluation pipeline for thesis
#
# Usage:
#   cd ~/clams-agent-prototype
#   bash eval/run_full_eval.sh

cd "$(dirname "$0")/.."

MODEL="${MODEL:-qwen3:8b}"
BENCHMARK="qa-data/benchmark/benchmark.jsonl"
INDEX_DIR="data/video_indexes"
INDEX_DIR_25="data/video_indexes_25_baseline"
RESULTS="eval/results"
TIMING_LOG="$RESULTS/timing.jsonl"
mkdir -p "$RESULTS"

log_gpu() {
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "n/a"
}

run_stage() {
    local name="$1"
    shift
    local start=$(date +%s)
    echo ""
    echo "=== $name ==="
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"

    "$@"

    local end=$(date +%s)
    local elapsed=$((end - start))
    echo "  Duration: ${elapsed}s ($((elapsed/60))m $((elapsed%60))s)"
    echo "  GPU: $(log_gpu)"
    echo "{\"stage\": \"$name\", \"seconds\": $elapsed, \"timestamp\": \"$(date -Iseconds)\"}" >> "$TIMING_LOG"
}

EVAL_START=$(date +%s)
echo "============================================================"
echo "CLAMS-Agent Full Evaluation"
echo "Model: $MODEL"
echo "Benchmark: $(wc -l < $BENCHMARK) questions"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "GPU: $(log_gpu)"
echo "============================================================"

# --- Stage 1: No-tools baseline ---
run_stage "No-tools baseline" bash -c "
if [ ! -f $RESULTS/no_tools_baseline.jsonl ]; then
    python3 eval/run_ablation.py --benchmark $BENCHMARK --ablation no_index --output $RESULTS/no_tools_baseline.jsonl --model $MODEL 2>&1 | tail -5
else
    echo '  Already exists, skipping'
fi"

# --- Stage 2a: Index QA (3.5 captions) ---
run_stage "Index QA (3.5)" bash -c "
if [ ! -f $RESULTS/index_qa_35.jsonl ]; then
    python3 eval/run_index_qa.py --benchmark $BENCHMARK --index-dir $INDEX_DIR --output $RESULTS/index_qa_35.jsonl --model $MODEL 2>&1 | tail -5
else
    echo '  Already exists, skipping'
fi"

# --- Stage 2b: Index QA (2.5 baseline) ---
run_stage "Index QA (2.5)" bash -c "
if [ ! -f $RESULTS/index_qa_25.jsonl ]; then
    python3 eval/run_index_qa.py --benchmark $BENCHMARK --index-dir $INDEX_DIR_25 --output $RESULTS/index_qa_25.jsonl --model $MODEL 2>&1 | tail -5
else
    echo '  Already exists, skipping'
fi"

# --- Stage 3: Ablations ---
for abl in no_asr no_visual no_ocr no_entities no_relations asr_only visual_only; do
    run_stage "Ablation: $abl" bash -c "
    if [ ! -f $RESULTS/ablation_${abl}.jsonl ]; then
        python3 eval/run_ablation.py --benchmark $BENCHMARK --ablation $abl --output $RESULTS/ablation_${abl}.jsonl --model $MODEL 2>&1 | tail -3
    else
        echo '  Already exists, skipping'
    fi"
done

# --- Stage 4: Score all ---
run_stage "Scoring" bash -c "
for f in $RESULTS/*.jsonl; do
    name=\$(basename \$f .jsonl)
    [ \"\$name\" = \"timing\" ] && continue
    score_file=$RESULTS/\${name}_scores.json
    if [ ! -f \"\$score_file\" ]; then
        echo \"  Scoring: \$name\"
        python3 eval/score_predictions.py --predictions \$f --benchmark $BENCHMARK --mode mc-exact --output \$score_file 2>&1 | tail -3
    fi
done"

# --- Summary ---
EVAL_END=$(date +%s)
EVAL_TOTAL=$((EVAL_END - EVAL_START))

echo ""
echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
python3 -c "
import json, os
for f in sorted(os.listdir('$RESULTS')):
    if not f.endswith('_scores.json'): continue
    name = f.replace('_scores.json', '')
    with open(os.path.join('$RESULTS', f)) as fh:
        scores = json.load(fh)
    mc_acc = scores.get('mc_accuracy', scores.get('accuracy', 0))
    total = scores.get('total', 0)
    print('  %-30s  MC acc: %.1f%%  (%d questions)' % (name, mc_acc * 100, total))
"

echo ""
echo "=== Timing ==="
cat "$TIMING_LOG" 2>/dev/null | python3 -c "
import json, sys
total = 0
for line in sys.stdin:
    d = json.loads(line)
    s = d['seconds']
    total += s
    print('  %-30s %5ds (%dm %ds)' % (d['stage'], s, s//60, s%60))
print('  %-30s %5ds (%dm %ds)' % ('TOTAL', total, total//60, total%60))
"

echo ""
echo "Total eval time: ${EVAL_TOTAL}s ($((EVAL_TOTAL/60))m $((EVAL_TOTAL%60))s)"
echo "Done. Results in $RESULTS/"
