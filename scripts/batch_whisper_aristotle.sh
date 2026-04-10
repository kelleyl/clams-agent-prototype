#!/bin/bash
# Batch run whisper-wrapper on Chicago TV videos via podman on aristotle.
#
# Usage (from local machine):
#   1. Copy MMIF files to aristotle:
#      scp data/qwen_outputs/*_caption.mmif aristotle:/home/kmlynch/chicago_tv/whisper_batch/input/
#   2. SSH to aristotle and run:
#      bash /home/kmlynch/chicago_tv/whisper_batch/batch_whisper_aristotle.sh
#   3. Copy results back:
#      scp aristotle:/home/kmlynch/chicago_tv/whisper_batch/output/*.mmif data/whisper_outputs/

set -euo pipefail

INPUT_DIR="/home/kmlynch/chicago_tv/whisper_batch/input"
OUTPUT_DIR="/home/kmlynch/chicago_tv/whisper_batch/output"
VIDEO_DIR="/home/kmlynch/chicago_tv/FuzzyMemoriesTV"
IMAGE="ghcr.io/clamsproject/app-whisper-wrapper:v15"
MODEL="turbo"

# Shared cache dirs (matches clamspod conventions on aristotle)
if [ -d /localcache ]; then
    TORCH_CACHE_DIR=/localcache/shared/torch_home
    WHISPER_CACHE_DIR=/localcache/shared/whisper
    HF_CACHE_DIR=/localcache/shared/hf_cache/hub
    mkdir -p "$TORCH_CACHE_DIR" "$WHISPER_CACHE_DIR" "$HF_CACHE_DIR"
else
    TORCH_CACHE_DIR=$HOME/.cache/torch
    WHISPER_CACHE_DIR=$HOME/.cache/whisper
    HF_CACHE_DIR=$HOME/.cache/huggingface
fi

mkdir -p "$OUTPUT_DIR"

# Clean up empty output files from failed previous run
find "$OUTPUT_DIR" -name "*.mmif" -empty -delete 2>/dev/null

# Count files
TOTAL=$(ls "$INPUT_DIR"/*.mmif 2>/dev/null | wc -l)
CURRENT=0

echo "============================================"
echo "Whisper batch: $TOTAL MMIF files"
echo "Model: $MODEL | GPU: all"
echo "============================================"

for MMIF_FILE in "$INPUT_DIR"/*.mmif; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$MMIF_FILE" .mmif)
    OUTPUT_FILE="$OUTPUT_DIR/${BASENAME}_whisper.mmif"

    # Skip if already processed (non-empty)
    if [ -s "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (exists): $BASENAME"
        continue
    fi

    echo "[$CURRENT/$TOTAL] Processing: $BASENAME"
    START_TIME=$(date +%s)

    podman run --rm \
        --pids-limit 16384 \
        --device nvidia.com/gpu=all \
        --security-opt=label=disable \
        -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1 \
        -v "$VIDEO_DIR:$VIDEO_DIR:ro" \
        -v "$INPUT_DIR:/input:ro" \
        -v "$OUTPUT_DIR:/output" \
        -v "$TORCH_CACHE_DIR:/cache/torch" \
        -v "$WHISPER_CACHE_DIR:/cache/whisper" \
        -v "$HF_CACHE_DIR:/cache/huggingface/hub" \
        "$IMAGE" \
        python3 cli.py \
            --model "$MODEL" \
            --language en \
            "/input/$(basename "$MMIF_FILE")" \
            "/output/${BASENAME}_whisper.mmif"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[$CURRENT/$TOTAL] Done: ${ELAPSED}s — $BASENAME"
    echo ""
done

echo "============================================"
echo "Batch complete. Output in: $OUTPUT_DIR"
echo "============================================"
