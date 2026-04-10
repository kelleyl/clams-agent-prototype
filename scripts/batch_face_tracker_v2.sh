#!/bin/bash
# Batch run face-tracker v2 (quality filtering + tighter clustering) on aristotle.
# Outputs go to a separate directory for side-by-side comparison.

set -euo pipefail

APP_DIR="/home/kmlynch/clams_apps/app-face-tracker"
VENV="$APP_DIR/.venv"
INPUT_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/input"
OUTPUT_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/output_v2"
VIDEO_DIR="/home/kmlynch/chicago_tv/FuzzyMemoriesTV"

# v2 parameters — quality filtering + tighter clustering
SAMPLE_RATE=30
MIN_TRACK_LENGTH=3
CLUSTER_DISTANCE=0.65       # was 0.8
MIN_COHERENCE=0.5
MIN_FACE_QUALITY=0.35       # NEW — filters low-quality detections
USE_GPU=true

mkdir -p "$OUTPUT_DIR"

source "$VENV/bin/activate"

TOTAL=$(ls "$INPUT_DIR"/*.mmif 2>/dev/null | wc -l)
CURRENT=0

echo "============================================"
echo "Face tracker v2 batch: $TOTAL MMIF files"
echo "Cluster dist: $CLUSTER_DISTANCE | Quality: $MIN_FACE_QUALITY"
echo "============================================"

for MMIF_FILE in "$INPUT_DIR"/*.mmif; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$MMIF_FILE" .mmif)
    VIDEO_NAME="${BASENAME%_caption}"
    VIDEO_NAME="${VIDEO_NAME%_ocr}"
    VIDEO_NAME="${VIDEO_NAME%_whisper}"
    OUTPUT_FILE="$OUTPUT_DIR/${VIDEO_NAME}_faces_v2.mmif"

    if [ -s "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (exists): $VIDEO_NAME"
        continue
    fi

    VIDEO_FILE="$VIDEO_DIR/${VIDEO_NAME}.mp4"
    if [ ! -f "$VIDEO_FILE" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (no video): $VIDEO_NAME"
        continue
    fi

    echo "[$CURRENT/$TOTAL] Processing: $VIDEO_NAME"
    START_TIME=$(date +%s)

    cd "$APP_DIR"
    python cli.py \
        --sampleRate "$SAMPLE_RATE" \
        --minTrackLength "$MIN_TRACK_LENGTH" \
        --clusterDistanceThreshold "$CLUSTER_DISTANCE" \
        --minClusterCoherence "$MIN_COHERENCE" \
        --minFaceQuality "$MIN_FACE_QUALITY" \
        --useGpu "$USE_GPU" \
        "$MMIF_FILE" \
        "$OUTPUT_FILE"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[$CURRENT/$TOTAL] Done: ${ELAPSED}s — $VIDEO_NAME"
    echo ""
done

echo "============================================"
echo "v2 batch complete. Output in: $OUTPUT_DIR"
echo "============================================"
