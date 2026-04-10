#!/bin/bash
# Batch run face-tracker on Chicago TV videos on aristotle.
#
# Prerequisites:
#   1. Set up the face tracker on aristotle:
#      ssh aristotle
#      cd /home/kmlynch/clams_apps
#      git clone /path/to/app-face-tracker  # or scp the directory
#      python3 -m venv app-face-tracker/.venv
#      source app-face-tracker/.venv/bin/activate
#      pip install -r app-face-tracker/requirements.txt
#
#   2. Copy the latest pipeline MMIF files (qwen caption outputs) to input dir:
#      From local:
#        scp data/qwen_outputs/*_caption.mmif aristotle:/home/kmlynch/chicago_tv/face_tracker_batch/input/
#      Or if they're already on aristotle, just point INPUT_DIR to them.
#
#   3. Run this script on aristotle:
#      bash /home/kmlynch/chicago_tv/face_tracker_batch/batch_face_tracker.sh
#
#   4. Copy results back to local:
#      scp aristotle:/home/kmlynch/chicago_tv/face_tracker_batch/output/*.mmif data/face_tracker_outputs/

set -euo pipefail

APP_DIR="/home/kmlynch/clams_apps/app-face-tracker"
VENV="$APP_DIR/.venv"
INPUT_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/input"
OUTPUT_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/output"
VIDEO_DIR="/home/kmlynch/chicago_tv/FuzzyMemoriesTV"

# Face tracker parameters
SAMPLE_RATE=30          # ~1 fps for 30fps video
MIN_TRACK_LENGTH=3
CLUSTER_DISTANCE=0.8
MIN_COHERENCE=0.5
USE_GPU=true            # Will use CUDA on aristotle

mkdir -p "$OUTPUT_DIR"

# Activate venv
source "$VENV/bin/activate"

# Count files
TOTAL=$(ls "$INPUT_DIR"/*.mmif 2>/dev/null | wc -l)
CURRENT=0

echo "============================================"
echo "Face tracker batch: $TOTAL MMIF files"
echo "Sample rate: $SAMPLE_RATE | GPU: $USE_GPU"
echo "============================================"

for MMIF_FILE in "$INPUT_DIR"/*.mmif; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$MMIF_FILE" .mmif)
    # Strip any existing suffix to get clean video name
    VIDEO_NAME="${BASENAME%_caption}"
    VIDEO_NAME="${VIDEO_NAME%_ocr}"
    VIDEO_NAME="${VIDEO_NAME%_whisper}"
    OUTPUT_FILE="$OUTPUT_DIR/${VIDEO_NAME}_faces.mmif"

    # Skip if already processed (non-empty)
    if [ -s "$OUTPUT_FILE" ]; then
        echo "[$CURRENT/$TOTAL] SKIP (exists): $VIDEO_NAME"
        continue
    fi

    # Verify video file exists
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
        --useGpu "$USE_GPU" \
        "$MMIF_FILE" \
        "$OUTPUT_FILE"

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[$CURRENT/$TOTAL] Done: ${ELAPSED}s — $VIDEO_NAME"
    echo ""
done

echo "============================================"
echo "Batch complete. Output in: $OUTPUT_DIR"
echo "============================================"
