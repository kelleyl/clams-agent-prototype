#!/bin/bash
# Generate side-by-side v1 vs v2 timeline plots on aristotle.

set -euo pipefail

APP_DIR="/home/kmlynch/clams_apps/app-face-tracker"
VENV="$APP_DIR/.venv"
VIDEO_DIR="/home/kmlynch/chicago_tv/FuzzyMemoriesTV"
V1_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/output"
V2_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/output_v2"
PLOT_DIR="/home/kmlynch/chicago_tv/face_tracker_batch/comparison_plots"

source "$VENV/bin/activate"
mkdir -p "$PLOT_DIR"

# Videos to compare (most interesting changes)
VIDEOS=(
    "ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977"
    "ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985"
    "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980"
    "ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982"
)

cd "$APP_DIR"

for VIDEO_NAME in "${VIDEOS[@]}"; do
    SHORT=$(echo "$VIDEO_NAME" | cut -c1-40)
    VIDEO_FILE="$VIDEO_DIR/${VIDEO_NAME}.mp4"
    V1_MMIF="$V1_DIR/${VIDEO_NAME}_faces.mmif"
    V2_MMIF="$V2_DIR/${VIDEO_NAME}_faces_v2.mmif"

    if [ ! -f "$VIDEO_FILE" ]; then
        echo "SKIP (no video): $SHORT"
        continue
    fi

    echo "Generating: $SHORT"

    # v1 plot
    if [ -f "$V1_MMIF" ]; then
        python visualize_faces_timeline.py "$V1_MMIF" "$VIDEO_FILE" \
            --top 15 --output "$PLOT_DIR/${VIDEO_NAME}_v1.png"
        echo "  v1 done"
    fi

    # v2 plot
    if [ -f "$V2_MMIF" ]; then
        python visualize_faces_timeline.py "$V2_MMIF" "$VIDEO_FILE" \
            --top 15 --output "$PLOT_DIR/${VIDEO_NAME}_v2.png"
        echo "  v2 done"
    fi
done

echo "All plots saved to: $PLOT_DIR"
