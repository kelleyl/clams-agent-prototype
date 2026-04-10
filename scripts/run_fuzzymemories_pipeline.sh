#!/bin/bash
set -e

# Full pipeline re-run for FuzzyMemories videos with watermark masked out
# Step 0: Mask watermark → Step 1-7: Same pipeline as NewsHour

WORK_DIR=$HOME/fuzzymemories_rerun
SRC_DIR=/home/kmlynch/chicago_tv/FuzzyMemoriesTV
QWEN_DIR=$HOME/clams_apps/app-qwen3vl-captioner

GUIDS=(
  ABC_Network_-_Barbary_Coast_-_Funny_Money_-_WCVB_Channel_5_Complete_Broadcast_9_8_1975
  ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977
  ABC_Network_-_General_Hospital_Last_14_Minutes_11_20_1979
  ABC_Network_-_Let_s_Make_a_Deal_-_WLS_Channel_7_Pre-Show_Break_Opening_Moments_2_5_1976
  ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982
  ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985
  ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_2_18_1979
  ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980
  A_CBS_Family_Presentation_Bumper_1973
  Alexander_s_Star_by_Ideal_-_Don_t_Pick_It_Up_Commercial_1982
)
total=${#GUIDS[@]}

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

mkdir -p $WORK_DIR/{masked_videos,step0_initial,step1_swt,step2_parakeet,step3_transnet_raw,qwen_swt_out,qwen_shots_out,final,diarization}

# ============================================================
# Step 0: Mask watermark (black rectangle over bottom-left)
# Watermark at ~x:0-430, y:1020-1080 on 1440x1080 frames
# ============================================================
echo "=== Step 0: Masking watermark ==="
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  SRC="$SRC_DIR/${g}.mp4"
  DST="$WORK_DIR/masked_videos/${g}.mp4"
  if [ -f "$DST" ]; then
    echo "[$i/$total] SKIP (exists): $g"
    continue
  fi
  if [ ! -f "$SRC" ]; then
    echo "[$i/$total] MISSING: $g"
    continue
  fi
  echo "[$i/$total] Masking: $g"
  t=$(date +%s)
  ffmpeg -i "$SRC" \
    -vf "drawbox=x=0:y=ih-65:w=430:h=65:color=black:t=fill" \
    -c:a copy -c:v libx264 -preset fast -crf 18 \
    "$DST" -y 2>/dev/null
  echo "[$i/$total] Done in $(($(date +%s)-t))s"
done

VIDEO_DIR="$WORK_DIR/masked_videos"

# Step 0b: Create initial MMIFs
echo "=== Creating initial MMIFs ==="
for g in "${GUIDS[@]}"; do
  MMIF="$WORK_DIR/step0_initial/${g}.mmif"
  [ -f "$MMIF" ] && continue
  VIDEO="$VIDEO_DIR/${g}.mp4"
  [ ! -f "$VIDEO" ] && continue
  cat > "$MMIF" << EOF
{"metadata":{"mmif":"http://mmif.clams.ai/1.0"},"documents":[{"@type":"http://mmif.clams.ai/vocabulary/VideoDocument/v1","properties":{"mime":"video/mp4","id":"d1","location":"file://${VIDEO}"}}],"views":[]}
EOF
done

# Step 1: SWT
echo "=== Step 1: SWT ==="
podman run --rm -d --name swt_fm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
  -v "$VIDEO_DIR:$VIDEO_DIR" -p 5561:5000 \
  ghcr.io/clamsproject/app-swt-detection:v8.3 python3 /app/app.py --production
for j in $(seq 1 60); do curl -s http://127.0.0.1:5561/ > /dev/null 2>&1 && break; sleep 2; done
echo "  SWT ready"
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/step1_swt/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  IN="$WORK_DIR/step0_initial/${g}.mmif"
  [ ! -f "$IN" ] && echo "[$i/$total] NO INPUT: $g" && continue
  echo "[$i/$total] SWT: $g"; t=$(date +%s)
  curl -s -X POST http://127.0.0.1:5561/ -H "Content-Type: application/json" -d @"$IN" -o "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done
podman stop swt_fm 2>/dev/null

# Step 2: Parakeet ASR
echo ""; echo "=== Step 2: Parakeet ASR ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/step2_parakeet/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  IN="$WORK_DIR/step1_swt/${g}.mmif"
  [ ! -f "$IN" ] && continue
  echo "[$i/$total] Parakeet: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 cli.py --modelSize 0.6b "$IN" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 3: TransNet (CLI)
echo ""; echo "=== Step 3: TransNet ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  TN_OUT="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  [ -f "$TN_OUT" ] && [ -s "$TN_OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  IN="$WORK_DIR/step2_parakeet/${g}.mmif"
  [ ! -f "$IN" ] && continue
  python3 $HOME/newshour_pipeline/create_minimal.py "$IN" "$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif"
  echo "[$i/$total] TransNet: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    localhost/app-transnet-wrapper:local python3 cli.py "$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif" "$TN_OUT"
  echo "[$i/$total] Done $(du -h "$TN_OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 4: Qwen SWT
echo ""; echo "=== Step 4: Qwen SWT ==="
export CUDA_VISIBLE_DEVICES=1; cd $QWEN_DIR && source .venv/bin/activate
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/qwen_swt_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  IN="$WORK_DIR/step1_swt/${g}.mmif"
  [ ! -f "$IN" ] && continue
  echo "[$i/$total] Qwen(SWT): $g"; t=$(date +%s)
  python3 cli.py --config config/swt_transcription.yaml --batchSize 64 "$IN" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 5: Qwen shots
echo ""; echo "=== Step 5: Qwen Shots ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/qwen_shots_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  IN="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  [ ! -f "$IN" ] && continue
  echo "[$i/$total] Qwen(shots): $g"; t=$(date +%s)
  python3 cli.py --config config/transnet_shots.yaml --batchSize 64 "$IN" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 6: Merge
echo ""; echo "=== Step 6: Merge ==="
python3 << 'PYEOF'
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/newshour_pipeline"))
from merge_mmif_views import merge_mmifs
work = os.path.expanduser("~/fuzzymemories_rerun")
guids = [
  "ABC_Network_-_Barbary_Coast_-_Funny_Money_-_WCVB_Channel_5_Complete_Broadcast_9_8_1975",
  "ABC_Network_-_Future_Cop_-_The_Kansas_City_Kid_-_WCVB-TV_Complete_Broadcast_4_30_1977",
  "ABC_Network_-_General_Hospital_Last_14_Minutes_11_20_1979",
  "ABC_Network_-_Let_s_Make_a_Deal_-_WLS_Channel_7_Pre-Show_Break_Opening_Moments_2_5_1976",
  "ABC_News_Nightline_-_Bitter_Cold_-_WLS_Channel_7_1st_16_Minutes_1_11_1982",
  "ABC_News_Special_Report_-_The_President_at_the_United_Nations_-_WLS-TV_1985",
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_2_18_1979",
  "ABC_News_Weekend_Report_-_WLS_Channel_7_Complete_Broadcast_3_9_1980",
  "A_CBS_Family_Presentation_Bumper_1973",
  "Alexander_s_Star_by_Ideal_-_Don_t_Pick_It_Up_Commercial_1982",
]
for i, g in enumerate(guids, 1):
    base = f"{work}/step2_parakeet/{g}.mmif"
    if not os.path.exists(base): continue
    additional = [f"{work}/step3_transnet_raw/{g}_transnet.mmif", f"{work}/qwen_swt_out/{g}.mmif", f"{work}/qwen_shots_out/{g}.mmif"]
    n = merge_mmifs(base, additional, f"{work}/final/{g}.mmif", app_filters=["transnet","captioner","captioner"])
    print(f"  [{i}/{len(guids)}] {g[:40]}: {n} views merged")
PYEOF

echo ""; echo "=== FUZZYMEMORIES PIPELINE COMPLETE ==="
ls -lh $WORK_DIR/final/ | wc -l
