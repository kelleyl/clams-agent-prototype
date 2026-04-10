#!/bin/bash
set -e

# Batch 2: 10 more NewsHour videos — full pipeline
# SWT → Parakeet → TransNet (CLI) → Qwen SWT → Qwen shots → merge → diarization

WORK_DIR=$HOME/newshour_batch2
VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour
QWEN_DIR=$HOME/clams_apps/app-qwen3vl-captioner

GUIDS=(
  cpb-aacip-507-pr7mp4wf25
  cpb-aacip-507-r785h7cp0z
  cpb-aacip-507-v11vd6pz5w
  cpb-aacip-507-bz6154fc44
  cpb-aacip-507-m61bk17f5g
  cpb-aacip-507-vm42r3pt6h
  cpb-aacip-507-zk55d8pd1h
  cpb-aacip-507-zw18k75z4h
  cpb-aacip-507-4746q1t25k
  cpb-aacip-507-6h4cn6zk04
)
total=${#GUIDS[@]}

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

mkdir -p $WORK_DIR/{step0_initial,step1_swt,step2_parakeet,step3_transnet_raw,qwen_swt_out,qwen_shots_out,final,diarization}

# ============================================================
# Step 0: Create initial MMIFs
# ============================================================
echo "=== Creating initial MMIFs ==="
for g in "${GUIDS[@]}"; do
  MMIF="$WORK_DIR/step0_initial/${g}.mmif"
  if [ -f "$MMIF" ]; then continue; fi
  cat > "$MMIF" << EOF
{"metadata":{"mmif":"http://mmif.clams.ai/1.0"},"documents":[{"@type":"http://mmif.clams.ai/vocabulary/VideoDocument/v1","properties":{"mime":"video/mp4","id":"d1","location":"file://${VIDEO_DIR}/${g}.mp4"}}],"views":[]}
EOF
done

# ============================================================
# Step 1: SWT
# ============================================================
echo "=== Step 1: SWT Detection ==="
podman run --rm -d --name swt_b2 --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
  -v "$VIDEO_DIR:$VIDEO_DIR" -p 5561:5000 \
  ghcr.io/clamsproject/app-swt-detection:v8.3 python3 /app/app.py --production
for i in $(seq 1 60); do curl -s http://127.0.0.1:5561/ > /dev/null 2>&1 && break; sleep 2; done
echo "  SWT ready"

i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  OUT="$WORK_DIR/step1_swt/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] SWT: $g"
  t=$(date +%s)
  curl -s -X POST http://127.0.0.1:5561/ -H "Content-Type: application/json" -d @"$WORK_DIR/step0_initial/${g}.mmif" -o "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done
podman stop swt_b2 2>/dev/null

# ============================================================
# Step 2: Parakeet ASR
# ============================================================
echo ""
echo "=== Step 2: Parakeet ASR ==="
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  OUT="$WORK_DIR/step2_parakeet/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Parakeet: $g"
  t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 cli.py --modelSize 0.6b "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# ============================================================
# Step 3: TransNet (CLI, minimal MMIFs)
# ============================================================
echo ""
echo "=== Step 3: TransNet ==="
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  MINIMAL="$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif"
  TN_OUT="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  [ -f "$TN_OUT" ] && [ -s "$TN_OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  python3 $HOME/newshour_pipeline/create_minimal.py "$WORK_DIR/step2_parakeet/${g}.mmif" "$MINIMAL"
  echo "[$i/$total] TransNet: $g"
  t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    localhost/app-transnet-wrapper:local python3 cli.py "$MINIMAL" "$TN_OUT"
  echo "[$i/$total] Done $(du -h "$TN_OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# ============================================================
# Step 4: Qwen SWT OCR (input: SWT-only MMIF)
# ============================================================
echo ""
echo "=== Step 4: Qwen SWT ==="
export CUDA_VISIBLE_DEVICES=1
cd $QWEN_DIR && source .venv/bin/activate

i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  OUT="$WORK_DIR/qwen_swt_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Qwen(SWT): $g"
  t=$(date +%s)
  python3 cli.py --config config/swt_transcription.yaml --batchSize 64 "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# ============================================================
# Step 5: Qwen TransNet shots (input: TransNet-only MMIF)
# ============================================================
echo ""
echo "=== Step 5: Qwen Shots ==="
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  OUT="$WORK_DIR/qwen_shots_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Qwen(shots): $g"
  t=$(date +%s)
  python3 cli.py --config config/transnet_shots.yaml --batchSize 64 "$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# ============================================================
# Step 6: Merge all views
# ============================================================
echo ""
echo "=== Step 6: Merge ==="
python3 << 'PYEOF'
import json, re, os, sys
sys.path.insert(0, os.path.expanduser("~/newshour_pipeline"))
from merge_mmif_views import merge_mmifs

work = os.path.expanduser("~/newshour_batch2")
guids = [
    "cpb-aacip-507-pr7mp4wf25","cpb-aacip-507-r785h7cp0z","cpb-aacip-507-v11vd6pz5w",
    "cpb-aacip-507-bz6154fc44","cpb-aacip-507-m61bk17f5g","cpb-aacip-507-vm42r3pt6h",
    "cpb-aacip-507-zk55d8pd1h","cpb-aacip-507-zw18k75z4h","cpb-aacip-507-4746q1t25k",
    "cpb-aacip-507-6h4cn6zk04",
]
for i, g in enumerate(guids, 1):
    base = f"{work}/step2_parakeet/{g}.mmif"
    additional = [
        f"{work}/step3_transnet_raw/{g}_transnet.mmif",
        f"{work}/qwen_swt_out/{g}.mmif",
        f"{work}/qwen_shots_out/{g}.mmif",
    ]
    app_filters = ["transnet", "captioner", "captioner"]
    output = f"{work}/final/{g}.mmif"
    n = merge_mmifs(base, additional, output, app_filters=app_filters)
    print(f"  [{i}/{len(guids)}] {g}: {n} views merged")
PYEOF

# ============================================================
# Step 7: Diarization
# ============================================================
echo ""
echo "=== Step 7: Diarization ==="
i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  OUT="$WORK_DIR/diarization/${g}.json"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Diarizing: $g"
  t=$(date +%s)
  podman run --rm --pids-limit 16384 --ipc=host $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    -v "$HOME/newshour_pipeline/run_diarization.py:/scripts/run_diarization.py" \
    -v /localcache/shared/torch_home:/root/.cache/torch \
    -v /localcache/shared/hf_cache/hub:/root/.cache/huggingface/hub \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 /scripts/run_diarization.py "$VIDEO_DIR/${g}.mp4" "$OUT"
  t_end=$(date +%s)
  if [ -f "$OUT" ]; then
    speakers=$(python3 -c "import json; print(json.load(open('$OUT')).get('num_speakers',0))")
    echo "[$i/$total] Done in $((t_end-t))s: $speakers speakers"
  else
    echo "[$i/$total] FAILED"
  fi
done

echo ""
echo "=== BATCH 2 PIPELINE COMPLETE ==="
ls -lh $WORK_DIR/final/
