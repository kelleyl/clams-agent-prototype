#!/bin/bash
set -e

WORK_DIR=$HOME/newshour_pipeline
VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour

GUIDS=(
  cpb-aacip-507-154dn40c26 cpb-aacip-507-1v5bc3tf81
  cpb-aacip-507-4t6f18t178 cpb-aacip-507-6w96689725
  cpb-aacip-507-7659c6sk7z cpb-aacip-507-9882j68s35
  cpb-aacip-507-cf9j38m509 cpb-aacip-507-n29p26qt59
  cpb-aacip-507-nk3610wp6s cpb-aacip-507-pc2t43js98
)
total=${#GUIDS[@]}

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

# ============================================================
# Step 1: Run Parakeet ASR on SWT MMIFs (replacing Whisper)
# ============================================================
echo "=== Parakeet ASR ==="
mkdir -p $WORK_DIR/step2_parakeet

i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  INPUT="$WORK_DIR/step1_swt/${g}.mmif"
  OUTPUT="$WORK_DIR/step2_parakeet/${g}.mmif"

  if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    echo "[$i/$total] SKIP (exists): $g"
    continue
  fi

  echo "[$i/$total] Parakeet: $g"
  t_start=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" \
    -v "$WORK_DIR:$WORK_DIR" \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 cli.py --modelSize 0.6b \
    "$INPUT" "$OUTPUT"
  t_end=$(date +%s)
  sz=$(du -h "$OUTPUT" | cut -f1)
  echo "[$i/$total] Done ${sz} in $((t_end - t_start))s"
done

# ============================================================
# Step 2: Re-merge with parakeet instead of whisper
# Uses existing TransNet + Qwen SWT + Qwen shots outputs
# ============================================================
echo ""
echo "=== Re-merging with Parakeet ASR ==="
mkdir -p $WORK_DIR/final_parakeet

python3 $WORK_DIR/merge_mmif_views.py --batch-parakeet $WORK_DIR

echo ""
echo "=== DONE ==="
ls -lh $WORK_DIR/final_parakeet/
