#!/bin/bash
set -e

# Batch 3: 25 more NewsHour videos — full pipeline
WORK_DIR=$HOME/newshour_batch3
VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour
QWEN_DIR=$HOME/clams_apps/app-qwen3vl-captioner

GUIDS=(
  cpb-aacip-507-5q4rj49c6q cpb-aacip-507-z892805z01 cpb-aacip-507-4b2x34n67h
  cpb-aacip-507-7659c6sk9k cpb-aacip-507-ns0ks6jx8k cpb-aacip-507-3b5w669q04
  cpb-aacip-507-h707w67w4s cpb-aacip-507-dj58c9rs7q cpb-aacip-507-fj29883b24
  cpb-aacip-507-f47gq6rq94 cpb-aacip-507-6h4cn6zj6q cpb-aacip-507-ws8hd7pn8f
  cpb-aacip-507-4t6f18sz37 cpb-aacip-507-4t6f18t00t cpb-aacip-507-3f4kk94w29
  cpb-aacip-507-kk94747k0n cpb-aacip-507-4f1mg7gc44 cpb-aacip-507-5x2599zp0m
  cpb-aacip-507-251fj2b03t cpb-aacip-507-jw86h4dh3c cpb-aacip-507-4t6f18t15n
  cpb-aacip-507-bn9x05xz7r cpb-aacip-507-jm23b5x19r cpb-aacip-507-3x83j39n00
  cpb-aacip-507-dv1cj8881m
)
total=${#GUIDS[@]}

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

mkdir -p $WORK_DIR/{step0_initial,step1_swt,step2_parakeet,step3_transnet_raw,qwen_swt_out,qwen_shots_out,final,diarization}

# Step 0: Initial MMIFs
echo "=== Creating initial MMIFs ==="
for g in "${GUIDS[@]}"; do
  MMIF="$WORK_DIR/step0_initial/${g}.mmif"
  [ -f "$MMIF" ] && continue
  cat > "$MMIF" << EOF
{"metadata":{"mmif":"http://mmif.clams.ai/1.0"},"documents":[{"@type":"http://mmif.clams.ai/vocabulary/VideoDocument/v1","properties":{"mime":"video/mp4","id":"d1","location":"file://${VIDEO_DIR}/${g}.mp4"}}],"views":[]}
EOF
done

# Step 1: SWT
echo "=== Step 1: SWT ==="
podman run --rm -d --name swt_b3 --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
  -v "$VIDEO_DIR:$VIDEO_DIR" -p 5561:5000 \
  ghcr.io/clamsproject/app-swt-detection:v8.3 python3 /app/app.py --production
for i in $(seq 1 60); do curl -s http://127.0.0.1:5561/ > /dev/null 2>&1 && break; sleep 2; done
echo "  SWT ready"
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/step1_swt/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] SWT: $g"; t=$(date +%s)
  curl -s -X POST http://127.0.0.1:5561/ -H "Content-Type: application/json" -d @"$WORK_DIR/step0_initial/${g}.mmif" -o "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done
podman stop swt_b3 2>/dev/null

# Step 2: Parakeet
echo ""; echo "=== Step 2: Parakeet ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/step2_parakeet/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Parakeet: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 cli.py --modelSize 0.6b "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 3: TransNet (CLI)
echo ""; echo "=== Step 3: TransNet ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  TN_OUT="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  [ -f "$TN_OUT" ] && [ -s "$TN_OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  python3 $HOME/newshour_pipeline/create_minimal.py "$WORK_DIR/step2_parakeet/${g}.mmif" "$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif"
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
  echo "[$i/$total] Qwen(SWT): $g"; t=$(date +%s)
  python3 cli.py --config config/swt_transcription.yaml --batchSize 64 "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 5: Qwen shots
echo ""; echo "=== Step 5: Qwen Shots ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/qwen_shots_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Qwen(shots): $g"; t=$(date +%s)
  python3 cli.py --config config/transnet_shots.yaml --batchSize 64 "$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif" "$OUT"
  echo "[$i/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 6: Merge
echo ""; echo "=== Step 6: Merge ==="
python3 << 'PYEOF'
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/newshour_pipeline"))
from merge_mmif_views import merge_mmifs
work = os.path.expanduser("~/newshour_batch3")
guids = [
  "cpb-aacip-507-5q4rj49c6q","cpb-aacip-507-z892805z01","cpb-aacip-507-4b2x34n67h",
  "cpb-aacip-507-7659c6sk9k","cpb-aacip-507-ns0ks6jx8k","cpb-aacip-507-3b5w669q04",
  "cpb-aacip-507-h707w67w4s","cpb-aacip-507-dj58c9rs7q","cpb-aacip-507-fj29883b24",
  "cpb-aacip-507-f47gq6rq94","cpb-aacip-507-6h4cn6zj6q","cpb-aacip-507-ws8hd7pn8f",
  "cpb-aacip-507-4t6f18sz37","cpb-aacip-507-4t6f18t00t","cpb-aacip-507-3f4kk94w29",
  "cpb-aacip-507-kk94747k0n","cpb-aacip-507-4f1mg7gc44","cpb-aacip-507-5x2599zp0m",
  "cpb-aacip-507-251fj2b03t","cpb-aacip-507-jw86h4dh3c","cpb-aacip-507-4t6f18t15n",
  "cpb-aacip-507-bn9x05xz7r","cpb-aacip-507-jm23b5x19r","cpb-aacip-507-3x83j39n00",
  "cpb-aacip-507-dv1cj8881m",
]
for i, g in enumerate(guids, 1):
    base = f"{work}/step2_parakeet/{g}.mmif"
    additional = [f"{work}/step3_transnet_raw/{g}_transnet.mmif", f"{work}/qwen_swt_out/{g}.mmif", f"{work}/qwen_shots_out/{g}.mmif"]
    n = merge_mmifs(base, additional, f"{work}/final/{g}.mmif", app_filters=["transnet","captioner","captioner"])
    print(f"  [{i}/{len(guids)}] {g}: {n} views merged")
PYEOF

# Step 7: Diarization
echo ""; echo "=== Step 7: Diarization ==="
i=0; for g in "${GUIDS[@]}"; do i=$((i+1))
  OUT="$WORK_DIR/diarization/${g}.json"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$i/$total] SKIP: $g" && continue
  echo "[$i/$total] Diarizing: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 --ipc=host $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" -v "$WORK_DIR:$WORK_DIR" \
    -v "$HOME/newshour_pipeline/run_diarization.py:/scripts/run_diarization.py" \
    -v /localcache/shared/torch_home:/root/.cache/torch \
    -v /localcache/shared/hf_cache/hub:/root/.cache/huggingface/hub \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 /scripts/run_diarization.py "$VIDEO_DIR/${g}.mp4" "$OUT"
  if [ -f "$OUT" ]; then
    speakers=$(python3 -c "import json; print(json.load(open('$OUT')).get('num_speakers',0))")
    echo "[$i/$total] Done in $(($(date +%s)-t))s: $speakers speakers"
  else echo "[$i/$total] FAILED"; fi
done

echo ""; echo "=== BATCH 3 PIPELINE COMPLETE ==="
ls -lh $WORK_DIR/final/ | wc -l
