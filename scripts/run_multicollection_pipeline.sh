#!/bin/bash
set -e

# Multi-collection pipeline: Peabody + NJN + IMLS videos
# Runs full pipeline: SWT → Parakeet → TransNet → Qwen SWT → Qwen shots → merge → diarization
# Then rebuild_indexes.py handles NER + relations + grounding + indexing

WORK_DIR=$HOME/multicollection_batch
QWEN_DIR=$HOME/clams_apps/app-qwen3vl-captioner
VIDEO_LIST=$HOME/clams-agent-prototype/scripts/new_video_list.txt

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

mkdir -p $WORK_DIR/{step0_initial,step1_swt,step2_parakeet,step3_transnet_raw,qwen_swt_out,qwen_shots_out,final,diarization}

# Read video list: collection\tpath\tduration
declare -a VIDEO_PATHS
declare -a VIDEO_IDS
declare -a VIDEO_DIRS

while IFS=$'\t' read -r collection path duration; do
  fname=$(basename "$path" .mp4)
  # Strip .h264 suffix if present
  fname="${fname%.h264}"
  VIDEO_IDS+=("$fname")
  VIDEO_PATHS+=("$path")
  VIDEO_DIRS+=("$(dirname "$path")")
done < "$VIDEO_LIST"

total=${#VIDEO_IDS[@]}
echo "=== Multi-Collection Pipeline: $total videos ==="
echo "  $(grep -c 'Peabody' $VIDEO_LIST) Peabody, $(grep -c 'NJN' $VIDEO_LIST) NJN, $(grep -c 'imls' $VIDEO_LIST) IMLS"

# Step 0: Initial MMIFs
echo ""; echo "=== Step 0: Creating initial MMIFs ==="
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  vpath="${VIDEO_PATHS[$idx]}"
  MMIF="$WORK_DIR/step0_initial/${g}.mmif"
  [ -f "$MMIF" ] && continue
  cat > "$MMIF" << EOF
{"metadata":{"mmif":"http://mmif.clams.ai/1.0"},"documents":[{"@type":"http://mmif.clams.ai/vocabulary/VideoDocument/v1","properties":{"mime":"video/mp4","id":"d1","location":"file://${vpath}"}}],"views":[]}
EOF
done
echo "  Created $total initial MMIFs"

# Step 1: SWT
echo ""; echo "=== Step 1: SWT ==="
# Collect all unique video directories for volume mounts
declare -A UNIQUE_DIRS
for d in "${VIDEO_DIRS[@]}"; do UNIQUE_DIRS["$d"]=1; done
VOL_MOUNTS=""
for d in "${!UNIQUE_DIRS[@]}"; do VOL_MOUNTS="$VOL_MOUNTS -v $d:$d"; done

podman run --rm -d --name swt_mc --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
  $VOL_MOUNTS -p 5571:5000 \
  ghcr.io/clamsproject/app-swt-detection:v8.3 python3 /app/app.py --production
for i in $(seq 1 60); do curl -s http://127.0.0.1:5571/ > /dev/null 2>&1 && break; sleep 2; done
echo "  SWT ready"

for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  OUT="$WORK_DIR/step1_swt/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  echo "[$((idx+1))/$total] SWT: $g"; t=$(date +%s)
  curl -s -X POST http://127.0.0.1:5571/ -H "Content-Type: application/json" -d @"$WORK_DIR/step0_initial/${g}.mmif" -o "$OUT"
  echo "[$((idx+1))/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done
podman stop swt_mc 2>/dev/null
echo "  SWT complete"

# Step 2: Parakeet
echo ""; echo "=== Step 2: Parakeet ==="
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  vdir="${VIDEO_DIRS[$idx]}"
  OUT="$WORK_DIR/step2_parakeet/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  echo "[$((idx+1))/$total] Parakeet: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$vdir:$vdir" -v "$WORK_DIR:$WORK_DIR" \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 cli.py --modelSize 0.6b "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$((idx+1))/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 3: TransNet
echo ""; echo "=== Step 3: TransNet ==="
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  vdir="${VIDEO_DIRS[$idx]}"
  TN_OUT="$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif"
  [ -f "$TN_OUT" ] && [ -s "$TN_OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  # Create minimal MMIF (just video doc) for TransNet
  python3 $HOME/newshour_pipeline/create_minimal.py "$WORK_DIR/step2_parakeet/${g}.mmif" "$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif"
  echo "[$((idx+1))/$total] TransNet: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 $GPU_FLAGS $CACHE_FLAGS \
    -v "$vdir:$vdir" -v "$WORK_DIR:$WORK_DIR" \
    localhost/app-transnet-wrapper:local python3 cli.py "$WORK_DIR/step3_transnet_raw/${g}_minimal.mmif" "$TN_OUT"
  echo "[$((idx+1))/$total] Done $(du -h "$TN_OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 4: Qwen SWT OCR
echo ""; echo "=== Step 4: Qwen SWT ==="
export CUDA_VISIBLE_DEVICES=1
cd $QWEN_DIR && source .venv/bin/activate
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  OUT="$WORK_DIR/qwen_swt_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  echo "[$((idx+1))/$total] Qwen(SWT): $g"; t=$(date +%s)
  python3 cli.py --config config/swt_transcription.yaml --batchSize 64 "$WORK_DIR/step1_swt/${g}.mmif" "$OUT"
  echo "[$((idx+1))/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 5: Qwen shots (visual description per TransNet shot boundary)
echo ""; echo "=== Step 5: Qwen Shots ==="
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  OUT="$WORK_DIR/qwen_shots_out/${g}.mmif"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  echo "[$((idx+1))/$total] Qwen(shots): $g"; t=$(date +%s)
  python3 cli.py --config config/transnet_shots.yaml --batchSize 64 "$WORK_DIR/step3_transnet_raw/${g}_transnet.mmif" "$OUT"
  echo "[$((idx+1))/$total] Done $(du -h "$OUT" | cut -f1) in $(($(date +%s)-t))s"
done

# Step 6: Merge
echo ""; echo "=== Step 6: Merge ==="
cd $HOME/clams-agent-prototype
python3 << 'PYEOF'
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/newshour_pipeline"))
from merge_mmif_views import merge_mmifs
work = os.path.expanduser("~/multicollection_batch")
video_list = os.path.expanduser("~/clams-agent-prototype/scripts/new_video_list.txt")

guids = []
with open(video_list) as f:
    for line in f:
        parts = line.strip().split("\t")
        fname = os.path.basename(parts[1]).replace(".mp4", "").replace(".h264", "")
        guids.append(fname)

for i, g in enumerate(guids, 1):
    out = f"{work}/final/{g}.mmif"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"  [{i}/{len(guids)}] SKIP: {g}")
        continue
    base = f"{work}/step2_parakeet/{g}.mmif"
    additional = [
        f"{work}/step3_transnet_raw/{g}_transnet.mmif",
        f"{work}/qwen_swt_out/{g}.mmif",
        f"{work}/qwen_shots_out/{g}.mmif",
    ]
    # Only merge files that exist
    additional = [a for a in additional if os.path.exists(a)]
    try:
        n = merge_mmifs(base, additional, out, app_filters=["transnet","captioner","captioner"])
        print(f"  [{i}/{len(guids)}] {g}: {n} views merged")
    except Exception as e:
        print(f"  [{i}/{len(guids)}] {g}: MERGE FAILED: {e}")
PYEOF

# Step 7: Diarization
echo ""; echo "=== Step 7: Diarization ==="
for idx in $(seq 0 $((total-1))); do
  g="${VIDEO_IDS[$idx]}"
  vpath="${VIDEO_PATHS[$idx]}"
  OUT="$WORK_DIR/diarization/${g}.json"
  [ -f "$OUT" ] && [ -s "$OUT" ] && echo "[$((idx+1))/$total] SKIP: $g" && continue
  echo "[$((idx+1))/$total] Diarizing: $g"; t=$(date +%s)
  podman run --rm --pids-limit 16384 --ipc=host $GPU_FLAGS $CACHE_FLAGS \
    -v "$(dirname "$vpath"):$(dirname "$vpath")" -v "$WORK_DIR:$WORK_DIR" \
    -v "$HOME/newshour_pipeline/run_diarization.py:/scripts/run_diarization.py" \
    -v /localcache/shared/torch_home:/root/.cache/torch \
    -v /localcache/shared/hf_cache/hub:/root/.cache/huggingface/hub \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 /scripts/run_diarization.py "$vpath" "$OUT"
  if [ -f "$OUT" ]; then
    speakers=$(python3 -c "import json; print(json.load(open('$OUT')).get('num_speakers',0))")
    echo "[$((idx+1))/$total] Done in $(($(date +%s)-t))s: $speakers speakers"
  else echo "[$((idx+1))/$total] FAILED"; fi
done

# Step 8: Copy finals to combined_outputs and rebuild indexes
echo ""; echo "=== Step 8: Copy to combined_outputs ==="
cp $WORK_DIR/final/*.mmif $HOME/clams-agent-prototype/data/combined_outputs/ 2>/dev/null || true
echo "  Copied $(ls $WORK_DIR/final/*.mmif 2>/dev/null | wc -l) MMIFs"

echo ""; echo "=== Step 9: Rebuild indexes (NER + relations + grounding) ==="
cd $HOME/clams-agent-prototype
python3 scripts/rebuild_indexes.py --mmif-dir $WORK_DIR/final

echo ""; echo "=== MULTI-COLLECTION PIPELINE COMPLETE ==="
echo "Total indexed videos: $(ls data/video_indexes/*.json 2>/dev/null | wc -l)"
