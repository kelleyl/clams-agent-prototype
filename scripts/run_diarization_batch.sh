#!/bin/bash
set -e

WORK_DIR=$HOME/newshour_pipeline
VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour
SCRIPT_DIR=$WORK_DIR

GUIDS=(
  cpb-aacip-507-154dn40c26 cpb-aacip-507-1v5bc3tf81
  cpb-aacip-507-4t6f18t178 cpb-aacip-507-6w96689725
  cpb-aacip-507-7659c6sk7z cpb-aacip-507-9882j68s35
  cpb-aacip-507-cf9j38m509 cpb-aacip-507-n29p26qt59
  cpb-aacip-507-nk3610wp6s cpb-aacip-507-pc2t43js98
)
total=${#GUIDS[@]}

GPU_FLAGS="--device nvidia.com/gpu=all --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1"
CACHE_FLAGS="-v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub"

echo "=== Speaker Diarization (NeMo ClusteringDiarizer) ==="
mkdir -p $WORK_DIR/diarization

i=0
for g in "${GUIDS[@]}"; do
  i=$((i+1))
  VIDEO="$VIDEO_DIR/${g}.mp4"
  OUTPUT="$WORK_DIR/diarization/${g}.json"

  if [ -f "$OUTPUT" ] && [ -s "$OUTPUT" ]; then
    echo "[$i/$total] SKIP (exists): $g"
    continue
  fi

  echo "[$i/$total] Diarizing: $g"
  t_start=$(date +%s)
  podman run --rm --pids-limit 16384 --ipc=host $GPU_FLAGS $CACHE_FLAGS \
    -v "$VIDEO_DIR:$VIDEO_DIR" \
    -v "$WORK_DIR:$WORK_DIR" \
    -v "$SCRIPT_DIR/run_diarization.py:/scripts/run_diarization.py" \
    -v /localcache/shared/torch_home:/root/.cache/torch \
    -v /localcache/shared/hf_cache/hub:/root/.cache/huggingface/hub \
    ghcr.io/clamsproject/app-parakeet-wrapper:v1.0 \
    python3 /scripts/run_diarization.py "$VIDEO" "$OUTPUT"
  t_end=$(date +%s)

  if [ -f "$OUTPUT" ]; then
    speakers=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(d.get('num_speakers',0))")
    segs=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(len(d.get('segments',[])))")
    echo "[$i/$total] Done in $((t_end - t_start))s: $speakers speakers, $segs segments"
  else
    echo "[$i/$total] FAILED in $((t_end - t_start))s"
  fi
done

echo "=== Diarization Complete ==="
