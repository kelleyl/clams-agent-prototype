#!/usr/bin/env bash
set -euo pipefail

# Starts selected CLAMS Docker/Podman app images as HTTP services on Aristotle.
# By default this starts only SWT, because the VLM/ASR services reserve GPU
# memory. Pass service ids as arguments to start more:
#   scripts/run_clams_tool_services_aristotle.sh swt_detection smolvlm2_captioner

ENGINE="${ENGINE:-podman}"
GPU_DEVICE="${GPU_DEVICE:-2}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}"
CLAMS_PRODUCTION="${CLAMS_PRODUCTION:-1}"
GUNICORN_CMD_ARGS="${GUNICORN_CMD_ARGS:---workers ${WEB_CONCURRENCY}}"
PRODUCTION_FLAG=""
if [ "$CLAMS_PRODUCTION" = "1" ]; then
  PRODUCTION_FLAG="--production"
fi
GPU_FLAGS="${GPU_FLAGS:---device nvidia.com/gpu=${GPU_DEVICE} --security-opt=label=disable -e TRITON_LIBCUDA_PATH=/lib64/libcuda.so.1}"
CACHE_FLAGS="${CACHE_FLAGS:--v /localcache/shared/torch_home:/cache/torch -v /localcache/shared/whisper:/cache/whisper -v /localcache/shared/hf_cache/hub:/cache/huggingface/hub}"
VIDEO_MOUNTS="${VIDEO_MOUNTS:--v /llc_data:/llc_data -v /mnt/llc:/mnt/llc -v /home/kmlynch/chicago_tv:/home/kmlynch/chicago_tv}"

if [ "$#" -eq 0 ]; then
  set -- swt_detection
fi

start_service() {
  local id="$1"
  local name image port cmd
  case "$id" in
    swt_detection)
      name="clams_swt_detection"
      image="ghcr.io/clamsproject/app-swt-detection:v8.3"
      port="5561"
      cmd="python3 /app/app.py ${PRODUCTION_FLAG}"
      ;;
    asr_whisper)
      name="clams_whisper_wrapper"
      image="ghcr.io/clamsproject/app-whisper-wrapper:v15"
      port="5562"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    smolvlm2_captioner|scene_summary)
      name="clams_smolvlm2_captioner"
      image="ghcr.io/clamsproject/app-smolvlm2-captioner:latest"
      port="5563"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    vlm_ocr_kie|credits_ocr)
      name="clams_qwen_ocr"
      image="localhost/app-qwen-ocr:test"
      port="5564"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    pyscenedetect)
      name="clams_pyscenedetect_wrapper"
      image="ghcr.io/clamsproject/app-pyscenedetect-wrapper:v3"
      port="5565"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    spacy)
      name="clams_spacy_wrapper"
      image="ghcr.io/clamsproject/app-spacy-wrapper:v2.1"
      port="5566"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    transnet)
      name="clams_transnet_wrapper"
      image="localhost/app-transnet-wrapper:local"
      port="5555"
      cmd="python3 app.py ${PRODUCTION_FLAG}"
      ;;
    *)
      echo "Unknown service id: $id" >&2
      return 2
      ;;
  esac

  if "$ENGINE" ps --format '{{.Names}}' | grep -qx "$name"; then
    echo "$id already running as $name on port $port"
    return 0
  fi

  "$ENGINE" rm -f "$name" >/dev/null 2>&1 || true
  echo "Starting $id -> http://127.0.0.1:${port}/ ($image)"
  # shellcheck disable=SC2086
  "$ENGINE" run -d --name "$name" --pids-limit 16384 \
    $GPU_FLAGS $CACHE_FLAGS $VIDEO_MOUNTS \
    -e WEB_CONCURRENCY="$WEB_CONCURRENCY" \
    -e GUNICORN_CMD_ARGS="$GUNICORN_CMD_ARGS" \
    -p "${port}:5000" "$image" $cmd >/dev/null
}

for service_id in "$@"; do
  start_service "$service_id"
done

echo
"$ENGINE" ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
