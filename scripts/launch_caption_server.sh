#!/bin/bash
source ~/LongVideo-R1/.venv/bin/activate
kill $(pgrep -f "port 9081") 2>/dev/null
sleep 2

CAPTION_MODEL="/localcache/shared/hf_cache/hub/models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ/snapshots/66c370b74a18e7b1e871c97918f032ed3578dfef"

CUDA_VISIBLE_DEVICES=1,2 nohup vllm serve "$CAPTION_MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 9081 \
  --served-model-name Qwen2.5-VL-32B \
  --quantization awq \
  --dtype float16 \
  > /tmp/vllm_caption.log 2>&1 &

echo "Caption server PID: $!"
