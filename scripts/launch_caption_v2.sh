#!/bin/bash
source ~/LongVideo-R1/.venv/bin/activate
pkill -f "port 9081" 2>/dev/null
sleep 3

MODEL="/localcache/shared/hf_cache/hub/models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ/snapshots/66c370b74a18e7b1e871c97918f032ed3578dfef"

CUDA_VISIBLE_DEVICES=1,2 nohup vllm serve "$MODEL" \
  --tensor-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 9081 \
  --served-model-name Qwen2.5-VL-32B \
  --quantization awq \
  --dtype float16 \
  --enforce-eager \
  --trust-remote-code \
  > /tmp/vllm_caption.log 2>&1 &

echo "PID: $!"
