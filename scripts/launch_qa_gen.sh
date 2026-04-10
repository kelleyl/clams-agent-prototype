#!/bin/bash
pkill -f generate_llm 2>/dev/null
sleep 1
cd ~/clams-agent-prototype
export OLLAMA_URL=http://localhost:11434
export OLLAMA_MODEL=qwen3:8b
nohup python3 qa-data/generate_llm_comprehension.py > /tmp/qa_gen.log 2>&1 &
echo "Started PID $!"
