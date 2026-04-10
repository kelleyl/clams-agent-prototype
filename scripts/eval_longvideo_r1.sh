#!/bin/bash
source ~/LongVideo-R1/.venv/bin/activate
cd ~/LongVideo-R1

VIDEO_DIR=/mnt/llc/llc_data/clams/wgbh/NewsHour

# Question 1
echo "=== Q1: cpb-aacip-507-jw86h4dh3c ==="
python3 cli.py \
  --video_path $VIDEO_DIR/cpb-aacip-507-jw86h4dh3c.mp4 \
  --question "What information did the on-screen text 'Dallas' and 'KERBY ANDERSON' provide about the speaker Kirby Anderson?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1 \
  --caption_model Qwen2.5-VL-32B \
  --videoqa_model Qwen2.5-VL-32B \
  2>&1 | grep -E "^\[Answer\]|^\[Question\]|^\[Model"

echo ""
echo "=== Q2: cpb-aacip-507-5q4rj49c6q ==="
python3 cli.py \
  --video_path $VIDEO_DIR/cpb-aacip-507-5q4rj49c6q.mp4 \
  --question "How did the attacks during Shiite religious ceremonies compare to attacks on coalition forces in Iraq?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1 \
  --caption_model Qwen2.5-VL-32B \
  --videoqa_model Qwen2.5-VL-32B \
  2>&1 | grep -E "^\[Answer\]|^\[Question\]|^\[Model"

echo ""
echo "=== Q3: cpb-aacip-507-4t6f18t00t ==="
python3 cli.py \
  --video_path $VIDEO_DIR/cpb-aacip-507-4t6f18t00t.mp4 \
  --question "How did the speaker describe the potential physical impact of an attack on a major financial institution compared to the psychological impact?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1 \
  --caption_model Qwen2.5-VL-32B \
  --videoqa_model Qwen2.5-VL-32B \
  2>&1 | grep -E "^\[Answer\]|^\[Question\]|^\[Model"

echo ""
echo "=== Q4: cpb-aacip-507-6h4cn6zk04 ==="
python3 cli.py \
  --video_path $VIDEO_DIR/cpb-aacip-507-6h4cn6zk04.mp4 \
  --question "What was the speaker's main criticism regarding the handling of the Hassenfuss incident?" \
  --reasoning_base_url http://127.0.0.1:25600/v1 \
  --caption_base_url http://127.0.0.1:9081/v1 \
  --videoqa_base_url http://127.0.0.1:9081/v1 \
  --caption_model Qwen2.5-VL-32B \
  --videoqa_model Qwen2.5-VL-32B \
  2>&1 | grep -E "^\[Answer\]|^\[Question\]|^\[Model"

echo ""
echo "=== EVAL COMPLETE ==="
