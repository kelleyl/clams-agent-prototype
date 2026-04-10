#!/usr/bin/env python3
"""Generate caption/OCR variants for video indexes using multiple VLMs and task modes.

Extracts keyframes from videos and runs each (model, task_mode) combination,
storing results as separate layers in the index. This provides real model-specific
output for GRPO training simulation instead of synthetic degradation.

Models (served via vLLM):
  - SmolVLM2-2.2B (smolvlm)
  - Qwen3.5-0.8B (qwen-small)
  - Qwen3.5-9B (qwen-8b)
  - Qwen3.5-27B (qwen-30b)

Task modes:
  - general_scene: broad visual description
  - text_focus: OCR-like text extraction
  - person_identity: person identification
  - action_event: action/event description

Usage:
    # Start vLLM server first:
    # vllm serve Qwen/Qwen3.5-9B --port 8200 --max-model-len 8192

    python training_data/generate_caption_variants.py \
        --index-dir data/video_indexes \
        --vllm-url http://localhost:8200 \
        --model qwen-8b \
        --task-modes general_scene text_focus person_identity action_event \
        --frames-per-video 20 \
        --max-videos 10
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from prompt_configs import get_caption_prompt, TASK_MODES

# Map our model param names to actual HuggingFace model IDs for vLLM
MODEL_IDS = {
    "smolvlm": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "qwen-small": "Qwen/Qwen3.5-0.8B",
    "qwen-8b": "Qwen/Qwen3.5-9B",
    "qwen-30b": "Qwen/Qwen3.5-27B",
}

# Ollama model names (alternative backend)
OLLAMA_MODELS = {
    "smolvlm": "richardyoung/smolvlm2-2.2b-instruct",
    "qwen-small": "qwen3.5:0.8b",
    "qwen-8b": "qwen3.5:9b",
    "qwen-30b": "qwen3.5:27b",
}

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")


def extract_keyframes(video_path, index_data, n_frames=20, tmp_dir=None):
    """Extract keyframes at scene boundaries or evenly spaced."""
    duration_ms = index_data.get("duration_ms", 0)
    if duration_ms <= 0:
        return []

    # Try to use scene boundaries for more interesting frames
    scenes = index_data.get("layers", {}).get("scenes", {}).get("items", [])
    timestamps_ms = []

    if scenes and len(scenes) >= n_frames // 2:
        # Sample from scene start times
        for item in scenes:
            ts = item.get("start_ms", 0) + 500  # slightly after scene start
            timestamps_ms.append(ts)
        # Subsample if too many
        if len(timestamps_ms) > n_frames:
            step = len(timestamps_ms) / n_frames
            timestamps_ms = [timestamps_ms[int(i * step)] for i in range(n_frames)]
    else:
        # Evenly spaced
        step = duration_ms / (n_frames + 1)
        timestamps_ms = [int((i + 1) * step) for i in range(n_frames)]

    # Extract frames with ffmpeg
    frames = []
    for i, ts_ms in enumerate(timestamps_ms):
        t_sec = ts_ms / 1000.0
        out_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(t_sec),
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",
                out_path,
            ], capture_output=True, timeout=15, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                frames.append({"path": out_path, "timestamp_ms": ts_ms})
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue

    return frames


def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_frame_vllm(frame_path, prompt, vllm_url, model_id):
    """Send frame + prompt to vLLM OpenAI-compatible vision API."""
    b64 = encode_image_base64(frame_path)
    resp = requests.post(
        f"{vllm_url}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def caption_frame_ollama(frame_path, prompt, model_name):
    """Send frame + prompt to Ollama vision API."""
    b64 = encode_image_base64(frame_path)
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [b64],
            }],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500},
        },
        timeout=120,
    )
    resp.raise_for_status()
    import re
    content = resp.json().get("message", {}).get("content", "")
    # Strip thinking tags if present
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


# Global transformers model/processor (loaded once)
_hf_model = None
_hf_processor = None


def _load_hf_model(model_id, device="cuda", quantize=False):
    """Load a HuggingFace VLM model + processor."""
    global _hf_model, _hf_processor
    if _hf_model is not None:
        return _hf_model, _hf_processor

    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if quantize:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"  Loading {model_id} via transformers (4-bit quantized)...", flush=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print(f"  Loading {model_id} via transformers (bf16)...", flush=True)

    _hf_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _hf_model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    _hf_model.eval()
    return _hf_model, _hf_processor


def caption_frame_transformers(frame_path, prompt, model_id):
    """Run VLM inference directly via HuggingFace transformers (single frame)."""
    return caption_batch_transformers([(frame_path, prompt)], model_id)[0]


def caption_batch_transformers(frame_prompt_pairs, model_id, quantize=False):
    """Run batched VLM inference on multiple (frame_path, prompt) pairs."""
    import torch
    from PIL import Image
    import re as _re

    model, processor = _load_hf_model(model_id, quantize=quantize)

    results = []
    # Process in batches of 8
    batch_size = 8
    for batch_start in range(0, len(frame_prompt_pairs), batch_size):
        batch = frame_prompt_pairs[batch_start:batch_start + batch_size]

        images = []
        text_inputs = []
        for frame_path, prompt in batch:
            image = Image.open(frame_path).convert("RGB")
            images.append(image)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text_input = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)
            text_inputs.append(text_input)

        inputs = processor(text=text_inputs, images=images, return_tensors="pt",
                           padding=True).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=500,
                                         temperature=0.1, do_sample=True)

        # Decode each sample in the batch
        input_len = inputs["input_ids"].shape[1]
        for i in range(len(batch)):
            generated = output_ids[i][input_len:]
            text = processor.decode(generated, skip_special_tokens=True).strip()
            # Strip thinking blocks (tagged or untagged)
            text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
            # Strip "The user wants me to..." preamble from Qwen3.5
            text = _re.sub(r'^The user wants.*?\n\n', '', text, flags=_re.DOTALL).strip()
            text = _re.sub(r'^\*\*\d+\.\s+', '', text).strip()
            results.append(text)

    return results


def process_video(index_path, index_data, model, task_modes, backend, vllm_url,
                  n_frames=20, quantize=False):
    """Run VLM on frames for all task modes, return new layer data."""
    video_path = index_data.get("video_path", "")
    if not video_path or not os.path.exists(video_path):
        return None

    model_id = MODEL_IDS.get(model, model)
    ollama_name = OLLAMA_MODELS.get(model, model)

    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = extract_keyframes(video_path, index_data, n_frames, tmp_dir)
        if not frames:
            print(f"    No frames extracted", flush=True)
            return None

        results = {}
        for task_mode in task_modes:
            prompt = get_caption_prompt(model, task_mode)
            items = []

            if backend == "transformers":
                # Batch all frames for this task mode
                pairs = [(f["path"], prompt) for f in frames]
                try:
                    texts = caption_batch_transformers(pairs, model_id, quantize=quantize)
                except Exception as e:
                    texts = [f"Error: {e}"] * len(frames)
                for j, frame in enumerate(frames):
                    items.append({
                        "id": f"vcap_{j}",
                        "start_ms": frame["timestamp_ms"],
                        "end_ms": frame["timestamp_ms"] + 1000,
                        "text": texts[j],
                    })
            else:
                for frame in frames:
                    try:
                        if backend == "vllm":
                            text = caption_frame_vllm(frame["path"], prompt,
                                                      vllm_url, model_id)
                        else:
                            text = caption_frame_ollama(frame["path"], prompt,
                                                       ollama_name)
                    except Exception as e:
                        text = f"Error: {e}"
                    items.append({
                        "id": f"vcap_{len(items)}",
                        "start_ms": frame["timestamp_ms"],
                        "end_ms": frame["timestamp_ms"] + 1000,
                        "text": text,
                    })

            layer_name = f"caption_{model}_{task_mode}"
            results[layer_name] = {
                "source": model_id,
                "model": model,
                "task_mode": task_mode,
                "prompt": prompt[:200],
                "n_frames": len(items),
                "items": items,
            }
            print(f"    {task_mode}: {len(items)} frames captioned", flush=True)

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--model", type=str, required=True,
                        choices=["smolvlm", "qwen-small", "qwen-8b", "qwen-30b"])
    parser.add_argument("--task-modes", nargs="+", default=TASK_MODES,
                        choices=TASK_MODES)
    parser.add_argument("--backend", choices=["vllm", "ollama", "transformers"],
                        default="ollama")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8200")
    parser.add_argument("--frames-per-video", type=int, default=20)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--regenerate", action="store_true", default=False,
                        help="Regenerate caption layers even if they already exist")
    parser.add_argument("--quantize", action="store_true", default=False,
                        help="Load model in 4-bit quantization (for large models)")
    args = parser.parse_args()

    index_files = sorted(args.index_dir.glob("*.json"))
    print(f"Model: {args.model} ({MODEL_IDS.get(args.model, '?')})")
    print(f"Backend: {args.backend}")
    print(f"Task modes: {args.task_modes}")
    print(f"Frames per video: {args.frames_per_video}")
    print(f"Indexes: {len(index_files)}")

    processed = 0
    skipped = 0
    errors = 0

    for idx_path in index_files:
        if args.max_videos and processed >= args.max_videos:
            break

        idx = json.load(open(idx_path))

        # Check if all requested layers already exist
        if not args.regenerate:
            expected = [f"caption_{args.model}_{tm}" for tm in args.task_modes]
            if all(layer in idx.get("layers", {}) for layer in expected):
                skipped += 1
                continue

        print(f"[{processed + 1}] {idx_path.stem}")
        t0 = time.time()

        try:
            new_layers = process_video(
                idx_path, idx, args.model, args.task_modes, args.backend,
                args.vllm_url, args.frames_per_video,
                quantize=args.quantize,
            )

            if new_layers:
                if "layers" not in idx:
                    idx["layers"] = {}
                idx["layers"].update(new_layers)

                with open(idx_path, "w") as f:
                    json.dump(idx, f, ensure_ascii=False, indent=2)

                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.1f}s")
                processed += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")


if __name__ == "__main__":
    main()
