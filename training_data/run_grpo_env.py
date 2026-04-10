#!/usr/bin/env python3
"""GRPO with TRL environment_factory for multi-turn tool execution.

Uses TRL's built-in multi-turn support: the environment class defines
tools as methods, TRL handles the generate -> execute -> feed back loop.

Usage:
    CUDA_VISIBLE_DEVICES=2 python training_data/run_grpo_env.py \
        --model training_data/output/qwen35-9b-tool-exec-lora/adapter \
        --base-model Qwen/Qwen3.5-9B \
        --questions qa-data/raw/qa_v3_combined_valid.jsonl \
        --index-dir data/video_indexes \
        --output training_data/output/qwen35-9b-grpo-env
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

import torch
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, str(Path(__file__).parent))
from construct_tool_trajectories import simulate_tool_output
# prompt_configs.py holds VLM prompt templates for real CLAMS app integration.
# During GRPO simulation, task_mode shapes output via index layer selection
# in the environment class methods, not via prompt templates.
from prompt_configs import TASK_MODES


# ============================================================
# Environment class -- methods become tools
# ============================================================

# Global index storage
_all_indexes = {}
_all_questions = []
_question_idx = 0
_rollout_log = None
_global_step = 0


class VideoIndexEnv:
    """Environment for querying a video index.

    Each public method becomes a tool available to the model.
    TRL discovers methods, builds schemas from docstrings, and
    runs the multi-turn loop automatically.
    """

    def __init__(self):
        global _question_idx
        self.reward = 0.0
        self.expected_answer = ""
        self.tool_calls = []

        # Get current question
        if _question_idx < len(_all_questions):
            q = _all_questions[_question_idx]
            self.video_id = q.get("video_id", "")
            self.index_data = _all_indexes.get(self.video_id, {})
            self.expected_answer = q.get("answer", "")
        else:
            self.video_id = ""
            self.index_data = {}

    def reset(self, **kwargs) -> str:
        """Reset the environment for a new question."""
        global _question_idx
        if _question_idx < len(_all_questions):
            q = _all_questions[_question_idx]
            self.video_id = q.get("video_id", "")
            self.index_data = _all_indexes.get(self.video_id, {})
            self.expected_answer = q.get("answer", "")
            _question_idx += 1
            return q["question"]
        return ""

    def _parse_ts(self, timestamp):
        """Parse MM:SS timestamp to milliseconds."""
        m = re.match(r'(\d+):(\d+)', timestamp)
        return (int(m.group(1)) * 60 + int(m.group(2))) * 1000 if m else 0

    def run_asr(self, model: str = "parakeet") -> str:
        """Run speech recognition on the video. Returns timestamped transcript.

        Args:
            model: ASR model to use. 'parakeet' is fast but may miss unclear speech. 'whisper' is slower but more accurate on noisy audio.
        """
        self.tool_calls.append(("run_asr", model, "speech_lookup"))
        return simulate_tool_output("run_asr", self.index_data, model=model)

    def detect_text_scenes(self) -> str:
        """Detect frames containing text (chyrons, slates, credits). Fast but may miss some."""
        self.tool_calls.append(("detect_text_scenes", None, None))
        return simulate_tool_output("detect_text_scenes", self.index_data)

    def extract_text(self, timestamp: str, model: str = "qwen-8b") -> str:
        """Run OCR on a specific frame to read on-screen text.

        Args:
            timestamp: Time in the video (e.g., '15:47')
            model: VLM for OCR. 'qwen-8b' is fast, good for clear text. 'qwen-30b' is slower but reads degraded or small text better.
        """
        self.tool_calls.append(("extract_text", model, "text_focus"))
        return simulate_tool_output("extract_text", self.index_data,
                                    self._parse_ts(timestamp), model=model)

    def caption_frame(self, timestamp: str, model: str = "smolvlm",
                      task_mode: str = "general_scene") -> str:
        """Analyze a video frame with a vision-language model.

        Args:
            timestamp: Time in the video (e.g., '15:47')
            model: VLM to use. 'smolvlm' is fastest. 'qwen-8b' balances speed and detail. 'qwen-30b' gives richest output.
            task_mode: What to look for. 'general_scene' for broad description. 'text_focus' to read on-screen text. 'person_identity' to identify people. 'action_event' to describe what is happening.
        """
        self.tool_calls.append(("caption_frame", model, task_mode))
        return simulate_tool_output("caption_frame", self.index_data,
                                    self._parse_ts(timestamp),
                                    model=model, task_mode=task_mode)

    def identify_speakers(self) -> str:
        """Run speaker diarization to identify who speaks when."""
        self.tool_calls.append(("identify_speakers", None, None))
        return simulate_tool_output("identify_speakers", self.index_data)

    def browse_timeline(self, start_time: str, end_time: str) -> str:
        """Get everything happening in a time range across all layers.

        Args:
            start_time: Start time (e.g., '15:00')
            end_time: End time (e.g., '16:00')
        """
        self.tool_calls.append(("browse_timeline", None, None))
        start_ms = end_ms = 0
        m = re.match(r'(\d+):(\d+)', start_time)
        if m:
            start_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
        m = re.match(r'(\d+):(\d+)', end_time)
        if m:
            end_ms = (int(m.group(1)) * 60 + int(m.group(2))) * 1000
        return simulate_tool_output("browse_timeline", self.index_data, start_ms,
                                    {"start_ms": start_ms, "end_ms": end_ms})


# Computational cost per (tool, model) pair, reflecting actual resource usage.
# Scale: 0-1 where 1 = most expensive (large VLM inference).
# VLM models: SmolVLM2-2.2B, Qwen3.5-0.8B, Qwen3.5-9B, Qwen3.5-27B
TOOL_MODEL_COSTS = {
    ("browse_timeline", None): 0.0,
    ("detect_text_scenes", None): 0.1,      # SWT, lightweight CNN
    ("run_asr", "parakeet"): 0.2,            # NVIDIA Parakeet, fast CTC
    ("run_asr", "whisper"): 0.5,             # Whisper large-v3, slower but higher quality
    ("identify_speakers", None): 0.3,        # pyannote diarization
    ("extract_text", "qwen-small"): 0.2,     # Qwen3.5-0.8B, fastest but limited
    ("extract_text", "qwen-8b"): 0.4,        # Qwen3.5-9B, good OCR
    ("extract_text", "qwen-30b"): 0.8,       # Qwen3.5-27B, best OCR quality
    ("caption_frame", "smolvlm"): 0.3,       # SmolVLM2-2.2B, fastest
    ("caption_frame", "qwen-small"): 0.3,    # Qwen3.5-0.8B, fast but brief
    ("caption_frame", "qwen-8b"): 0.5,       # Qwen3.5-9B, good balance
    ("caption_frame", "qwen-30b"): 0.9,      # Qwen3.5-27B, richest output
}

# Valid models per tool (for parameter validation)
TOOL_MODELS = {
    "run_asr": ["parakeet", "whisper"],
    "extract_text": ["qwen-small", "qwen-8b", "qwen-30b"],
    "caption_frame": ["smolvlm", "qwen-small", "qwen-8b", "qwen-30b"],
}

# Default model per tool (cheapest)
TOOL_MODEL_DEFAULTS = {
    "run_asr": "parakeet",
    "extract_text": "qwen-small",
    "caption_frame": "smolvlm",
}


def get_tool_cost(tool_name, model=None, task_mode=None):
    """Look up cost for a (tool, model) pair. task_mode doesn't affect cost."""
    if model is None:
        model = TOOL_MODEL_DEFAULTS.get(tool_name)
    return TOOL_MODEL_COSTS.get((tool_name, model), 0.5)


def reward_func(prompts=None, completions=None, environments=None, **kwargs):
    """Compute rewards based on answer correctness + cost-weighted efficiency.

    The reward gates on correctness: if the answer is wrong, reward is 0.
    If correct, reward is scaled by computational cost of tools used.

    TRL passes: prompts, completions, completion_ids, environments
    """
    global _global_step
    _global_step += 1
    rewards = []
    envs = environments or []

    for i in range(len(envs)):
        env = envs[i]
        n_tools = len(env.tool_calls)
        expected = env.expected_answer.lower().strip()

        # Extract the model's final answer from the completion
        answer = ""
        if completions and i < len(completions):
            text = completions[i] if isinstance(completions[i], str) else str(completions[i])
            # Strip thinking and tool call blocks
            clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            clean = re.sub(r'<tool_call>.*?</tool_call>', '', clean, flags=re.DOTALL)
            clean = re.sub(r'<tool_response>.*?</tool_response>', '', clean, flags=re.DOTALL)
            # Take last substantive line as answer
            lines = [l.strip() for l in clean.split('\n') if l.strip() and len(l.strip()) > 3]
            answer = lines[-1].lower() if lines else ""

        # Correctness check (gates everything)
        correct = False
        if expected and answer:
            if expected in answer or answer in expected:
                correct = True
            else:
                exp_words = set(expected.split()) - {"the", "a", "an", "of", "in", "on", "at", "to", "and", "or"}
                ans_words = set(answer.split())
                if exp_words and len(exp_words & ans_words) >= len(exp_words) * 0.5:
                    correct = True

        if not correct:
            rewards.append(0.0)
        else:
            # Cost-weighted efficiency: sum actual computational cost of tools used
            # tool_calls are (tool_name, model, task_mode) tuples
            total_cost = sum(get_tool_cost(t, m, tm) for t, m, tm in env.tool_calls)
            # Efficiency reward: 1.0 for cheap pipelines, decays toward 0.3 for expensive ones
            # Cost budget of ~0.6 (e.g. SWT + Parakeet ASR) gets full reward
            efficiency = max(0.3, 1.0 - (total_cost - 0.6) * 0.35)
            efficiency = min(1.0, efficiency)
            rewards.append(efficiency)

        # Log rollout details
        if _rollout_log:
            prompt_text = ""
            if prompts and i < len(prompts):
                p = prompts[i]
                if isinstance(p, list):
                    prompt_text = p[-1].get("content", "") if p else ""
                else:
                    prompt_text = str(p)
            total_cost = sum(get_tool_cost(t, m, tm) for t, m, tm in env.tool_calls)
            _rollout_log.write(json.dumps({
                "step": _global_step,
                "idx": i,
                "video_id": getattr(env, "video_id", ""),
                "question": prompt_text[:200],
                "tool_calls": [{"tool": t, "model": m, "task_mode": tm}
                               for t, m, tm in env.tool_calls],
                "expected": expected,
                "predicted": answer[:200],
                "correct": correct,
                "reward": rewards[-1],
                "total_cost": round(total_cost, 2),
                "n_tools": n_tools,
            }) + "\n")
            _rollout_log.flush()

    return rewards


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-questions", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    args = parser.parse_args()

    global _all_indexes, _all_questions, _question_idx, _rollout_log

    # Load indexes
    for f in args.index_dir.glob("*.json"):
        _all_indexes[f.stem] = json.load(open(f))

    # Load questions (train split only!)
    with open(args.questions) as f:
        all_qs = [json.loads(l) for l in f]
    _all_questions = [q for q in all_qs if q.get("video_id") in _all_indexes][:args.max_questions]
    _question_idx = 0

    print(f"Questions loaded: {len(_all_questions)}")
    print(f"Expected answers available: {sum(1 for q in _all_questions if q.get('answer'))}")

    # Build dataset -- prompts as chat messages
    prompts = []
    for q in _all_questions:
        messages = [{"role": "user", "content": q["question"]}]
        prompts.append(messages)
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"Dataset: {len(dataset)} questions")

    # Load model
    print(f"Loading {args.base_model} + LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.model))
    model = model.merge_and_unload()

    # Fresh LoRA for GRPO
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    grpo_config = GRPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=1024,
        learning_rate=5e-6,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="tensorboard",
        logging_dir=str(args.output / "tensorboard"),
        gradient_checkpointing=True,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=VideoIndexEnv,
    )

    rollout_path = args.output / "rollouts.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)
    _rollout_log = open(rollout_path, "w")
    print(f"  Rollout log: {rollout_path}")

    print(f"\nStarting GRPO with environment_factory...")
    print(f"  Questions: {len(dataset)}")
    print(f"  Generations: {args.num_generations}")
    print(f"  Tools: run_asr, detect_text_scenes, extract_text, caption_frame, identify_speakers, browse_timeline")

    trainer.train()

    if _rollout_log:
        _rollout_log.close()

    print(f"\nSaving...")
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done!")


if __name__ == "__main__":
    main()
