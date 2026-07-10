#!/usr/bin/env python3
"""Two-stage eval: a policy model gathers evidence, then an answerer answers.

Stage 1 (Policy): The policy model decides which tools to call. It may be the
  base model with no adapter or a LoRA/SFT/GRPO policy. It generates tool
  calls, we execute them, and feed back observations. The policy only needs to
  be good at evidence gathering, not final answer synthesis.

Stage 2 (Answerer): A fresh LLM call with the question + all gathered
  evidence produces the answer. This model sees the full context and
  just needs reading comprehension.

This separates tool-use skill from answer synthesis skill.

Execution modes:
  prebuilt_index: warm-index archive QA. Tools read/search an existing,
    read-only multimodal index. This measures query-time evidence retrieval and
    provenance over a precomputed archive index.
  simulated_cold_cache: cold-cache artifact orchestration. The source index is
    still used as an immutable fixture, but the visible state starts empty and
    tools must create/cache artifacts before reading them. This approximates
    processing a new raw video without mutating data/video_indexes.

Usage:
    CUDA_VISIBLE_DEVICES=2 python eval/run_policy_answerer_eval.py \
        --benchmark qa-data/raw/qa_v3_test.jsonl \
        --index-dir data/video_indexes \
        --output eval/results/v3_policy_answerer_eval.jsonl \
        --policy-adapter training_data/output/qwen35-9b-tool-exec-lora/adapter \
        --answerer-backend local-base
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "training_data"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from construct_tool_trajectories import simulate_tool_output
from utils.answerer_utils import (
    call_answerer, call_loaded_answerer, extract_mc_prediction,
    map_freetext_to_mc,
)
from utils.artifact_registry import ArtifactRegistry
from utils.ctx_retrieval import retrieve_context, _toks
from utils.simulated_artifact_tools import SimulatedArtifactExecutor
from eval.v6_tools import V6_TOOL_SCHEMAS, V6Executor
from eval.v7_tools import V7_TOOL_SCHEMAS, V7Executor


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
ANSWERER_BACKEND = os.environ.get("ANSWERER_BACKEND", "vllm")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8200/v1")

# Tool schemas with model/task_mode parameters matching run_grpo_env.py.
# These give the policy model the ability to select model quality and task focus.
ASR_MODELS = ["parakeet", "whisper"]
OCR_MODELS = ["qwen-small", "qwen-8b", "qwen-30b", "qwen3vl-8b"]
VLM_MODELS = ["smolvlm", "qwen-small", "qwen-8b", "qwen-30b", "qwen3vl-8b"]
CAPTION_TASK_MODES = ["general_scene", "text_focus", "person_identity", "action_event"]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Search the video's speech transcript by keyword. Returns matching segments with timestamps. Use this to locate relevant moments in the video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms (e.g., 'unemployment', 'senator')"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Search on-screen text (chyrons, titles, graphics) by keyword. Returns matching text with timestamps and scene labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms (e.g., 'Urban League', 'Senator')"},
                    "top_k": {"type": "integer", "description": "Number of results to return (default 5)", "default": 5},
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Get speech transcript for a specific time range. Use after locating a relevant region via search_transcript or browse_timeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '16:00')"},
                    "model": {
                        "type": "string",
                        "description": "ASR model. 'parakeet' is fast. 'whisper' is more accurate on noisy audio.",
                        "enum": ASR_MODELS,
                        "default": "parakeet",
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_timeline",
            "description": "Get a compact summary of what happens in a time range: key topics, speakers, on-screen text, and notable timestamps. Use for coarse localization before inspecting specific moments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '20:00')"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_text_scenes",
            "description": "Find all frames containing text (chyrons, slates, credits). Returns timestamps and scene types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text",
            "description": "Run OCR on a specific frame to read on-screen text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video (e.g., '15:47')"},
                    "model": {
                        "type": "string",
                        "description": "VLM for OCR. 'qwen-small' is fastest. 'qwen-8b' is thorough. 'qwen-30b' is strongest when available.",
                        "enum": OCR_MODELS,
                        "default": "qwen-small",
                    },
                },
                "required": ["timestamp"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "caption_frame",
            "description": "Analyze video frames with a vision-language model. Provide a single timestamp for one frame, or start_time+end_time to caption all shots in a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Single frame time (e.g., '15:47'). Use this OR start_time+end_time."},
                    "start_time": {"type": "string", "description": "Start of time range (e.g., '1:00'). Use with end_time."},
                    "end_time": {"type": "string", "description": "End of time range (e.g., '1:30'). Use with start_time."},
                    "model": {
                        "type": "string",
                        "description": "VLM to use. 'qwen3vl-8b' has best coverage. 'qwen-8b' is detailed. 'qwen-30b' is strongest when available.",
                        "enum": VLM_MODELS,
                        "default": "qwen3vl-8b",
                    },
                    "task_mode": {
                        "type": "string",
                        "description": "What to look for. 'general_scene' for broad description, 'text_focus' for on-screen text, 'person_identity' for visible people, 'action_event' for actions/events.",
                        "enum": CAPTION_TASK_MODES,
                        "default": "general_scene",
                    },
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "identify_speakers",
            "description": "Identify who is speaking in a time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time (e.g., '15:00')"},
                    "end_time": {"type": "string", "description": "End time (e.g., '16:00')"},
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
]


def without_browse_timeline(tool_schemas):
    """Return a deep-copied schema list with browse_timeline fully hidden."""
    filtered = []
    for schema in json.loads(json.dumps(tool_schemas)):
        function = schema.get("function", {})
        if function.get("name") == "browse_timeline":
            continue
        description = function.get("description")
        if description:
            function["description"] = (
                description
                .replace(" or browse_timeline", "")
                .replace(" via search_transcript or browse_timeline", " via search_transcript")
                .replace(" via search or browsing", " via search")
            )
        filtered.append(schema)
    return filtered


V5_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_transcript",
            "description": "Search speech transcript by keyword. Requires run_asr to have been called first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms, e.g. 'unemployment' or 'Senator'"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_ocr",
            "description": "Search on-screen text by keyword. Requires run_swt + run_ocr to have been called first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search terms, e.g. 'director' or 'Urban League'"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_asr",
            "description": "Create or reuse a speech transcript artifact. Use this only when speech evidence is useful. Omit start/end to process the full video; provide start/end to process a bounded range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Optional start time, e.g. '15:00'"},
                    "end_time": {"type": "string", "description": "Optional end time, e.g. '16:00'"},
                    "model": {
                        "type": "string",
                        "description": "ASR model. 'parakeet' is fast; 'whisper' is stronger on noisy audio.",
                        "enum": ASR_MODELS,
                        "default": "parakeet",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_asr",
            "description": "Read bounded transcript evidence from an existing ASR artifact. Requires run_asr to have created an artifact that covers the requested range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Start time, e.g. '15:00'"},
                    "end_time": {"type": "string", "description": "End time, e.g. '16:00'"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_swt",
            "description": "Detect frames/scenes that contain visible text, such as slates, chyrons, title cards, and credits. Use this for production metadata or on-screen-name questions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_swt",
            "description": "Read text-scene candidates from an existing SWT artifact. Requires run_swt to have been called earlier for this video.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ocr",
            "description": "Run OCR on a specific timestamp to read visible text. Use after identifying a likely text scene or when the question asks about slates, credits, chyrons, or title cards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Time in the video, e.g. '15:47'"},
                    "model": {
                        "type": "string",
                        "description": "OCR/VLM model. 'qwen-small' is fast; 'qwen-8b' is thorough; 'qwen-30b' is strongest when available.",
                        "enum": OCR_MODELS,
                        "default": "qwen-small",
                    },
                },
                "required": ["timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_ocr",
            "description": "Read cached OCR evidence. Use artifact_id from list_cache_artifacts or provide a timestamp covered by a prior run_ocr call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "Optional OCR artifact id"},
                    "timestamp": {"type": "string", "description": "Optional timestamp, e.g. '15:47'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_captioner",
            "description": "Run a vision-language captioner on video frames. Provide a single timestamp for one frame, or start_time+end_time to caption all shots in a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string", "description": "Single frame time (e.g., '15:47'). Use this OR start_time+end_time."},
                    "start_time": {"type": "string", "description": "Start of time range (e.g., '1:00'). Use with end_time to caption all shots."},
                    "end_time": {"type": "string", "description": "End of time range (e.g., '1:30'). Use with start_time."},
                    "model": {
                        "type": "string",
                        "description": "VLM model. 'qwen3vl-8b' has best coverage. 'qwen-8b' gives more detail. 'qwen-30b' is strongest when available.",
                        "enum": VLM_MODELS,
                        "default": "qwen3vl-8b",
                    },
                    "task_mode": {
                        "type": "string",
                        "description": "What to focus on: broad scene, on-screen text, visible people, or actions/events.",
                        "enum": CAPTION_TASK_MODES,
                        "default": "general_scene",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_caption",
            "description": "Read cached visual caption evidence. Use artifact_id from list_cache_artifacts or provide a timestamp covered by a prior run_captioner call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "Optional caption artifact id"},
                    "timestamp": {"type": "string", "description": "Optional timestamp, e.g. '15:47'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_diarization",
            "description": "Create or read speaker diarization evidence over a time range. Use when speaker identity or who-spoke-when is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "Optional start time, e.g. '15:00'"},
                    "end_time": {"type": "string", "description": "Optional end time, e.g. '16:00'"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_diarization",
            "description": "Read cached speaker diarization evidence from a prior run_diarization call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artifact_id": {"type": "string", "description": "Optional diarization artifact id"},
                    "start_time": {"type": "string", "description": "Optional start time, e.g. '15:00'"},
                    "end_time": {"type": "string", "description": "Optional end time, e.g. '16:00'"},
                },
                "required": [],
            },
        },
    },
]

# Default model per tool (cheapest option, matches run_grpo_env.py)
TOOL_MODEL_DEFAULTS = {
    "run_asr": "parakeet",
    "extract_text": "qwen-small",
    "caption_frame": "smolvlm",
}


# ============================================================
# Stage 1: Policy model (tool selection)
# ============================================================

def load_policy_model(base_model, adapter_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter_path:
        print(f"Loading policy model: {base_model} + {adapter_path}")
    else:
        print(f"Loading base policy model (no adapter): {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def policy_generate(model, tokenizer, messages, tools, max_new_tokens=300):
    import torch

    text = tokenizer.apply_chat_template(
        messages, tools=tools,
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def policy_generate_ollama(messages, tools, model_name, ollama_url, max_new_tokens=500):
    """Generate policy output via Ollama, supporting multimodal (image) inputs.

    Messages may contain '_v6_image_b64' fields with base64-encoded images.
    These are converted to Ollama's 'images' format for VLM models.
    """
    import requests

    # Convert messages to Ollama format, extracting images
    ollama_messages = []
    for msg in messages:
        omsg = {"role": msg["role"], "content": msg.get("content", "")}

        # Handle tool calls in assistant messages
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            fn = tc["function"]
            omsg["content"] = (omsg["content"] or "") + f"\n\nI'll call {fn['name']}({json.dumps(fn.get('arguments', {}))})"

        # Handle images attached to tool responses
        if msg.get("_v6_image_b64"):
            omsg["images"] = [msg["_v6_image_b64"]]

        ollama_messages.append(omsg)

    # Build tool descriptions into the system prompt since Ollama doesn't
    # natively support tool schemas for all models.
    # No-tools mode (empty tools list): skip injection entirely so the
    # model is not primed to emit <tool_call> syntax.
    tool_desc = "You have access to these tools:\n\n"
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        param_strs = []
        for pname, pdef in params.items():
            param_strs.append(f"  - {pname}: {pdef.get('description', '')}")
        tool_desc += f"### {fn['name']}\n{fn.get('description', '')}\n"
        if param_strs:
            tool_desc += "Parameters:\n" + "\n".join(param_strs) + "\n"
        tool_desc += "\n"

    tool_desc += (
        "To call a tool, respond with EXACTLY this format:\n"
        "<tool_call>\n<function=TOOL_NAME>\n"
        "<parameter=PARAM_NAME>value</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "When you have enough evidence, provide your final answer as JSON:\n"
        '{"answer": "...", "evidence": [...], "rationale": "..."}'
    )

    # Prepend tool descriptions to system message or first user message
    if tools:
        if ollama_messages and ollama_messages[0]["role"] == "system":
            ollama_messages[0]["content"] = ollama_messages[0]["content"] + "\n\n" + tool_desc
        else:
            ollama_messages.insert(0, {"role": "system", "content": tool_desc})

    resp = requests.post(f"{ollama_url}/api/chat", json={
        "model": model_name,
        "messages": ollama_messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": max_new_tokens},
    }, timeout=120)

    if resp.status_code != 200:
        return f"Error: Ollama returned {resp.status_code}: {resp.text[:200]}"

    return resp.json().get("message", {}).get("content", "")


def policy_generate_vllm(messages, tools, model_name, vllm_url, max_new_tokens=500):
    """Generate policy output via vLLM's OpenAI-compatible API.

    Uses the /v1/chat/completions endpoint. Tool descriptions are injected
    into the system prompt for models that support tool calling via prompting.
    """
    import requests

    # Build tool descriptions into system prompt.
    # No-tools mode (empty tools list): skip injection entirely so the
    # model is not primed to emit <tool_call> syntax.
    tool_desc = "You are a video research agent. You MUST use tools to answer questions about videos. Never answer without gathering evidence first.\n\nAvailable tools:\n\n"
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        param_strs = []
        for pname, pdef in params.items():
            param_strs.append(f"  - {pname}: {pdef.get('description', '')}")
        tool_desc += f"### {fn['name']}\n{fn.get('description', '')}\n"
        if param_strs:
            tool_desc += "Parameters:\n" + "\n".join(param_strs) + "\n"
        tool_desc += "\n"

    tool_desc += (
        "To call a tool, respond with EXACTLY this format:\n"
        "<tool_call>\n<function=TOOL_NAME>\n"
        "<parameter=PARAM_NAME>value</parameter>\n"
        "</function>\n</tool_call>\n\n"
        "You MUST start by calling get_video_info, then get_video_grid to survey the video.\n"
        "After seeing the grid, use get_transcript, get_text_on_screen, or get_visual_description on specific time ranges.\n"
        "When you have enough evidence, provide your final answer as JSON (no tool call):\n"
        '{"answer": "your answer", "evidence": [{"tool": "...", "timestamp": "...", "quote_or_summary": "..."}], "rationale": "..."}\n\n'
        "For multiple-choice questions, select one option (A/B/C/D). Base your answer ONLY on evidence from tools."
    )

    # Build OpenAI-format messages with vision support
    api_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        image_b64 = msg.get("_v6_image_b64")

        if role == "system":
            if tools:
                api_messages.append({"role": "system", "content": content + "\n\n" + tool_desc})
            else:
                api_messages.append({"role": "system", "content": content})
        elif role == "user":
            if not api_messages and tools:
                api_messages.append({"role": "system", "content": tool_desc})
            api_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tc = msg.get("tool_calls")
            if tc:
                fn = tc[0]["function"]
                tool_text = (content or "") + f"\n<tool_call>\n<function={fn['name']}>\n"
                for k, v in fn.get("arguments", {}).items():
                    tool_text += f"<parameter={k}>{v}</parameter>\n"
                tool_text += "</function>\n</tool_call>"
                api_messages.append({"role": "assistant", "content": tool_text})
            else:
                api_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            # Tool response -- may include an image (grid contact sheet)
            if image_b64:
                # OpenAI vision format: multipart content with text + image
                api_messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"Tool result:\n{content}"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }},
                ]})
            else:
                api_messages.append({"role": "user", "content": f"Tool result:\n{content}"})

    # Normalize URL: ensure it ends with /v1 for OpenAI-compatible API
    base = vllm_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    resp = requests.post(f"{base}/chat/completions", json={
        "model": model_name,
        "messages": api_messages,
        "max_tokens": max_new_tokens,
        "temperature": 0.3,
    }, timeout=180)

    if resp.status_code != 200:
        return f"Error: vLLM returned {resp.status_code}: {resp.text[:300]}"

    choices = resp.json().get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    return "Error: no response from vLLM"


def parse_tool_call(text):
    # Qwen3.5 native format: <tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>
    match = re.search(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        text, re.DOTALL
    )
    if match:
        func_name = match.group(1)
        params = {}
        for m in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', match.group(2), re.DOTALL):
            val = m.group(2).strip()
            try:
                val = int(val)
            except ValueError:
                pass
            params[m.group(1)] = val
        return func_name, params

    # Ollama/VLM format: <function=NAME><parameter>K</parameter>V ... </function>
    match = re.search(r'<function=(\w+)>(.*?)</function>', text, re.DOTALL)
    if match:
        func_name = match.group(1)
        params = {}
        # Try <parameter=K>V</parameter> first
        for m in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', match.group(2), re.DOTALL):
            val = m.group(2).strip()
            try:
                val = int(val)
            except ValueError:
                pass
            params[m.group(1)] = val
        # Also try <parameter>K</parameter>V pattern (some VLMs produce this)
        if not params:
            pairs = re.findall(r'<parameter>(\w+)</parameter>\s*(.+?)(?=<parameter>|$)', match.group(2), re.DOTALL)
            for k, v in pairs:
                val = v.strip()
                try:
                    val = int(val)
                except ValueError:
                    pass
                params[k] = val
        return func_name, params

    return None, None


def _parse_time_ms(time_str):
    """Parse a time value to milliseconds.

    Accepts MM:SS (also H:MM:SS), bare numbers (seconds), and numeric types.
    Weaker policies (e.g. ollama models) emit all of these."""
    if time_str is None or time_str == "":
        return None
    if isinstance(time_str, (int, float)):
        return int(float(time_str) * 1000)
    s = str(time_str).strip()
    m = re.match(r'^(?:(\d+):)?(\d+):(\d+)', s)
    if m:
        hours = int(m.group(1)) if m.group(1) else 0
        return ((hours * 60 + int(m.group(2))) * 60 + int(m.group(3))) * 1000
    m = re.match(r'^(\d+(?:\.\d+)?)$', s)
    if m:
        return int(float(m.group(1)) * 1000)
    return None


def execute_tool(func_name, params, index_data):
    """Execute a tool call via the canonical simulate_tool_output interface.

    All tool logic lives in simulate_tool_output() in
    construct_tool_trajectories.py. This function parses params
    and delegates, ensuring train/eval parity.
    """
    model = params.get("model", TOOL_MODEL_DEFAULTS.get(func_name))
    task_mode = params.get("task_mode")
    query = params.get("query")
    top_k = params.get("top_k", 5)
    timestamp_ms = _parse_time_ms(params.get("timestamp"))
    start_ms = _parse_time_ms(params.get("start_time"))
    end_ms = _parse_time_ms(params.get("end_time"))

    # Defensive defaults: policies sometimes emit half a time range, which
    # crashes the range comparisons inside simulate_tool_output.
    if start_ms is not None and end_ms is None:
        end_ms = start_ms + 60000
    elif end_ms is not None and start_ms is None:
        start_ms = max(0, end_ms - 60000)

    return simulate_tool_output(
        func_name, index_data,
        timestamp_ms=timestamp_ms,
        context={"start_ms": start_ms, "end_ms": end_ms} if start_ms is not None else None,
        model=model, task_mode=task_mode,
        query=query, top_k=top_k,
        start_ms=start_ms, end_ms=end_ms,
    )


# ============================================================
# Policy-free context builders (--oracle-evidence / --rag)
# ============================================================

ORACLE_SPAN_CHAR_CAP = 8000
RAG_TOP_ASR = 8
RAG_MAX_CHARS_PER_ITEM = 4000


def _format_ms(ms):
    ms = int(ms or 0)
    return f"{ms // 60000}:{(ms // 1000) % 60:02d}"


def _layer_items(layer):
    if layer is None:
        return []
    return layer.get("items", []) if isinstance(layer, dict) else layer


def _primary_asr_items(layers):
    """Return the primary ASR layer items, falling back to any asr* layer."""
    items = _layer_items(layers.get("asr"))
    if items:
        return items
    for name in sorted(layers):
        if name.lower().startswith("asr"):
            items = _layer_items(layers[name])
            if items:
                return items
    return []


def _overlapping_items(items, start_ms, end_ms):
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        s = it.get("start_ms")
        if s is None:
            continue
        e = it.get("end_ms")
        if e is None:
            e = s
        if e >= start_ms and s <= end_ms:
            out.append(it)
    return out


def _is_visual_row(q):
    return bool((q.get("v6") or {}).get("visual_evidence")) \
        or q.get("modalities_required") == ["visual"]


def build_oracle_evidence(q, index_data):
    """Matched-evidence ceiling: build answerer context directly from the
    benchmark row's source_segment_times, with no policy loop.

    ASR text overlapping each span is always included. For visual rows
    (v6.visual_evidence present or modalities_required == ["visual"]),
    visual-evidence text stored in the index (OCR / text_focus / caption
    layers) for those spans is added, plus the row's stored
    v6.visual_evidence string when present.
    """
    layers = index_data.get("layers", {})
    asr_items = _primary_asr_items(layers)
    visual = _is_visual_row(q)
    evidence = []

    for span in q.get("source_segment_times") or []:
        if not isinstance(span, dict):
            continue
        start_ms = int(span.get("start_ms") or 0)
        end_ms = int(span.get("end_ms") or start_ms)

        lines = []
        for it in _overlapping_items(asr_items, start_ms, end_ms):
            txt = (it.get("text") or "").strip()
            if txt:
                lines.append(f"[{_format_ms(it.get('start_ms'))}] {txt}")
        text = "\n".join(lines)
        if len(text) > ORACLE_SPAN_CHAR_CAP:
            text = text[:ORACLE_SPAN_CHAR_CAP] + "\n... [truncated]"
        if text:
            evidence.append({
                "tool": "oracle_asr",
                "params": {"start_ms": start_ms, "end_ms": end_ms},
                "result": text,
                "execution_mode": "oracle_evidence",
            })

        if visual:
            vis_lines = []
            for lname in sorted(layers):
                ll = lname.lower()
                if not ("ocr" in ll or "text_focus" in ll or ll.startswith("caption")):
                    continue
                for it in _overlapping_items(_layer_items(layers[lname]), start_ms, end_ms):
                    txt = (it.get("text") or "").strip()
                    if txt:
                        vis_lines.append(
                            f"[{_format_ms(it.get('start_ms'))}] ({lname}) {txt}")
            vtext = "\n".join(vis_lines)
            if len(vtext) > ORACLE_SPAN_CHAR_CAP:
                vtext = vtext[:ORACLE_SPAN_CHAR_CAP] + "\n... [truncated]"
            if vtext:
                evidence.append({
                    "tool": "oracle_visual",
                    "params": {"start_ms": start_ms, "end_ms": end_ms},
                    "result": vtext,
                    "execution_mode": "oracle_evidence",
                })

    if visual:
        stored = (q.get("v6") or {}).get("visual_evidence")
        if stored:
            evidence.append({
                "tool": "oracle_stored_visual_evidence",
                "params": {},
                "result": str(stored)[:ORACLE_SPAN_CHAR_CAP],
                "execution_mode": "oracle_evidence",
            })

    return evidence


def build_rag_evidence(q, index_data):
    """RAG baseline: no policy. Keyword retrieval over the index via
    utils.ctx_retrieval.retrieve_context(doc, question, k=8, max_chars=3000)
    plus the top ASR segments (token-overlap scored, time-ordered, with
    timestamps)."""
    question = q["question"]
    evidence = []

    ctx = retrieve_context(index_data, question, k=8, max_chars=3000)
    if ctx:
        evidence.append({
            "tool": "rag_retrieval",
            "params": {"k": 8, "max_chars": 3000},
            "result": ctx,
            "execution_mode": "rag",
        })

    q_toks = set(_toks(question))
    scored = []
    if q_toks:
        for it in _primary_asr_items(index_data.get("layers", {})):
            if not isinstance(it, dict):
                continue
            toks = set(_toks(it.get("text") or ""))
            if not toks:
                continue
            overlap = len(q_toks & toks)
            if overlap:
                scored.append((overlap / (1 + len(toks) ** 0.5), it))
    scored.sort(key=lambda x: -x[0])
    top = [it for _, it in scored[:RAG_TOP_ASR]]
    top.sort(key=lambda it: it.get("start_ms") or 0)
    lines = []
    seen = set()
    for it in top:
        txt = (it.get("text") or "").strip()
        if not txt or txt[:60] in seen:
            continue
        seen.add(txt[:60])
        lines.append(f"[{_format_ms(it.get('start_ms'))}] {txt}")
    if lines:
        evidence.append({
            "tool": "rag_top_asr",
            "params": {"top_k": RAG_TOP_ASR},
            "result": "\n".join(lines),
            "execution_mode": "rag",
        })

    return evidence


SINGLE_AGENT_SYSTEM = """You are a video research agent. Use the available tools to find evidence in the video, then provide your final answer.

When you have gathered enough evidence, respond with a JSON object (no tool call):
{"answer": "your answer (for multiple-choice: the letter like A, B, C, or D)", "evidence": [{"tool": "tool_name", "timestamp": "MM:SS", "quote_or_summary": "key evidence text"}], "rationale": "brief explanation connecting evidence to answer"}

For multiple-choice questions, you MUST select one option (A/B/C/D). Base your answer ONLY on evidence you gathered from the tools, not prior knowledge."""


NO_TOOLS_SYSTEM = """You are answering questions about broadcast video content. You do NOT have access to any tools or video data. Answer based solely on your own knowledge.

Respond with a JSON object:
{"answer": "your answer", "rationale": "brief explanation"}

If the question shows multiple-choice options, you MUST answer with exactly one option letter (A, B, C, or D). Otherwise, give a short, direct free-text answer (a name, phrase, or number). If you are unsure, make your best guess."""


def gather_evidence(
    model,
    tokenizer,
    question,
    index_data,
    max_turns=3,
    tool_executor=None,
    tool_schemas=None,
    single_agent=False,
    policy_backend="local",
    policy_model_name=None,
    ollama_url=None,
    prefill_transcript=False,
):
    """Run model to gather evidence through tool calls.

    Returns (tool_calls, evidence_pieces, final_answer_text).
    final_answer_text is the model's last non-tool-call output, captured
    for single-agent mode. In two-stage mode it can be ignored.

    tool_executor can be a SimulatedArtifactExecutor (V5 cold-cache) or
    V6Executor (V6 cold-cache). Both implement execute(name, params) ->
    SimulatedToolResult. When None, falls back to direct index reads.

    policy_backend: "local" for HF model, "ollama" for Ollama API (supports multimodal).
    """
    messages = []
    tool_calls = []
    evidence_pieces = []
    final_answer_text = ""

    # No-tools mode: strip tool awareness entirely
    no_tools_mode = (max_turns == 0)
    if no_tools_mode:
        messages.append({"role": "system", "content": NO_TOOLS_SYSTEM})
        messages.append({"role": "user", "content": question})
        active_schemas = []  # no tools visible to model
    else:
        if single_agent:
            messages.append({"role": "system", "content": SINGLE_AGENT_SYSTEM})
        messages.append({"role": "user", "content": question})
        active_schemas = tool_schemas or TOOL_SCHEMAS

        # Hybrid mode: pre-load the full ASR transcript ONLY into the
        # evidence_pieces list (so the answerer sees it). The policy is
        # NOT shown the transcript via tool history — that turned out to
        # bias the policy into re-fetching transcript snippets instead of
        # exploring other modalities. To make the policy spend its budget
        # on multi-modal augmentation, we also strip get_transcript from
        # active_schemas.
        if prefill_transcript and index_data:
            layers = index_data.get("layers", {})
            asr_items = layers.get("asr", {}).get("items", [])
            if not asr_items:
                for name in layers:
                    if name.startswith("asr"):
                        asr_items = layers[name].get("items", [])
                        if asr_items:
                            break
            if asr_items:
                lines = []
                for it in asr_items:
                    ms = it.get("start_ms", 0)
                    m = ms // 60000
                    s = (ms // 1000) % 60
                    txt = (it.get("text") or "").strip()
                    if txt:
                        lines.append(f"[{m}:{s:02d}] {txt}")
                full_transcript = "\n".join(lines)
                if full_transcript:
                    if len(full_transcript) > 12000:
                        full_transcript = full_transcript[:12000] + "\n... [truncated]"
                    evidence_pieces.append({
                        "tool": "get_transcript",
                        "params": {"_prefill": True},
                        "result": full_transcript,
                        "execution_mode": "prefill",
                    })
                    # Strip transcript-fetching tool from policy's options so
                    # it focuses on OCR/visual/speakers
                    active_schemas = [
                        s for s in active_schemas
                        if (s.get("function", {}).get("name") if isinstance(s, dict) else "")
                        != "get_transcript"
                    ]

    def _generate(msgs, max_tokens=300):
        if policy_backend == "ollama":
            return policy_generate_ollama(
                msgs, active_schemas, policy_model_name,
                ollama_url or OLLAMA_URL, max_new_tokens=max_tokens,
            )
        elif policy_backend == "vllm":
            return policy_generate_vllm(
                msgs, active_schemas, policy_model_name,
                VLLM_URL,
                max_new_tokens=max_tokens,
            )
        else:
            resp = policy_generate(model, tokenizer, msgs, active_schemas, max_new_tokens=max_tokens)
            return resp.replace("<|im_end|>", "").strip()

    # No-tools mode: just generate one answer directly
    if no_tools_mode:
        final_answer_text = _generate(messages, max_tokens=500)
        return tool_calls, evidence_pieces, final_answer_text

    for turn in range(max_turns):
        response = _generate(messages)

        func_name, params = parse_tool_call(response)
        if func_name:
            try:
                if tool_executor:
                    tool_result = tool_executor.execute(func_name, params)
                    observation = tool_result.observation
                    trace_meta = tool_result.to_trace()
                else:
                    observation = execute_tool(func_name, params, index_data)
                    tool_result = None
                    trace_meta = {"execution_mode": "prebuilt_index"}
            except Exception as exc:
                # A malformed tool call must not abort a long eval run; show
                # the error to the policy so it can retry with valid params.
                observation = f"Error: tool call failed ({exc}). Check parameters and try again."
                tool_result = None
                trace_meta = {"execution_mode": "prebuilt_index", "tool_error": str(exc)}

            tool_trace = {
                "tool": func_name,
                "params": params,
            }
            tool_trace.update(trace_meta)
            tool_calls.append(tool_trace)
            evidence_pieces.append({
                "tool": func_name,
                "params": params,
                "result": observation[:500],
                **trace_meta,
            })

            # Add to conversation for next turn
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            think = think_match.group(1).strip() if think_match else ""

            messages.append({
                "role": "assistant",
                "content": f"<think>\n{think}\n</think>" if think else response,
                "tool_calls": [{"type": "function", "function": {"name": func_name, "arguments": params}}]
            })

            # Tool response message -- attach image if present (for multimodal)
            tool_msg = {"role": "tool", "content": observation[:800]}
            if tool_result and hasattr(tool_result, '_image_b64') and tool_result._image_b64:
                tool_msg["_v6_image_b64"] = tool_result._image_b64
            messages.append(tool_msg)
        else:
            # Model produced a final answer, not a tool call
            final_answer_text = response
            break

    # In single-agent mode, if the model used all turns on tool calls without
    # producing a final answer, give it one more generation with extra tokens.
    if single_agent and not final_answer_text:
        messages.append({"role": "user", "content": "You have used all your tool calls. Now provide your final answer as a JSON object with answer, evidence, and rationale."})
        final_answer_text = _generate(messages, max_tokens=500)

    return tool_calls, evidence_pieces, final_answer_text


def _extract_json_object(text):
    """Extract the outermost JSON object from text using balanced braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def parse_agent_answer(text, fmt):
    """Parse single-agent structured answer from model output."""
    text = re.sub(r'<think>.*?</think>', '', text or '', flags=re.DOTALL).strip()
    text = text.replace("<|im_end|>", "").strip()

    # Strip markdown fences if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*$', '', text).strip()

    # Try direct JSON parse first
    data = None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Balanced-brace extraction handles nested objects (evidence arrays)
        data = _extract_json_object(text)

    if data and isinstance(data, dict) and "answer" in data:
        answer = data.get("answer", "")
        if fmt == "multiple_choice":
            answer = extract_mc_prediction(str(answer))
        cited_evidence = data.get("evidence", [])
        rationale = data.get("rationale", "")
        return answer, cited_evidence, rationale

    # Fallback: treat as plain text answer
    answer = extract_mc_prediction(text) if fmt == "multiple_choice" else text
    return answer, [], ""


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=Path("data/video_indexes"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-adapter", type=Path, default=None,
                        help="LoRA adapter path. Omit to run base model without adapter.")
    parser.add_argument("--policy-base", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--policy-backend", choices=["local", "ollama", "vllm"], default="local",
                        help="Policy model backend. 'local' loads HF model on GPU. "
                             "'ollama' uses Ollama API. 'vllm' uses vLLM OpenAI-compatible API.")
    parser.add_argument("--policy-ollama-model", type=str, default="qwen2.5vl:7b",
                        help="Model name for --policy-backend ollama or vllm.")
    parser.add_argument("--answerer-model", type=str, default="Qwen/Qwen3.5-9B",
                        help="Answerer model name for HTTP backends. Ignored by --answerer-backend local-base.")
    parser.add_argument("--answerer-backend", choices=["env", "vllm", "ollama", "local-base"],
                        default="env",
                        help="Answer synthesis backend. local-base reuses the loaded policy base model with the LoRA adapter disabled.")
    parser.add_argument("--allow-answerer-errors", action="store_true",
                        help="Write answerer exceptions as predictions instead of failing fast. Intended only for debugging.")
    parser.add_argument("--agent-mode", choices=["two-stage", "single"],
                        default="two-stage",
                        help="two-stage: policy gathers evidence, separate answerer synthesizes. "
                             "single: one model does tool use + evidence selection + answer in one loop.")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--tool-execution-mode", choices=["prebuilt_index", "simulated_cold_cache"],
                        default="prebuilt_index",
                        help="prebuilt_index = warm-index archive QA over existing layers; simulated_cold_cache = explicit artifact/cache orchestration using the index as an immutable fixture.")
    parser.add_argument("--tool-schema-mode", choices=["auto", "legacy", "v5", "v6", "v7"], default="auto",
                        help="Tool schema exposed to the policy. auto uses legacy for prebuilt_index and V5 artifact tools for simulated_cold_cache. v6 uses grid navigation. v7 uses direct information-request tools.")
    parser.add_argument("--v6-grid-sampling", choices=["uniform", "shot", "end_biased"], default="uniform",
                        help="V6 grid sampling strategy.")
    parser.add_argument("--v7-execution-mode", choices=["warm_index", "cold_cache"], default="warm_index",
                        help="V7 execution mode. warm_index reads from prebuilt index. cold_cache tracks processed ranges.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Explicit run ID for artifact bundle naming. Defaults to output file stem.")
    parser.add_argument("--targeted-evidence-tools", action="store_true",
                        help="Expose only targeted evidence tools; hide the legacy timeline-summary tool.")
    parser.add_argument("--no-browse", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--artifact-registry", type=Path, default=None,
                        help="Artifact registry path for simulated_cold_cache. Defaults to OUTPUT.artifact_registry.json.")
    parser.add_argument("--cache-run-id", type=str, default=None,
                        help="Run id used to group disposable cache artifacts. Defaults to output stem.")
    parser.add_argument("--clear-artifact-cache", action="store_true",
                        help="Clear artifacts for cache-run-id before running eval.")
    parser.add_argument("--resume", action="store_true",
                        help="If output file already has rows, skip questions whose "
                             "question_id is already present and append to the file.")
    parser.add_argument("--blind-answerer", action="store_true",
                        help="Two-stage with options-blind answerer: the answerer "
                             "produces a free-text answer (does NOT see MC options); "
                             "a separate MC-mapper stage then maps the free-text "
                             "answer to A/B/C/D. Forces --agent-mode two-stage.")
    parser.add_argument("--prefill-transcript", action="store_true",
                        help="Inject the full ASR transcript as initial evidence "
                             "before the ReAct loop. Agent uses its tool budget "
                             "to augment with OCR/visual/speaker; transcript is "
                             "free. Floor = full-transcript baseline.")
    parser.add_argument("--oracle-evidence", action="store_true",
                        help="Matched-evidence ceiling: skip the policy loop "
                             "entirely and build answerer context directly from "
                             "the benchmark row's source_segment_times (ASR text "
                             "for those spans; plus stored/indexed visual "
                             "evidence for visual rows), then call the answerer.")
    parser.add_argument("--rag", action="store_true",
                        help="RAG baseline: no policy. Build context via keyword "
                             "retrieval over the index "
                             "(utils.ctx_retrieval.retrieve_context, k=8, "
                             "max_chars=3000) plus the top ASR segments, then "
                             "call the answerer.")
    args = parser.parse_args()

    if args.oracle_evidence and args.rag:
        parser.error("--oracle-evidence and --rag are mutually exclusive")
    context_mode = "oracle" if args.oracle_evidence else ("rag" if args.rag else None)
    if context_mode and args.answerer_backend == "local-base":
        parser.error(f"--{'oracle-evidence' if context_mode == 'oracle' else 'rag'} "
                     "does not load a policy model, so --answerer-backend local-base "
                     "is unavailable; use vllm or ollama.")

    with open(args.benchmark) as f:
        questions = [json.loads(l) for l in f]
    if args.max_questions:
        questions = questions[:args.max_questions]

    # Resume support: skip questions already in the output file.
    already_done_ids = set()
    if args.resume and args.output.exists():
        with args.output.open() as f_existing:
            for line in f_existing:
                if not line.strip():
                    continue
                try:
                    already_done_ids.add(json.loads(line).get("question_id"))
                except Exception:
                    pass
        if already_done_ids:
            n_before = len(questions)
            questions = [q for i, q in enumerate(questions)
                         if q.get("id", f"q-{i}") not in already_done_ids]
            print(f"[resume] skipping {n_before - len(questions)} already-completed "
                  f"questions; {len(questions)} remaining")

    if args.tool_schema_mode == "v5" and args.tool_execution_mode != "simulated_cold_cache":
        parser.error("--tool-schema-mode v5 requires --tool-execution-mode simulated_cold_cache")
    if args.tool_schema_mode == "v6" and args.tool_execution_mode == "simulated_cold_cache":
        parser.error("--tool-schema-mode v6 has its own cold-cache; do not combine with --tool-execution-mode simulated_cold_cache")
    if args.tool_schema_mode == "v7" and args.tool_execution_mode == "simulated_cold_cache":
        parser.error("--tool-schema-mode v7 manages its own execution mode via --v7-execution-mode")
    if args.tool_schema_mode == "legacy":
        tool_schemas = TOOL_SCHEMAS
        resolved_schema_mode = "legacy"
    elif args.tool_schema_mode == "v5":
        tool_schemas = V5_TOOL_SCHEMAS
        resolved_schema_mode = "v5"
    elif args.tool_schema_mode == "v6":
        tool_schemas = V6_TOOL_SCHEMAS
        resolved_schema_mode = "v6"
    elif args.tool_schema_mode == "v7":
        tool_schemas = V7_TOOL_SCHEMAS
        resolved_schema_mode = "v7"
    elif args.tool_execution_mode == "simulated_cold_cache":
        tool_schemas = V5_TOOL_SCHEMAS
        resolved_schema_mode = "v5"
    else:
        tool_schemas = TOOL_SCHEMAS
        resolved_schema_mode = "legacy"
    if args.targeted_evidence_tools or args.no_browse:
        print("NOTE: --targeted-evidence-tools/--no-browse is ignored; all tools are exposed for this run.")
        resolved_schema_mode = f"{resolved_schema_mode}+all_tools"

    indexes = {}
    for f in args.index_dir.glob("*.json"):
        indexes[f.stem] = json.load(open(f))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_registry = None
    cache_run_id = args.cache_run_id or args.output.stem
    artifact_registry_path = args.artifact_registry
    if args.tool_execution_mode == "simulated_cold_cache":
        if artifact_registry_path is None:
            artifact_registry_path = args.output.with_suffix(".artifact_registry.json")
        artifact_registry = ArtifactRegistry(artifact_registry_path)
        if args.clear_artifact_cache:
            removed = artifact_registry.clear(run_id=cache_run_id)
            print(f"Cleared {len(removed)} cached artifacts for run_id={cache_run_id}")

    # Load policy model (skipped in policy-free modes: oracle / rag)
    policy_model, policy_tokenizer = None, None
    if args.policy_backend == "local" and context_mode is None:
        policy_model, policy_tokenizer = load_policy_model(
            args.policy_base, str(args.policy_adapter) if args.policy_adapter else None)
    answerer_backend = ANSWERER_BACKEND if args.answerer_backend == "env" else args.answerer_backend

    # V6 multimodal: set render mode based on policy backend
    v6_render_mode = "text_sim"
    if args.policy_backend in ("ollama", "vllm") and resolved_schema_mode == "v6":
        v6_render_mode = "multimodal"

    if args.blind_answerer and args.agent_mode == "single":
        print("[blind-answerer] forcing --agent-mode two-stage", flush=True)
        args.agent_mode = "two-stage"
    single_agent = args.agent_mode == "single"
    if context_mode:
        label = "Oracle-Evidence (matched-evidence ceiling)" if context_mode == "oracle" else "RAG (keyword retrieval)"
        print(f"\n{label} Evaluation -- policy loop skipped")
        print(f"Answerer: {args.answerer_model} via {answerer_backend}")
    elif single_agent:
        print(f"\nSingle-Agent Evaluation")
        if args.policy_backend in ("ollama", "vllm"):
            print(f"Model: {args.policy_ollama_model} via {args.policy_backend}")
        else:
            print(f"Model: {args.policy_base} + {args.policy_adapter or 'no adapter'}")
    else:
        print(f"\nTwo-Stage Policy + Answerer Evaluation")
        print(f"Policy: {args.policy_base} + {args.policy_adapter}")
        if answerer_backend == "local-base":
            print(f"Answerer: {args.policy_base} via local-base (policy adapter disabled)")
        else:
            print(f"Answerer: {args.answerer_model} via {answerer_backend}")
    print(f"Questions: {len(questions)}")
    print(f"Max turns: {args.max_turns}")
    print(f"Tool execution mode: {args.tool_execution_mode}")
    print(f"Tool schema mode: {resolved_schema_mode}")
    if artifact_registry_path:
        print(f"Artifact registry: {artifact_registry_path}")
        print(f"Cache run id: {cache_run_id}")

    stats = {"total": 0, "no_index": 0, "total_tools": 0, "mc_total": 0, "mc_correct": 0}

    open_mode = "a" if (args.resume and already_done_ids) else "w"
    with open(args.output, open_mode) as fout:
        for i, q in enumerate(questions):
            video_id = q.get("video_id", "")
            fmt = (q.get("format") or "free_text").lower().replace("_", "")
            fmt = {"multiplechoice": "multiple_choice",
                   "retrievalset": "retrieval_set"}.get(fmt, "free_text")

            if fmt == "multiple_choice":
                expected = q.get("mc_correct", "")
                mc_options = q.get("mc_options", {})
            elif fmt == "retrieval_set":
                expected = q.get("answer", [])   # list of segment dicts
                mc_options = None
            else:
                expected = q.get("reference_answer", q.get("answer", ""))
                mc_options = None

            idx = indexes.get(video_id)
            if not idx:
                stats["no_index"] += 1
                continue

            # Stage 1: Policy gathers evidence (skipped in oracle/rag modes)
            tool_executor = None
            v6_executor = None
            v7_executor = None
            if context_mode:
                tool_calls, final_text = [], ""
                if context_mode == "oracle":
                    evidence = build_oracle_evidence(q, idx)
                else:
                    evidence = build_rag_evidence(q, idx)
            else:
                if resolved_schema_mode == "v7":
                    v7_executor = V7Executor(
                        index_data=idx,
                        video_id=video_id,
                        execution_mode=args.v7_execution_mode,
                    )
                    tool_executor = v7_executor
                elif resolved_schema_mode == "v6":
                    v6_executor = V6Executor(
                        index_data=idx,
                        video_id=video_id,
                        grid_sampling=args.v6_grid_sampling,
                        render_mode=v6_render_mode,
                    )
                    tool_executor = v6_executor
                elif artifact_registry is not None:
                    # Clear artifacts from previous question so each starts cold
                    artifact_registry.clear(run_id=cache_run_id, save=True)
                    tool_executor = SimulatedArtifactExecutor(
                        index_data=idx,
                        video_id=video_id,
                        registry=artifact_registry,
                        run_id=cache_run_id,
                    )
                # For single-agent mode, include MC options in the question
                question_text = q["question"]
                if single_agent and mc_options:
                    opts = "\n".join(f"  {k}. {v}" for k, v in sorted(mc_options.items()))
                    question_text = f"{question_text}\n\nOptions:\n{opts}"

                tool_calls, evidence, final_text = gather_evidence(
                    policy_model, policy_tokenizer,
                    question_text, idx, max_turns=args.max_turns,
                    tool_executor=tool_executor,
                    tool_schemas=tool_schemas,
                    single_agent=single_agent,
                    policy_backend=args.policy_backend,
                    policy_model_name=args.policy_ollama_model,
                    ollama_url=OLLAMA_URL,
                    prefill_transcript=args.prefill_transcript,
                )

            cited_evidence = None
            rationale = None

            if context_mode:
                # Policy-free modes: single answerer call over the built context.
                if evidence:
                    raw_answer = call_answerer(
                        q["question"], evidence,
                        mc_options=mc_options, model=args.answerer_model,
                        backend=answerer_backend,
                        ollama_url=OLLAMA_URL, vllm_url=VLLM_URL,
                        max_chars_per_item=(ORACLE_SPAN_CHAR_CAP
                                            if context_mode == "oracle"
                                            else RAG_MAX_CHARS_PER_ITEM),
                    )
                    if isinstance(raw_answer, str) and raw_answer.startswith("Error:") and not args.allow_answerer_errors:
                        raise RuntimeError(
                            "Answerer call failed; aborting eval rather than writing invalid predictions. "
                            f"Question {i + 1}/{len(questions)} ({q.get('id', f'q-{i}')}): {raw_answer}"
                        )
                else:
                    raw_answer = "insufficient evidence"
                predicted = raw_answer
                if fmt == "multiple_choice":
                    predicted = extract_mc_prediction(predicted)
            elif single_agent:
                # Single-agent: parse the model's own structured answer
                raw_answer = final_text or ""
                predicted, cited_evidence, rationale = parse_agent_answer(raw_answer, fmt)
            else:
                # Two-stage: call separate answerer.
                # In --blind-answerer mode, the answerer ALSO doesn't see MC
                # options; a third MC-mapper stage maps the free-text answer
                # to A/B/C/D after the fact.
                answerer_mc = None if args.blind_answerer else mc_options
                if evidence:
                    if answerer_backend == "local-base":
                        raw_answer = call_loaded_answerer(
                            q["question"], evidence,
                            mc_options=answerer_mc,
                            model=policy_model,
                            tokenizer=policy_tokenizer,
                            disable_adapter=True,
                        )
                    else:
                        raw_answer = call_answerer(
                            q["question"], evidence,
                            mc_options=answerer_mc, model=args.answerer_model,
                            backend=answerer_backend,
                            ollama_url=OLLAMA_URL, vllm_url=VLLM_URL,
                        )
                    if isinstance(raw_answer, str) and raw_answer.startswith("Error:") and not args.allow_answerer_errors:
                        raise RuntimeError(
                            "Answerer call failed; aborting eval rather than writing invalid predictions. "
                            f"Question {i + 1}/{len(questions)} ({q.get('id', f'q-{i}')}): {raw_answer}"
                        )
                else:
                    raw_answer = "insufficient evidence"

                predicted = raw_answer
                if fmt == "multiple_choice":
                    if args.blind_answerer and mc_options:
                        predicted = map_freetext_to_mc(
                            free_text=raw_answer,
                            question=q["question"],
                            mc_options=mc_options,
                            model=args.answerer_model,
                            backend=answerer_backend,
                            ollama_url=OLLAMA_URL, vllm_url=VLLM_URL,
                        )
                    else:
                        predicted = extract_mc_prediction(predicted)

            # Inline scoring preview: MC only. Free-text must be scored post-hoc.
            is_correct = None
            if fmt == "multiple_choice":
                stats["mc_total"] += 1
                is_correct = str(predicted).strip().upper()[:1] == str(expected).strip().upper()[:1]
                if is_correct:
                    stats["mc_correct"] += 1

            stats["total"] += 1
            stats["total_tools"] += len(tool_calls)

            output = {
                "question_id": q.get("id", f"q-{i}"),
                "video_id": video_id,
                "question": q["question"],
                "format": fmt,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "raw_answer": str(raw_answer)[:500],
                "agent_mode": args.agent_mode,
                "tool_calls": [tc["tool"] for tc in tool_calls],
                "tool_trace": tool_calls,
                "evidence": evidence,
                "num_tool_calls": len(tool_calls),
                "evidence_count": len(evidence),
                "reasoning_type": q.get("reasoning_type", ""),
                "modalities_required": q.get("modalities_required", []),
                "source_segment_times": q.get("source_segment_times", []),
                "inline_correct": is_correct,
                "needs_posthoc_scoring": fmt != "multiple_choice",
            }
            if context_mode:
                output["context_mode"] = context_mode
            if cited_evidence is not None:
                output["cited_evidence"] = cited_evidence
                output["rationale"] = rationale
            if v7_executor is not None:
                output["v7_metrics"] = v7_executor.get_metrics()
            elif v6_executor is not None:
                output["v6_metrics"] = v6_executor.get_metrics()
            fout.write(json.dumps(output) + "\n")
            fout.flush()

            if (i + 1) % 10 == 0:
                n = max(1, stats["total"])
                mc_n = max(1, stats["mc_total"])
                print(f"  [{i+1}/{len(questions)}] "
                      f"MC preview: {stats['mc_correct']}/{stats['mc_total']} ({stats['mc_correct']/mc_n*100:.0f}%), "
                      f"avg tools: {stats['total_tools']/n:.1f}", flush=True)

            time.sleep(0.3)

    n = max(1, stats["total"])
    print(f"\nDone.")
    mc_n = max(1, stats["mc_total"])
    print(f"MC preview only: {stats['mc_correct']}/{stats['mc_total']} ({stats['mc_correct']/mc_n*100:.1f}%)")
    print(f"Free-text requires post-hoc scoring with eval/score_predictions.py")
    print(f"Avg tools: {stats['total_tools']/n:.1f}")
    print(f"Output: {args.output}")
    if artifact_registry is not None:
        print(f"Cached artifacts: {len(artifact_registry.records_for_run(cache_run_id))}")

    # Write artifact bundle for V7 runs
    if resolved_schema_mode == "v7":
        _write_artifact_bundle(args, stats, resolved_schema_mode)


def _write_artifact_bundle(args, stats, resolved_schema_mode):
    """Write .meta.json and .summary.json alongside the .jsonl output."""
    import platform
    import subprocess

    output_stem = args.output.with_suffix("")
    run_id = args.run_id or output_stem.name

    # .meta.json
    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass

    meta = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": git_sha,
        "benchmark": str(args.benchmark),
        "policy_base": args.policy_base,
        "policy_adapter": str(args.policy_adapter) if args.policy_adapter else None,
        "policy_backend": args.policy_backend,
        "policy_model_name": getattr(args, "policy_ollama_model", None),
        "agent_mode": args.agent_mode,
        "tool_schema_mode": resolved_schema_mode,
        "execution_mode": getattr(args, "v7_execution_mode", "warm_index"),
        "max_turns": args.max_turns,
        "max_questions": args.max_questions,
        "host": platform.node(),
        "command": " ".join(sys.argv),
    }

    meta_path = output_stem.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta: {meta_path}")

    # .summary.json
    results = []
    try:
        with open(args.output) as f:
            results = [json.loads(l) for l in f if l.strip()]
    except Exception:
        pass

    mc_rows = [r for r in results if r.get("format") == "multiple_choice"]
    mc_correct = sum(1 for r in mc_rows if r.get("inline_correct"))
    mc_total = len(mc_rows)
    ft_rows = [r for r in results if r.get("format") != "multiple_choice"]

    # Grounding: compute evidence recall for MC rows
    from eval.score_grounding import evidence_recall, citation_validity
    grounded_correct = 0
    citation_valid_count = 0
    total_tools = 0
    total_cost = 0.0
    total_evidence = 0
    zero_tool_count = 0
    error_examples = []

    for r in results:
        n_tools = r.get("num_tool_calls", 0)
        total_tools += n_tools
        if n_tools == 0:
            zero_tool_count += 1

        metrics = r.get("v7_metrics") or r.get("v6_metrics") or {}
        total_cost += metrics.get("total_tool_cost", 0)
        total_evidence += metrics.get("evidence_items", 0)

    for r in mc_rows:
        expected_text = r.get("expected_answer", "")
        recall = evidence_recall(r, expected_text)
        is_correct = r.get("inline_correct", False)

        if is_correct and recall >= 0.5:
            grounded_correct += 1

        valid, total, cit_score = citation_validity(r)
        if total > 0 and cit_score >= 0.5:
            citation_valid_count += 1

        if not is_correct and len(error_examples) < 10:
            error_examples.append({
                "question_id": r.get("question_id"),
                "question": r.get("question", "")[:100],
                "expected": str(r.get("expected_answer", "")),
                "predicted": str(r.get("predicted_answer", "")),
                "num_tools": r.get("num_tool_calls", 0),
                "evidence_recall": round(recall, 3),
            })

    n = max(1, len(results))
    summary = {
        "run_id": run_id,
        "total_rows": len(results),
        "mc_rows": mc_total,
        "free_text_rows": len(ft_rows),
        "mc_correct": mc_correct,
        "mc_accuracy": round(mc_correct / max(1, mc_total), 4),
        "grounded_correct": grounded_correct,
        "grounded_accuracy": round(grounded_correct / max(1, mc_total), 4),
        "citation_valid_count": citation_valid_count,
        "average_tools": round(total_tools / n, 2),
        "average_tool_cost": round(total_cost / n, 4),
        "average_evidence_items": round(total_evidence / n, 2),
        "zero_tool_rows": zero_tool_count,
        "error_examples": error_examples,
    }

    summary_path = output_stem.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
