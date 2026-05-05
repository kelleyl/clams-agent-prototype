# CLAMS Haystack Chainlit App

This package contains the Haystack-based cataloging and QA app for thesis
dataset videos. It is the raw-video/cold-cache-facing prototype: users can
register new videos, browse existing MMIF outputs, ask questions about indexed
content, and request CLAMS tools through discoverable service endpoints.

## What Runs

- `chainlit_app.py` is the web chat UI.
- `agent.py` wraps Haystack RAG and searchable tool-agent paths.
- `mmif_inventory.py` discovers thesis MMIF outputs across configured roots.
- `tool_palette.py` defines human-invoked tools and their parameter forms.
- `service_registry.py` maps tool IDs to HTTP CLAMS services.
- `repository.py` stores catalog records, documents, and tool requests in JSONL.
- `cli.py` provides smoke-test commands for ingest, search, RAG, and metadata.

The app uses `Qwen/Qwen3.5-9B` by default through a vLLM OpenAI-compatible
endpoint. On Aristotle that endpoint is expected at `http://localhost:8890/v1`.

## Data Roots

The MMIF inventory reads from `CLAMS_HAYSTACK_MMIF_ROOTS`. The Aristotle launcher
defaults to:

```text
data/chicago:data/enriched:data/combined_outputs:data/pipeline_outputs:data/scene_summaries_smoke:data/qwen_outputs
```

When several MMIFs exist for one source video, the inventory prefers final
`*_enriched.mmif` outputs, then complete combined outputs, then intermediate
step outputs. It does not fabricate mock videos; if `/videos` is empty, the
configured roots do not contain readable MMIF files.

## Service Registry

Tool endpoints are configured in:

```text
data/tool_services.json
```

The registry currently covers Aristotle images for:

- SWT scene/text detection
- Whisper ASR
- SmolVLM2 captioning and scene description
- Qwen OCR/KIE and credits OCR
- PySceneDetect
- spaCy
- TransNet

Use `/services` in Chainlit to see which endpoints are reachable. By default,
tool forms record a request but do not POST to services. Set
`CLAMS_HAYSTACK_EXECUTE_SERVICES=1` to enable live endpoint calls.

## Aristotle Runbook

Start or restart the Chainlit app:

```bash
cd ~/clams_apps/clams-agent-prototype
scripts/run_haystack_chainlit_aristotle.sh
```

Start the Qwen 3.5 vLLM server if it is not already running:

```bash
cd ~/clams_apps/clams-agent-prototype
scripts/run_qwen35_vllm_aristotle.sh
```

Start one or more CLAMS services:

```bash
cd ~/clams_apps/clams-agent-prototype
scripts/run_clams_tool_services_aristotle.sh swt_detection
scripts/run_clams_tool_services_aristotle.sh swt_detection smolvlm2_captioner
```

The service script defaults to GPU 2 to avoid the Qwen vLLM engines on GPUs 0
and 1. Override with `GPU_DEVICE=3` or another available GPU when needed.

Access the app from a local machine:

```bash
ssh -L 8784:localhost:8784 aristotle
```

Then open `http://localhost:8784`.

## Important Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLAMS_HAYSTACK_MODEL` | `Qwen/Qwen3.5-9B` | Agent model name |
| `CLAMS_HAYSTACK_VLLM_URL` | `http://localhost:8890/v1` | vLLM OpenAI-compatible base URL |
| `CLAMS_HAYSTACK_CATALOG_ROOT` | `data/haystack_catalog` | JSONL catalog/document store |
| `CLAMS_HAYSTACK_MMIF_ROOTS` | launcher-defined | Colon-separated MMIF roots |
| `CLAMS_HAYSTACK_TOOL_SERVICES` | `data/tool_services.json` | Tool endpoint registry |
| `CLAMS_HAYSTACK_EXECUTE_SERVICES` | `0` | Whether confirmed tool forms POST to services |
| `CLAMS_HAYSTACK_COMPARE_MODELS` | current model | Comma-separated models for `/compare` |

## Chainlit Commands

- `/status` shows model, vLLM URL, catalog root, MMIF roots, and service registry.
- `/videos` lists discovered MMIF-backed videos.
- `/video <video_id>` sets the active video.
- `/tools` opens the manual tool palette.
- `/services` shows endpoint health.
- `/overview` shows catalog state for the active video.
- `/search <query>` searches catalog/content documents.
- `/compare <prompt>` runs the same prompt against configured comparison models.

## CLI Smoke Commands

```bash
python -m clams_haystack.cli ingest /path/to/video.mp4
python -m clams_haystack.cli search "guest names" --video-id my_video
python -m clams_haystack.cli ask "What is this video about?" --video-id my_video
python -m clams_haystack.cli agent "What metadata is missing for my_video?"
```

## Operational Notes

- The app is intentionally conservative about writes. Metadata changes require
  explicit write-tool enablement, and expensive CLAMS services require
  `CLAMS_HAYSTACK_EXECUTE_SERVICES=1`.
- Service calls POST a full MMIF JSON document to the configured endpoint with
  query-string parameters expected by CLAMS Restifier apps.
- Successful service outputs are saved under
  `data/haystack_catalog/tool_outputs/`.
- The current repository store is JSONL for inspectability. Replace it with a
  persistent Haystack document store once the document schema stabilizes.

## Troubleshooting

- Empty `/videos`: check `CLAMS_HAYSTACK_MMIF_ROOTS` and file permissions.
- `/services` shows `down`: start the matching container or update
  `data/tool_services.json`.
- Agent errors against vLLM: verify `curl http://127.0.0.1:8890/v1/models`.
- Tool form has no matching views: inspect the selected MMIF with `/videos` and
  choose a final/enriched MMIF root before intermediate step roots.
