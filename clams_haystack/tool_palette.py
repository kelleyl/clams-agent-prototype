"""Manual-invocation tool palette for the Chainlit Haystack UI.

Each tool defines a parameter schema that the form runner walks through
sequentially with the appropriate Chainlit widget. Param types whose names
start with ``mmif_`` (or ``video``) pull their option lists from the live
``MmifInventory`` so the user can target, e.g., the SWT chyron timeframes for
a specific video.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParam:
    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    multivalued: bool = False
    choices: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)
    placeholder: str = ""

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class ToolDef:
    id: str
    label: str
    description: str
    category: str
    params: list[ToolParam]
    runner: str
    notes: str = ""

    def get_param(self, name: str) -> ToolParam | None:
        for p in self.params:
            if p.name == name:
                return p
        return None


TOOLS: list[ToolDef] = [
    ToolDef(
        id="swt_detection",
        label="SWT — Scenes With Text",
        description=(
            "Identify video scenes that contain on-screen text (chyrons, slates, "
            "credits, lower-thirds). Produces TimePoint + TimeFrame views."
        ),
        category="detection",
        runner="run_swt_detection",
        params=[
            ToolParam(
                name="video_id",
                type="video",
                description="Source video to analyze",
            ),
            ToolParam(
                name="preset",
                type="choice",
                description="Stitcher / classifier preset",
                choices=["balanced", "high-recall", "high-precision", "modernized-default"],
                default="balanced",
                required=False,
            ),
            ToolParam(
                name="map_to_finegrained_labels",
                type="bool",
                description="Use fine-grained labels (Chyron/Slate/Credits/...) instead of coarse",
                default=True,
                required=False,
            ),
        ],
    ),
    ToolDef(
        id="smolvlm2_captioner",
        label="SmolVLM2 — OCR Captioner",
        description=(
            "Run SmolVLM2 OCR/caption on text-bearing TimeFrames from an existing "
            "SWT view. Targets specific labels (e.g. Chyron, Slate, Credits)."
        ),
        category="ocr",
        runner="run_smolvlm2_captioner",
        params=[
            ToolParam(
                name="video_id",
                type="video",
                description="Source video",
            ),
            ToolParam(
                name="source_view",
                type="mmif_view",
                description="SWT view to use as input (TimeFrame view)",
                depends_on=["video_id"],
                filters={"contains": "TimeFrame", "app_contains": "swt"},
            ),
            ToolParam(
                name="target_labels",
                type="tf_labels",
                description="TimeFrame labels to caption",
                multivalued=True,
                depends_on=["video_id", "source_view"],
            ),
            ToolParam(
                name="prompt_preset",
                type="choice",
                description="Prompt template",
                choices=["chyron-default", "slate-extraction", "credits-default", "generic-ocr"],
                default="chyron-default",
                required=False,
            ),
            ToolParam(
                name="frames_per_clip",
                type="number",
                description="Frames sampled per TimeFrame",
                default=3,
                required=False,
            ),
            ToolParam(
                name="preview_targets",
                type="tf_select",
                description="(Optional) preview the exact TimeFrames that will be captioned",
                multivalued=True,
                required=False,
                depends_on=["video_id", "source_view", "target_labels"],
            ),
        ],
    ),
    ToolDef(
        id="vlm_ocr_kie",
        label="VLM-OCR — Structured KIE",
        description=(
            "DSPy-optimized VLM key-information extraction (titles, dates, "
            "producers) from slates and credits. Best on short, high-signal frames."
        ),
        category="ocr",
        runner="run_vlm_ocr",
        params=[
            ToolParam(name="video_id", type="video", description="Source video"),
            ToolParam(
                name="source_view",
                type="mmif_view",
                description="View providing TimeFrames to extract from",
                depends_on=["video_id"],
                filters={"contains": "TimeFrame"},
            ),
            ToolParam(
                name="target_labels",
                type="tf_labels",
                description="Which labels to extract from (typically Slate, Credits)",
                multivalued=True,
                depends_on=["video_id", "source_view"],
            ),
            ToolParam(
                name="model",
                type="choice",
                description="VLM backend",
                choices=["qwen3.5-vl-7b", "qwen3.5-vl-32b", "smolvlm2-2.2b", "deepseek-ocr"],
                default="qwen3.5-vl-7b",
                required=False,
            ),
            ToolParam(
                name="schema",
                type="choice",
                description="KIE output schema",
                choices=["slate-default", "credits-default", "lower-third"],
                default="slate-default",
                required=False,
            ),
        ],
    ),
    ToolDef(
        id="asr_whisper",
        label="ASR — Whisper",
        description="Transcribe speech to text with timestamps and per-segment alignment.",
        category="asr",
        runner="run_asr",
        params=[
            ToolParam(name="video_id", type="video", description="Source video"),
            ToolParam(
                name="model",
                type="choice",
                description="Whisper model size",
                choices=["tiny", "base", "small", "medium", "large-v3"],
                default="large-v3",
                required=False,
            ),
            ToolParam(
                name="language",
                type="string",
                description="Force language (blank = auto-detect)",
                required=False,
                default="",
                placeholder="en",
            ),
        ],
    ),
    ToolDef(
        id="scene_summary",
        label="Scene Summaries (VLM)",
        description="Sample frames per scene and generate visual descriptions with broader context.",
        category="captioning",
        runner="run_scene_summary",
        params=[
            ToolParam(name="video_id", type="video", description="Source video"),
            ToolParam(
                name="granularity",
                type="choice",
                description="Segmentation granularity",
                choices=["scene", "shot", "fixed-30s", "fixed-60s"],
                default="scene",
                required=False,
            ),
            ToolParam(
                name="model",
                type="choice",
                description="VLM backend",
                choices=["qwen3.5-vl-7b", "qwen3.5-vl-32b", "smolvlm2-2.2b"],
                default="qwen3.5-vl-7b",
                required=False,
            ),
        ],
    ),
    ToolDef(
        id="credits_ocr",
        label="Credits OCR (Qwen3-VL pipeline)",
        description="Structured extraction of cast/crew/producer/copyright from credit rolls.",
        category="ocr",
        runner="run_credits_ocr",
        params=[
            ToolParam(name="video_id", type="video", description="Source video"),
            ToolParam(
                name="source_view",
                type="mmif_view",
                description="View containing Credits TimeFrames",
                depends_on=["video_id"],
                filters={"contains": "TimeFrame", "label": "Credits"},
            ),
            ToolParam(
                name="role_filters",
                type="choice",
                description="Restrict extraction to roles",
                multivalued=True,
                choices=["all", "cast", "crew", "producers", "writers", "rights"],
                default="all",
                required=False,
            ),
        ],
    ),
]


TOOLS_BY_ID: dict[str, ToolDef] = {t.id: t for t in TOOLS}


def get_tool(tool_id: str) -> ToolDef | None:
    return TOOLS_BY_ID.get(tool_id)


def categories() -> dict[str, list[ToolDef]]:
    out: dict[str, list[ToolDef]] = {}
    for t in TOOLS:
        out.setdefault(t.category, []).append(t)
    return out


def render_form_card(tool: ToolDef, values: dict[str, Any], focused: str | None = None) -> str:
    lines = [
        f"### Tool form · `{tool.id}` — {tool.label}",
        f"_{tool.description}_",
        "",
        "| | Field | Value |",
        "|---|---|---|",
    ]
    for p in tool.params:
        if p.name in values and values[p.name] not in (None, "", []):
            mark = "✅"
        elif p.name == focused:
            mark = "✏️"
        elif not p.required:
            mark = "▫️"
        else:
            mark = "☐"
        v = values.get(p.name, "")
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) or "_(none)_"
        v_str = str(v) if v not in (None, "") else "_(unset)_"
        v_str = v_str.replace("\n", " ")
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        req = "" if p.required else " _(optional)_"
        lines.append(f"| {mark} | **{p.display_name}**{req} — {p.description} | {v_str} |")
    return "\n".join(lines)
