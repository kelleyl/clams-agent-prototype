"""Prompt configurations for (model, task_mode) pairs.

Each CLAMS VLM tool call is parameterized by:
  - tool: which capability (caption_frame, extract_text)
  - model: which VLM (smolvlm, qwen-8b, qwen-30b)
  - task_mode: what information to extract

The prompt config maps (model, task_mode) -> prompt template that gets sent
to the actual CLAMS app. The agent never sees these prompts -- it just picks
the triple and gets back standardized output.

NOTE: During GRPO training, these prompts are NOT used. The simulated
environment in run_grpo_env.py assembles output from pre-computed index
layers, using task_mode to decide which layers to pull from and model to
scale output detail. These prompt templates are the reference for real
CLAMS app integration, where (model, task_mode) determines the actual
prompt sent to the VLM container.
"""

# ============================================================
# Task mode definitions
# ============================================================

TASK_MODES = ["general_scene", "text_focus", "person_identity", "action_event"]

TASK_MODE_DESCRIPTIONS = {
    "general_scene": "Broad visual read of the frame: scene description, salient people/objects, any obvious text.",
    "text_focus": "OCR-like inspection: exact visible text, layout/placement, confidence if partial.",
    "person_identity": "Who is visible: person description, distinguishing attributes, any name/title shown.",
    "action_event": "What is happening: main action, participants, setting, event interpretation.",
}

# ============================================================
# Prompt templates per (model, task_mode)
#
# These are the actual prompts sent to VLMs via CLAMS apps.
# {frame_context} is replaced with any available metadata
# (timestamp, scene label, etc.)
# ============================================================

CAPTION_PROMPTS = {
    # --- SmolVLM2 prompts (concise, single-sentence oriented) ---
    ("smolvlm", "general_scene"): (
        "Describe this frame from a broadcast video in 1-2 sentences. "
        "Note the setting, any people, and any visible text."
    ),
    ("smolvlm", "text_focus"): (
        "Read all visible text in this broadcast video frame. "
        "For each piece of text, state what it says and where it appears "
        "(chyron, headline, sign, graphic, credits). Be exact."
    ),
    ("smolvlm", "person_identity"): (
        "Describe any people visible in this broadcast video frame. "
        "Note their appearance, any name/title text shown on screen, "
        "and whether they appear to be a host, guest, or bystander."
    ),
    ("smolvlm", "action_event"): (
        "Describe what is happening in this broadcast video frame. "
        "Focus on actions and events, not just static description. "
        "Who is doing what?"
    ),

    # --- Qwen3.5-0.8B prompts (concise, resource-efficient) ---
    ("qwen-small", "general_scene"): (
        "Describe this broadcast video frame briefly: setting, people, any visible text."
    ),
    ("qwen-small", "text_focus"): (
        "Read all visible text in this broadcast video frame exactly as written. "
        "Note what type each text is (chyron, headline, sign, credits)."
    ),
    ("qwen-small", "person_identity"): (
        "Describe any people in this broadcast video frame. "
        "Note appearance, any name/title text on screen, and their role."
    ),
    ("qwen-small", "action_event"): (
        "What is happening in this broadcast video frame? "
        "Focus on actions and events, not just static description."
    ),

    # --- Qwen3.5-9B prompts (moderate detail) ---
    ("qwen-8b", "general_scene"): (
        "You are analyzing a frame from an archival broadcast video. "
        "Provide a concise description covering:\n"
        "- Scene setting and context\n"
        "- People present and their apparent roles\n"
        "- Any visible text (chyrons, graphics, signs)\n"
        "- Notable objects or visual elements"
    ),
    ("qwen-8b", "text_focus"): (
        "You are analyzing a frame from an archival broadcast video. "
        "Your task is to read ALL visible text exactly as written.\n\n"
        "For each text element, report:\n"
        "- Exact text content\n"
        "- Type: chyron / headline / sign / graphic / credits / other\n"
        "- Position: lower-third / upper / center / overlay\n"
        "- Legibility: clear / partial / degraded\n\n"
        "Do not paraphrase. Transcribe exactly what you see."
    ),
    ("qwen-8b", "person_identity"): (
        "You are analyzing a frame from an archival broadcast video. "
        "Focus on identifying people in the frame.\n\n"
        "For each person visible, report:\n"
        "- Physical description (age range, hair, clothing)\n"
        "- Any on-screen name/title text\n"
        "- Apparent role (anchor, reporter, guest, interviewee, bystander)\n"
        "- Whether their identity can be determined from visual cues alone"
    ),
    ("qwen-8b", "action_event"): (
        "You are analyzing a frame from an archival broadcast video. "
        "Describe the action or event taking place.\n\n"
        "Report:\n"
        "- Main action or event\n"
        "- Who is involved\n"
        "- Setting and context\n"
        "- Whether this appears to be live footage, b-roll, or studio\n\n"
        "Focus on what is happening, not just what is visible."
    ),

    # --- Qwen3.5-27B prompts (detailed, structured) ---
    ("qwen-30b", "general_scene"): (
        "You are an expert analyst examining a frame from an archival "
        "broadcast video. Provide a thorough description covering:\n\n"
        "1. SCENE: Setting, studio vs. field, time period cues\n"
        "2. PEOPLE: Everyone visible, their roles, interactions\n"
        "3. TEXT: All on-screen text, verbatim\n"
        "4. OBJECTS: Notable items, graphics, props\n"
        "5. CONTEXT: What type of segment this likely is "
        "(news report, interview, commercial, documentary)"
    ),
    ("qwen-30b", "text_focus"): (
        "You are an expert OCR analyst examining a frame from an archival "
        "broadcast video. This footage may be degraded or low-resolution.\n\n"
        "Transcribe ALL visible text with maximum accuracy:\n"
        "- Exact text content (preserve capitalization, punctuation)\n"
        "- Text type: chyron / lower-third / headline / crawl / "
        "graphic / slate / credits / sign / handwritten\n"
        "- Position on screen\n"
        "- Legibility: clear / partial / degraded / occluded\n"
        "- For partial text, indicate missing portions with [...]\n\n"
        "Pay special attention to names, dates, and titles. "
        "Do not guess illegible text -- mark it as unclear."
    ),
    ("qwen-30b", "person_identity"): (
        "You are an expert analyst examining a frame from an archival "
        "broadcast video. Focus on person identification.\n\n"
        "For each person visible:\n"
        "- Detailed physical description\n"
        "- Clothing, accessories, distinguishing features\n"
        "- Any on-screen identification (name chyron, title card)\n"
        "- Apparent role and relationship to other people in frame\n"
        "- Approximate age range and demographic presentation\n"
        "- Body language and expression\n\n"
        "Note: Only state what is visually evident. Do not speculate "
        "on identity beyond what on-screen cues provide."
    ),
    ("qwen-30b", "action_event"): (
        "You are an expert analyst examining a frame from an archival "
        "broadcast video. Describe the action or event.\n\n"
        "Report:\n"
        "- Primary action or event taking place\n"
        "- All participants and their roles in the action\n"
        "- Setting and environment\n"
        "- Segment type: live coverage / b-roll / interview / "
        "studio discussion / ceremony / demonstration\n"
        "- Temporal cues: is this a single moment or ongoing action?\n"
        "- Any causal or narrative context visible in the frame\n\n"
        "Focus on interpretation grounded in visual evidence."
    ),
}

# ============================================================
# OCR-specific prompts (for extract_text tool)
# These are simpler since extract_text is already text-focused.
# ============================================================

OCR_PROMPTS = {
    ("qwen-small", "default"): (
        "Read all visible text in this broadcast video frame exactly as written."
    ),
    ("qwen-8b", "default"): (
        "Read all text visible in this image from a broadcast video frame. "
        "Return the exact text content, noting the type "
        "(chyron, slate, credits, sign, graphic)."
    ),
    ("qwen-30b", "default"): (
        "You are an expert OCR system analyzing a frame from archival "
        "broadcast video. This footage may be degraded.\n\n"
        "Transcribe ALL visible text with exact spelling, capitalization, "
        "and punctuation. For each text element, note:\n"
        "- Content\n"
        "- Type (chyron, slate, credits, sign, graphic)\n"
        "- Legibility (clear/partial/degraded)\n"
        "Mark illegible portions with [...]."
    ),
}


# ============================================================
# Output formatting per task_mode
#
# These functions take raw VLM/index output and normalize it
# into a consistent format the answerer can consume, regardless
# of which model produced it.
# ============================================================

def format_general_scene(raw_output):
    """Normalize general_scene output."""
    return raw_output


def format_text_focus(raw_output):
    """Normalize text_focus output."""
    if not raw_output:
        return "No visible text detected."
    return raw_output


def format_person_identity(raw_output):
    """Normalize person_identity output."""
    if not raw_output:
        return "No people clearly visible in frame."
    return raw_output


def format_action_event(raw_output):
    """Normalize action_event output."""
    if not raw_output:
        return "Unable to determine action from this frame."
    return raw_output


OUTPUT_FORMATTERS = {
    "general_scene": format_general_scene,
    "text_focus": format_text_focus,
    "person_identity": format_person_identity,
    "action_event": format_action_event,
}


def get_caption_prompt(model, task_mode):
    """Get the prompt template for a (model, task_mode) pair."""
    prompt = CAPTION_PROMPTS.get((model, task_mode))
    if prompt is None:
        # Fall back to general_scene for unknown modes
        prompt = CAPTION_PROMPTS.get((model, "general_scene"), "Describe this image.")
    return prompt


def get_ocr_prompt(model):
    """Get the OCR prompt for a model."""
    return OCR_PROMPTS.get((model, "default"), "Read all visible text in this image.")
