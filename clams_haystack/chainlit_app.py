"""Chainlit frontend for the Haystack video catalog agent."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch, TextInput

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clams_haystack.agent import HaystackVideoAgent
from clams_haystack.mmif_inventory import (
    MmifInventory,
    TimeFrameInfo,
    ViewInfo,
    normalise_roots,
)
from clams_haystack.repository import CatalogRepository
from clams_haystack.service_registry import ServiceRegistry, service_config_path
from clams_haystack.tool_palette import (
    TOOLS,
    ToolDef,
    ToolParam,
    categories,
    get_tool,
    render_form_card,
)


DEFAULT_MODEL = os.environ.get("CLAMS_HAYSTACK_MODEL", "Qwen/Qwen3.5-9B")
DEFAULT_VLLM_URL = os.environ.get("CLAMS_HAYSTACK_VLLM_URL", "http://localhost:8890/v1")
DEFAULT_CATALOG_ROOT = os.environ.get("CLAMS_HAYSTACK_CATALOG_ROOT", "data/haystack_catalog")
DEFAULT_MMIF_ROOTS = os.environ.get(
    "CLAMS_HAYSTACK_MMIF_ROOTS",
    os.pathsep.join(str(p) for p in MmifInventory.default_roots()),
)
DEFAULT_TOOL_SERVICES = os.environ.get(
    "CLAMS_HAYSTACK_TOOL_SERVICES",
    str(service_config_path()),
)


@cl.set_starters
async def set_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="🛠 Open tool palette",
            message="/tools",
        ),
        cl.Starter(
            label="📋 List videos",
            message="/videos",
        ),
        cl.Starter(
            label="ℹ Status",
            message="/status",
        ),
        cl.Starter(
            label="Services",
            message="/services",
        ),
    ]


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await send_settings()
    configure_session(settings)
    await cl.Message(content=current_status()).send()
    await cl.Message(
        content=(
            "Type `/tools` to open the manual tool palette, or chat normally to "
            "drive the agent."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    configure_session(settings)
    await cl.Message(content=current_status()).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent = get_agent()
    uploaded = await register_uploaded_videos(agent, message)
    if uploaded:
        await cl.Message(content="\n".join(uploaded)).send()
        return

    content = message.content.strip()
    if not content:
        return

    try:
        if content.startswith("/"):
            await handle_command(agent, content)
            return

        video_id = cl.user_session.get("video_id")
        prompt = content
        if video_id:
            prompt = f"Active video_id: {video_id}\nUser request: {content}"

        searchable = bool(cl.user_session.get("searchable_tools", True))
        allow_writes = bool(cl.user_session.get("allow_write_tools", False))
        async with cl.Step(name="Haystack searchable tool agent") as step:
            step.input = prompt
            answer = await asyncio.to_thread(
                agent.run_tool_agent,
                prompt,
                searchable,
                allow_writes,
            )
            step.output = answer
        await cl.Message(content=answer).send()
    except Exception as exc:
        await cl.Message(content=f"Error: {exc}").send()


async def send_settings() -> dict[str, Any]:
    return await cl.ChatSettings(
        [
            TextInput(
                id="CatalogRoot",
                label="Catalog root",
                initial=DEFAULT_CATALOG_ROOT,
            ),
            TextInput(
                id="MMIFRoots",
                label="MMIF roots",
                initial=DEFAULT_MMIF_ROOTS,
            ),
            TextInput(
                id="ToolServices",
                label="Tool service registry",
                initial=DEFAULT_TOOL_SERVICES,
            ),
            TextInput(
                id="VideoID",
                label="Active video ID",
                initial=os.environ.get("CLAMS_HAYSTACK_VIDEO_ID", ""),
            ),
            TextInput(
                id="VLLMUrl",
                label="vLLM URL",
                initial=DEFAULT_VLLM_URL,
            ),
            Select(
                id="Model",
                label="Agent model",
                values=available_models(),
                initial_index=0,
            ),
            Switch(
                id="SearchableTools",
                label="Searchable tools",
                initial=True,
            ),
            Switch(
                id="AllowWriteTools",
                label="Allow write tools",
                initial=False,
            ),
            Slider(
                id="TopK",
                label="Retrieval top-k",
                initial=8,
                min=1,
                max=20,
                step=1,
            ),
        ]
    ).send()


def configure_session(settings: dict[str, Any]) -> None:
    catalog_root = settings.get("CatalogRoot") or DEFAULT_CATALOG_ROOT
    mmif_roots = settings.get("MMIFRoots") or DEFAULT_MMIF_ROOTS
    tool_services = settings.get("ToolServices") or DEFAULT_TOOL_SERVICES
    model = settings.get("Model") or DEFAULT_MODEL
    vllm_url = settings.get("VLLMUrl") or DEFAULT_VLLM_URL
    top_k = int(settings.get("TopK") or 8)

    repository = CatalogRepository(catalog_root)
    agent = HaystackVideoAgent(
        repository=repository,
        provider="vllm",
        model=model,
        api_base_url=vllm_url,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        top_k=top_k,
    )
    cl.user_session.set("agent", agent)
    cl.user_session.set("catalog_root", catalog_root)
    cl.user_session.set("mmif_roots", mmif_roots)
    cl.user_session.set("tool_services", tool_services)
    cl.user_session.set("mmif_inventory", None)
    cl.user_session.set("service_registry", None)
    cl.user_session.set("model", model)
    cl.user_session.set("vllm_url", vllm_url)
    cl.user_session.set("searchable_tools", bool(settings.get("SearchableTools", True)))
    cl.user_session.set("allow_write_tools", bool(settings.get("AllowWriteTools", False)))
    video_id = (settings.get("VideoID") or "").strip()
    cl.user_session.set("video_id", video_id or None)


def get_agent() -> HaystackVideoAgent:
    agent = cl.user_session.get("agent")
    if agent is None:
        configure_session({})
        agent = cl.user_session.get("agent")
    return agent


async def register_uploaded_videos(
    agent: HaystackVideoAgent,
    message: cl.Message,
) -> list[str]:
    responses = []
    for element in getattr(message, "elements", []) or []:
        path = getattr(element, "path", None)
        name = getattr(element, "name", None) or (Path(path).name if path else "")
        if not path or not looks_like_video(name):
            continue
        record = await asyncio.to_thread(agent.register_video, path)
        cl.user_session.set("video_id", record.video_id)
        responses.append(
            f"Registered `{name}` as `{record.video_id}`. "
            f"Status: `{record.status}`. Missing fields: "
            f"{', '.join(record.missing_fields) if record.missing_fields else 'none'}."
        )
    return responses


async def handle_command(agent: HaystackVideoAgent, content: str) -> None:
    stripped = content.strip()

    if stripped == "/tools":
        await render_tool_palette()
        return

    if stripped == "/videos":
        await render_video_inventory()
        return

    if stripped == "/status":
        await cl.Message(content=current_status()).send()
        return

    if stripped == "/services":
        await render_services()
        return

    if content.startswith("/register "):
        args = content.split(maxsplit=2)
        video_path = args[1]
        metadata = json.loads(args[2]) if len(args) > 2 else {}
        record = await asyncio.to_thread(
            agent.register_video,
            video_path,
            metadata,
        )
        cl.user_session.set("video_id", record.video_id)
        await cl.Message(content=json.dumps(record.to_dict(), indent=2)).send()
        return

    if content.startswith("/video "):
        video_id = content.split(maxsplit=1)[1].strip()
        cl.user_session.set("video_id", video_id)
        await cl.Message(content=current_status()).send()
        return

    if content == "/overview":
        video_id = require_active_video()
        overview = await asyncio.to_thread(agent.get_video_overview, video_id)
        await cl.Message(content=json.dumps(overview, indent=2)).send()
        return

    if content.startswith("/search "):
        query = content.split(maxsplit=1)[1]
        video_id = cl.user_session.get("video_id")
        result = await asyncio.to_thread(agent.search_content, query, video_id)
        await cl.Message(content=result["formatted"]).send()
        return

    if content.startswith("/set "):
        video_id = require_active_video()
        _, field, value = content.split(maxsplit=2)
        result = await asyncio.to_thread(
            agent.update_catalog_metadata,
            video_id,
            field,
            value,
            "user",
        )
        await cl.Message(content=json.dumps(result, indent=2)).send()
        return

    if content.startswith("/compare "):
        await compare_models(agent, content.split(maxsplit=1)[1])
        return

    await cl.Message(content="Unknown command.").send()


async def compare_models(agent: HaystackVideoAgent, query: str) -> None:
    models = compare_model_names()
    video_id = cl.user_session.get("video_id")
    prompt = query if not video_id else f"Active video_id: {video_id}\nUser request: {query}"
    outputs = []
    for model in models:
        candidate = HaystackVideoAgent(
            repository=agent.repository,
            provider="vllm",
            model=model,
            api_base_url=cl.user_session.get("vllm_url") or DEFAULT_VLLM_URL,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
            top_k=agent.top_k,
        )
        answer = await asyncio.to_thread(candidate.run_tool_agent, prompt, True, False)
        outputs.append(f"## {model}\n\n{answer}")
    await cl.Message(content="\n\n---\n\n".join(outputs)).send()


def current_status() -> str:
    video_id = cl.user_session.get("video_id") or "none"
    model = cl.user_session.get("model") or DEFAULT_MODEL
    vllm_url = cl.user_session.get("vllm_url") or DEFAULT_VLLM_URL
    catalog_root = cl.user_session.get("catalog_root") or DEFAULT_CATALOG_ROOT
    mmif_roots = cl.user_session.get("mmif_roots") or DEFAULT_MMIF_ROOTS
    service_path = cl.user_session.get("tool_services") or DEFAULT_TOOL_SERVICES
    root_count = len(normalise_roots(mmif_roots))
    return (
        f"Model: `{model}`\n"
        f"vLLM: `{vllm_url}`\n"
        f"Catalog: `{catalog_root}`\n"
        f"MMIF roots: `{root_count}` configured\n"
        f"Tool services: `{service_path}`\n"
        f"Active video: `{video_id}`"
    )


def require_active_video() -> str:
    video_id = cl.user_session.get("video_id")
    if not video_id:
        raise ValueError("No active video_id is set.")
    return video_id


def available_models() -> list[str]:
    names = [DEFAULT_MODEL]
    for name in compare_model_names():
        if name not in names:
            names.append(name)
    return names


def compare_model_names() -> list[str]:
    raw = os.environ.get("CLAMS_HAYSTACK_COMPARE_MODELS", DEFAULT_MODEL)
    return [name.strip() for name in raw.split(",") if name.strip()]


def looks_like_video(name: str) -> bool:
    return Path(name).suffix.lower() in {
        ".mp4",
        ".mov",
        ".mkv",
        ".avi",
        ".m4v",
        ".webm",
        ".mpg",
        ".mpeg",
    }


# ---------------------------------------------------------------------------
# Tool palette + form runner
# ---------------------------------------------------------------------------


def get_inventory() -> MmifInventory:
    inv = cl.user_session.get("mmif_inventory")
    if inv is None:
        inv = MmifInventory(
            mmif_roots=cl.user_session.get("mmif_roots") or DEFAULT_MMIF_ROOTS,
            repository=get_agent().repository,
        )
        cl.user_session.set("mmif_inventory", inv)
    return inv


def get_service_registry() -> ServiceRegistry:
    registry = cl.user_session.get("service_registry")
    if registry is None:
        service_path = cl.user_session.get("tool_services") or DEFAULT_TOOL_SERVICES
        path = Path(service_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        registry = ServiceRegistry.from_path(path) if path.exists() else ServiceRegistry.from_env()
        cl.user_session.set("service_registry", registry)
    return registry


async def render_tool_palette() -> None:
    cats = categories()
    lines = ["## 🛠 CLAMS Tool Palette", "_Click a tool to open its parameter form._", ""]
    actions: list[cl.Action] = []
    for cat_name in sorted(cats.keys()):
        lines.append(f"### {cat_name.title()}")
        for tool in cats[cat_name]:
            lines.append(f"- **{tool.label}** — {tool.description}")
            actions.append(
                cl.Action(
                    name="invoke_tool",
                    payload={"tool_id": tool.id},
                    label=tool.label,
                    tooltip=tool.description,
                )
            )
        lines.append("")
    await cl.Message(content="\n".join(lines), actions=actions).send()


async def render_video_inventory() -> None:
    inv = get_inventory()
    videos = inv.list_videos()
    if not videos:
        await cl.Message(content="No videos available.").send()
        return
    lines = ["## 📋 Indexed videos"]
    for v in videos[:30]:
        path = f" `{v.mmif_path}`" if v.mmif_path else ""
        lines.append(f"- **{v.video_id}**{path}")
    if len(videos) > 30:
        lines.append(f"_…and {len(videos) - 30} more_")
    await cl.Message(content="\n".join(lines)).send()


async def render_services() -> None:
    registry = get_service_registry()
    lines = [
        "## CLAMS tool services",
        "",
        f"Registry: `{cl.user_session.get('tool_services') or DEFAULT_TOOL_SERVICES}`",
        "",
    ]
    for service in registry.list_services():
        health = await asyncio.to_thread(registry.health, service)
        state = "up" if health.ok else "down"
        image = f" · `{service.image}`" if service.image else ""
        lines.append(
            f"- **{service.tool_id}** -> `{service.request_url}` "
            f"({state}){image}"
        )
        if service.notes:
            lines.append(f"  {service.notes}")
    await cl.Message(content="\n".join(lines)).send()


@cl.action_callback("invoke_tool")
async def on_invoke_tool(action: cl.Action) -> None:
    tool_id = action.payload.get("tool_id") if action.payload else None
    tool = get_tool(tool_id) if tool_id else None
    if tool is None:
        await cl.Message(content=f"Unknown tool: {tool_id!r}").send()
        return
    try:
        await run_form(tool)
    except FormCancelled:
        await cl.Message(content=f"Cancelled `{tool.id}`.").send()
    except Exception as exc:
        await cl.Message(content=f"Form error: {exc}").send()


class FormCancelled(Exception):
    pass


_SKIP = object()


async def run_form(tool: ToolDef) -> None:
    values: dict[str, Any] = {}
    card = cl.Message(content=render_form_card(tool, values, focused=tool.params[0].name if tool.params else None))
    await card.send()

    for param in tool.params:
        card.content = render_form_card(tool, values, focused=param.name)
        await card.update()
        try:
            value = await prompt_param(tool, param, values)
        except FormCancelled:
            card.content = render_form_card(tool, values) + "\n\n_Cancelled._"
            await card.update()
            raise
        if value is _SKIP:
            continue
        values[param.name] = value
        card.content = render_form_card(tool, values)
        await card.update()

    confirmed = await confirm_run(tool, values)
    if not confirmed:
        await cl.Message(content="Run cancelled.").send()
        return
    summary = await execute_tool(tool, values)
    await cl.Message(content=summary).send()


async def prompt_param(tool: ToolDef, param: ToolParam, values: dict[str, Any]) -> Any:
    optional_actions: list[cl.Action] = []
    if not param.required:
        optional_actions.append(
            cl.Action(
                name="form_skip",
                payload={"_skip": True},
                label="↷ Skip",
                tooltip="Leave this parameter at its default",
            )
        )
    optional_actions.append(
        cl.Action(name="form_cancel", payload={"_cancel": True}, label="✖ Cancel form")
    )

    if param.type == "video":
        return await prompt_video(param, optional_actions)
    if param.type == "mmif_view":
        video_id = values.get("video_id")
        return await prompt_view(param, video_id, optional_actions)
    if param.type == "tf_labels":
        video_id = values.get("video_id")
        view_id = values.get("source_view")
        return await prompt_tf_labels(param, video_id, view_id, optional_actions)
    if param.type == "tf_select":
        return await prompt_tf_select(
            param,
            values.get("video_id"),
            values.get("source_view"),
            values.get("target_labels") or [],
            optional_actions,
        )
    if param.type == "choice":
        return await prompt_choice(param, optional_actions)
    if param.type == "bool":
        return await prompt_bool(param, optional_actions)
    if param.type == "number":
        return await prompt_number(param, optional_actions)
    return await prompt_string(param, optional_actions)


async def prompt_video(param: ToolParam, opt_actions: list[cl.Action]) -> str:
    inv = get_inventory()
    videos = inv.list_videos()
    if not videos:
        raise FormCancelled("No videos available.")
    actions = []
    for v in videos[:24]:
        suffix = " ✓mmif" if v.mmif_path else ""
        actions.append(
            cl.Action(
                name="form_video",
                payload={"value": v.video_id},
                label=f"{v.label[:50]}{suffix}",
                tooltip=v.video_id,
            )
        )
    actions.extend(opt_actions)
    res = await cl.AskActionMessage(
        content=f"**{param.display_name}** — {param.description}",
        actions=actions,
        timeout=240,
    ).send()
    return _interpret_action(res, param)


async def prompt_view(
    param: ToolParam,
    video_id: str | None,
    opt_actions: list[cl.Action],
) -> str:
    if not video_id:
        raise FormCancelled("Video must be chosen before selecting a view.")
    inv = get_inventory()
    snap = inv.snapshot(video_id)
    contains_filter = (param.filters.get("contains") or "").lower()
    app_filter = (param.filters.get("app_contains") or "").lower()
    label_filter = (param.filters.get("label") or "").lower()
    candidates: list[ViewInfo] = []
    for v in snap.views:
        if contains_filter and not any(contains_filter in c.lower() for c in v.contains):
            continue
        if app_filter and app_filter not in v.app.lower():
            continue
        if label_filter and label_filter not in {l.lower() for l in v.label_counts}:
            continue
        candidates.append(v)
    if not candidates:
        candidates = snap.views
    if not candidates:
        raise FormCancelled(f"No views found for {video_id}.")
    actions = []
    for v in candidates[:20]:
        labels = ", ".join(
            f"{lbl}×{c}"
            for lbl, c in sorted(v.label_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )
        suffix = f" — {labels}" if labels else ""
        actions.append(
            cl.Action(
                name="form_view",
                payload={"value": v.view_id},
                label=f"{v.short_app} · {v.view_id}{suffix}"[:90],
                tooltip=v.display(),
            )
        )
    actions.extend(opt_actions)
    res = await cl.AskActionMessage(
        content=f"**{param.display_name}** — {param.description}",
        actions=actions,
        timeout=240,
    ).send()
    return _interpret_action(res, param)


async def prompt_tf_labels(
    param: ToolParam,
    video_id: str | None,
    view_id: str | None,
    opt_actions: list[cl.Action],
) -> list[str]:
    if not (video_id and view_id):
        raise FormCancelled("Video and view must be chosen first.")
    inv = get_inventory()
    options = [lbl for lbl, _ in inv.labels_in_view(video_id, view_id)]
    if not options:
        raise FormCancelled("No labels available in selected view.")
    selected: list[str] = []
    while True:
        actions = []
        for lbl in options:
            mark = "☑" if lbl in selected else "☐"
            count = dict(inv.labels_in_view(video_id, view_id)).get(lbl, 0)
            actions.append(
                cl.Action(
                    name="form_tf_label_toggle",
                    payload={"value": lbl},
                    label=f"{mark} {lbl} ({count})",
                )
            )
        actions.append(
            cl.Action(name="form_done", payload={"_done": True}, label="✔ Use selection")
        )
        actions.extend(opt_actions)
        body = (
            f"**{param.display_name}** — {param.description}\n"
            f"Selected: {', '.join(selected) if selected else '_(none)_'}"
        )
        res = await cl.AskActionMessage(content=body, actions=actions, timeout=240).send()
        if res is None:
            raise FormCancelled("Timed out.")
        if res.get("payload", {}).get("_cancel"):
            raise FormCancelled()
        if res.get("payload", {}).get("_skip"):
            return _SKIP  # type: ignore[return-value]
        if res.get("payload", {}).get("_done"):
            if not selected and param.required:
                continue
            return selected
        v = res.get("payload", {}).get("value")
        if v in selected:
            selected.remove(v)
        elif v is not None:
            selected.append(v)


async def prompt_tf_select(
    param: ToolParam,
    video_id: str | None,
    view_id: str | None,
    labels: list[str],
    opt_actions: list[cl.Action],
) -> list[str]:
    if not (video_id and view_id):
        if not param.required:
            return _SKIP  # type: ignore[return-value]
        raise FormCancelled("Video and view must be chosen first.")
    inv = get_inventory()
    tfs = inv.timeframes(video_id, view_id, labels=labels or None, limit=24)
    if not tfs:
        if not param.required:
            return _SKIP  # type: ignore[return-value]
        raise FormCancelled("No matching TimeFrames.")
    selected: list[str] = []
    while True:
        actions = []
        for tf in tfs:
            mark = "☑" if tf.annotation_id in selected else "☐"
            actions.append(
                cl.Action(
                    name="form_tf_pick",
                    payload={"value": tf.annotation_id},
                    label=f"{mark} {tf.display()}"[:120],
                    tooltip=tf.text_preview or tf.annotation_id,
                )
            )
        actions.append(
            cl.Action(
                name="form_select_all",
                payload={"_all": True},
                label="☑ Select all",
            )
        )
        actions.append(
            cl.Action(name="form_done", payload={"_done": True}, label="✔ Done")
        )
        actions.extend(opt_actions)
        body = (
            f"**{param.display_name}** — {param.description}\n"
            f"Picked: {len(selected)}/{len(tfs)}"
        )
        res = await cl.AskActionMessage(content=body, actions=actions, timeout=240).send()
        if res is None:
            raise FormCancelled("Timed out.")
        payload = res.get("payload") or {}
        if payload.get("_cancel"):
            raise FormCancelled()
        if payload.get("_skip"):
            return _SKIP  # type: ignore[return-value]
        if payload.get("_all"):
            selected = [tf.annotation_id for tf in tfs]
            continue
        if payload.get("_done"):
            return selected
        v = payload.get("value")
        if v in selected:
            selected.remove(v)
        elif v is not None:
            selected.append(v)


async def prompt_choice(param: ToolParam, opt_actions: list[cl.Action]) -> Any:
    if param.multivalued:
        # multi-select over choice list
        selected: list[str] = []
        while True:
            actions = []
            for c in param.choices:
                mark = "☑" if c in selected else "☐"
                actions.append(
                    cl.Action(name="form_choice_toggle", payload={"value": c}, label=f"{mark} {c}")
                )
            actions.append(cl.Action(name="form_done", payload={"_done": True}, label="✔ Done"))
            actions.extend(opt_actions)
            body = (
                f"**{param.display_name}** — {param.description}\n"
                f"Selected: {', '.join(selected) if selected else '_(none)_'}"
            )
            res = await cl.AskActionMessage(content=body, actions=actions, timeout=240).send()
            if res is None:
                raise FormCancelled("Timed out.")
            payload = res.get("payload") or {}
            if payload.get("_cancel"):
                raise FormCancelled()
            if payload.get("_skip"):
                return _SKIP
            if payload.get("_done"):
                return selected if selected else (param.default or [])
            v = payload.get("value")
            if v in selected:
                selected.remove(v)
            elif v is not None:
                selected.append(v)
    actions = [
        cl.Action(name="form_choice", payload={"value": c}, label=c) for c in param.choices
    ]
    actions.extend(opt_actions)
    body = f"**{param.display_name}** — {param.description}"
    if param.default is not None:
        body += f"\n_default: `{param.default}`_"
    res = await cl.AskActionMessage(content=body, actions=actions, timeout=240).send()
    return _interpret_action(res, param)


async def prompt_bool(param: ToolParam, opt_actions: list[cl.Action]) -> bool:
    actions = [
        cl.Action(name="form_bool", payload={"value": True}, label="✅ Yes"),
        cl.Action(name="form_bool", payload={"value": False}, label="⛔ No"),
    ]
    actions.extend(opt_actions)
    body = f"**{param.display_name}** — {param.description}"
    if param.default is not None:
        body += f"\n_default: `{param.default}`_"
    res = await cl.AskActionMessage(content=body, actions=actions, timeout=240).send()
    return _interpret_action(res, param)


async def prompt_number(param: ToolParam, opt_actions: list[cl.Action]) -> float | int:
    body = f"**{param.display_name}** — {param.description}"
    if param.default is not None:
        body += f" (default: `{param.default}`, blank to accept)"
    res = await cl.AskUserMessage(content=body, timeout=240).send()
    if res is None:
        raise FormCancelled("Timed out.")
    raw = (res.get("output") or "").strip()
    if not raw and param.default is not None:
        return param.default
    if not raw and not param.required:
        return _SKIP
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        await cl.Message(content=f"Couldn't parse `{raw}` as a number; using default.").send()
        return param.default if param.default is not None else 0


async def prompt_string(param: ToolParam, opt_actions: list[cl.Action]) -> str:
    body = f"**{param.display_name}** — {param.description}"
    if param.default not in (None, ""):
        body += f" (default: `{param.default}`, blank to accept)"
    elif not param.required:
        body += " (blank to skip)"
    if param.placeholder:
        body += f" e.g. `{param.placeholder}`"
    res = await cl.AskUserMessage(content=body, timeout=240).send()
    if res is None:
        raise FormCancelled("Timed out.")
    raw = (res.get("output") or "").strip()
    if not raw:
        if param.default not in (None, ""):
            return param.default
        if not param.required:
            return _SKIP
    return raw


def _interpret_action(res: dict | None, param: ToolParam) -> Any:
    if res is None:
        raise FormCancelled("Timed out.")
    payload = res.get("payload") or {}
    if payload.get("_cancel"):
        raise FormCancelled()
    if payload.get("_skip"):
        return _SKIP
    return payload.get("value")


async def confirm_run(tool: ToolDef, values: dict[str, Any]) -> bool:
    summary_lines = [f"### Ready to run `{tool.id}` — {tool.label}", ""]
    for p in tool.params:
        v = values.get(p.name, "_(unset)_")
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) if v else "_(none)_"
        summary_lines.append(f"- **{p.display_name}**: `{v}`")
    summary_lines.append("")
    if _service_execution_enabled():
        summary_lines.append("_Confirm to POST the selected MMIF to the configured CLAMS service._")
    else:
        summary_lines.append(
            "_Confirm to record a service request. Set "
            "`CLAMS_HAYSTACK_EXECUTE_SERVICES=1` to enable live endpoint calls._"
        )
    res = await cl.AskActionMessage(
        content="\n".join(summary_lines),
        actions=[
            cl.Action(name="form_confirm", payload={"go": True}, label="🚀 Run"),
            cl.Action(name="form_confirm", payload={"go": False}, label="✖ Cancel"),
        ],
        timeout=240,
    ).send()
    if res is None:
        return False
    return bool((res.get("payload") or {}).get("go"))


async def execute_tool(tool: ToolDef, values: dict[str, Any]) -> str:
    """Record or execute a CLAMS tool request through the service registry."""
    agent = get_agent()
    registry = get_service_registry()
    service = registry.for_tool(tool.id)
    video_id = values.get("video_id")
    note = {
        "requested_tool": tool.id,
        "tool_label": tool.label,
        "params": _jsonable(values),
        "service": service.to_dict() if service else None,
    }
    agent.repository.record_tool_request(
        video_id=video_id,
        tool_id=tool.id,
        request=note,
        status="requested",
    )

    service_result = None
    output_path = None
    if _service_execution_enabled():
        if service is None:
            service_result = {"ok": False, "error": f"No service configured for {tool.id}"}
        elif not video_id:
            service_result = {"ok": False, "error": "No video_id selected."}
        else:
            snap = get_inventory().snapshot(video_id)
            if not snap.mmif_path:
                service_result = {
                    "ok": False,
                    "error": f"No MMIF file found for {video_id}.",
                }
            else:
                with Path(snap.mmif_path).open() as f:
                    input_mmif = json.load(f)
                params = _service_params(tool, values)
                call = await asyncio.to_thread(
                    registry.invoke_mmif,
                    tool.id,
                    input_mmif,
                    params,
                )
                service_result = {
                    "ok": call.ok,
                    "tool_id": call.tool_id,
                    "url": call.url,
                    "status_code": call.status_code,
                    "error": call.error,
                    "has_json_response": call.response_json is not None,
                    "response_preview": call.response_text[:1000],
                }
                if call.ok and call.response_json is not None:
                    output_path = _write_tool_output(
                        agent.repository.root,
                        video_id,
                        tool.id,
                        call.response_json,
                    )
                    agent.repository.record_tool_request(
                        video_id=video_id,
                        tool_id=tool.id,
                        request={
                            **note,
                            "output_path": str(output_path),
                            "status_code": call.status_code,
                        },
                        status="executed",
                    )

    lines = [
        f"### Tool request `{tool.id}` for `{video_id or '(no video)'}`",
        "",
        "**Effective parameters**:",
        "```json",
        json.dumps(_jsonable(values), indent=2, sort_keys=True),
        "```",
        "",
    ]
    if service:
        lines.append(f"Service endpoint: `{service.request_url}`")
    else:
        lines.append("Service endpoint: _(not configured)_")
    if _service_execution_enabled():
        lines.extend(["", "**Service result**:", "```json"])
        lines.append(json.dumps(service_result, indent=2, sort_keys=True, default=str))
        lines.append("```")
        if output_path:
            lines.append(f"Saved output MMIF: `{output_path}`")
    else:
        lines.append("")
        lines.append("_Recorded only; no endpoint call was made._")
    return "\n".join(lines)


def _service_execution_enabled() -> bool:
    return os.environ.get("CLAMS_HAYSTACK_EXECUTE_SERVICES", "").lower() in {
        "1",
        "true",
        "yes",
    }


def _service_params(tool: ToolDef, values: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if tool.id == "asr_whisper":
        if values.get("model"):
            params["modelSize"] = values["model"]
        if values.get("language"):
            params["modelLang"] = values["language"]
    elif tool.id == "smolvlm2_captioner":
        if values.get("target_labels"):
            params["label"] = values["target_labels"]
        if values.get("frames_per_clip"):
            params["sampleRatio"] = values["frames_per_clip"]
    elif tool.id in {"vlm_ocr_kie", "credits_ocr"}:
        if values.get("target_labels"):
            params["frameType"] = values["target_labels"]
        if values.get("model"):
            params["modelName"] = values["model"]
    elif tool.id == "scene_summary":
        params["prompt"] = (
            "Describe the visible scene, people, setting, on-screen text, "
            "and visual evidence without assuming unseen context."
        )
    return params


def _write_tool_output(root: Path, video_id: str, tool_id: str, mmif: dict) -> Path:
    out_dir = root / "tool_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_video = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_id)
    safe_tool = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_id)
    out_path = out_dir / f"{safe_video}_{safe_tool}_{stamp}.mmif"
    with out_path.open("w") as f:
        json.dump(mmif, f, sort_keys=True)
    return out_path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
