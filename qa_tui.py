#!/usr/bin/env python3
"""Knowledge Graph QA TUI — multi-panel interface for exploring the KG and
answering multi-hop questions over broadcast news video archives.

Usage:
    python qa_tui.py                    # Launch with KG tools only (no LLM agent)
    python qa_tui.py --agent            # Launch with full LangGraph agent
"""

import argparse
import json
import logging
import os
import sys
from uuid import uuid4

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Input, Static, Tree, Label, Rule, TabbedContent, TabPane,
)
from textual.widgets.tree import TreeNode

from utils.kg_tools import kg_search, kg_neighbors, kg_path, kg_query

logger = logging.getLogger(__name__)

# ── Color scheme (matches explore_graph.py) ──
TYPE_COLORS = {
    "person": "dodger_blue",
    "organization": "dark_orange",
    "location": "green",
    "event": "red",
    "product": "medium_purple",
    "title": "dark_cyan",
    "other": "grey62",
    "role": "yellow",
    "wikidata_value": "light_pink1",
}

PRED_STYLES = {
    "mentioned_with": "dim",
    "quoted_by": "cyan",
    "interacted_with": "bold cyan",
    "has_role": "yellow",
    "affiliated_with": "dark_orange",
    "occupation": "light_pink1",
    "position_held": "red",
    "political_party": "medium_purple",
    "educated_at": "green",
    "employer": "dark_cyan",
}


# ── Widgets ──

class UserMessage(Static):
    DEFAULT_CSS = """
    UserMessage {
        background: #1a2a3a;
        margin: 1 0;
        padding: 1 2;
        border-left: thick dodger_blue;
    }
    """


class AssistantMessage(Static):
    DEFAULT_CSS = """
    AssistantMessage {
        margin: 1 0;
        padding: 1 2;
        border-left: thick green;
    }
    """


class ToolResult(Static):
    DEFAULT_CSS = """
    ToolResult {
        color: $text-muted;
        margin: 0 0;
        padding: 0 2;
        border-left: thick yellow;
    }
    """


class EntityCard(Static):
    """Clickable entity display in the KG panel."""

    DEFAULT_CSS = """
    EntityCard {
        padding: 0 1;
        margin: 0 0;
        height: auto;
    }
    EntityCard:hover {
        background: #2a2a4a;
    }
    """

    def __init__(self, entity_data: dict, **kwargs):
        self.entity_data = entity_data
        name = entity_data.get("name", entity_data.get("canonical_name", "?"))
        etype = entity_data.get("type", entity_data.get("entity_type", ""))
        color = TYPE_COLORS.get(etype, "white")
        label = f"[{color}]{name}[/{color}] [{etype}]"
        super().__init__(label, **kwargs)

    async def on_click(self) -> None:
        app = self.app
        if isinstance(app, QATui):
            entity_id = self.entity_data.get("entity_id", "")
            name = self.entity_data.get("name", self.entity_data.get("canonical_name", ""))
            app.explore_entity(entity_id or name)


class RelationLine(Static):
    DEFAULT_CSS = """
    RelationLine {
        padding: 0 2;
        color: grey62;
        height: auto;
    }
    """


class SectionHeader(Static):
    DEFAULT_CSS = """
    SectionHeader {
        padding: 0 1;
        margin: 1 0 0 0;
        color: grey50;
        text-style: bold;
        text-transform: uppercase;
    }
    """


# ── Main App ──

class QATui(App):
    """Multi-panel KG QA interface."""

    TITLE = "CLAMS KG Explorer"
    SUB_TITLE = "Multi-Hop QA over Broadcast News"

    CSS = """
    Screen {
        layout: vertical;
        background: #0f0f1a;
    }

    #main-area {
        height: 1fr;
    }

    #left-panel {
        width: 28;
        border-right: tall #2a2a4a;
        overflow-y: auto;
        padding: 0 1;
    }

    #center-panel {
        width: 1fr;
        overflow-y: auto;
        padding: 0 1;
    }

    #right-panel {
        width: 36;
        border-left: tall #2a2a4a;
        overflow-y: auto;
        padding: 0 1;
    }

    #reasoning-tree {
        height: 1fr;
    }

    #input-area {
        dock: bottom;
        height: auto;
        max-height: 5;
        padding: 0 2;
        background: #1a1a2e;
    }

    #user-input {
        border: tall #3a3a5a;
    }
    #user-input:focus {
        border: tall #4488cc;
    }

    .panel-title {
        padding: 0 1;
        margin: 0 0 1 0;
        color: #7777aa;
        text-style: bold;
    }

    Tree {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+r", "focus_right", "KG Panel", show=True),
    ]

    def __init__(self, graph=None, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
        self.thread_id = f"qa-{uuid4().hex[:8]}"
        self._is_streaming = False
        self._hop_counter = 0
        self._current_question_node = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-area"):
            # Left: Reasoning chain
            with Vertical(id="left-panel"):
                yield Label("Reasoning", classes="panel-title")
                yield Tree("Questions", id="reasoning-tree")

            # Center: Chat
            with VerticalScroll(id="center-panel"):
                yield Static(
                    "[bold #7777aa]CLAMS KG Explorer[/bold #7777aa]\n"
                    "Ask questions about the broadcast news archive.\n"
                    "Try: [italic]Who from Harvard appeared in the same video?[/italic]\n"
                    "Or explore: [italic]/search Clinton[/italic]  "
                    "[italic]/neighbors Putin[/italic]  "
                    "[italic]/path Lincoln Bush[/italic]\n"
                    "[italic]/query educated_at Harvard[/italic]",
                    id="welcome",
                )

            # Right: KG context
            with VerticalScroll(id="right-panel"):
                yield Label("Knowledge Graph", classes="panel-title")

        yield Input(
            placeholder="Ask a question or /search, /neighbors, /path, /query...",
            id="user-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#reasoning-tree", Tree)
        tree.show_root = False
        self.query_one("#user-input").focus()

    # ── Input dispatch ──

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        if text.startswith("/"):
            self._handle_command(text)
        else:
            self._handle_question(text)

    # ── Slash commands for direct KG exploration ──

    @work(exclusive=True, thread=False)
    async def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        chat = self.query_one("#center-panel")

        if cmd == "/search":
            if not arg:
                await chat.mount(ToolResult("Usage: /search <entity name>"))
                return
            await chat.mount(UserMessage(f"[bold]Search:[/bold] {arg}"))
            result = kg_search(arg)
            await chat.mount(ToolResult(result["result_summary"]))
            self._populate_kg_panel(result.get("entities", []), f"Search: {arg}")
            chat.scroll_end(animate=False)

        elif cmd == "/neighbors" or cmd == "/n":
            if not arg:
                await chat.mount(ToolResult("Usage: /neighbors <entity name or id>"))
                return
            await chat.mount(UserMessage(f"[bold]Neighbors:[/bold] {arg}"))
            result = kg_neighbors(arg)
            await chat.mount(ToolResult(result["result_summary"]))
            if result.get("detail"):
                await chat.mount(ToolResult(result["detail"][:1000]))
            self._populate_kg_panel_from_neighbors(result)
            chat.scroll_end(animate=False)

        elif cmd == "/path":
            args_split = arg.split(",") if "," in arg else arg.split(maxsplit=1)
            if len(args_split) < 2:
                await chat.mount(ToolResult("Usage: /path <entity1>, <entity2>"))
                return
            e1, e2 = args_split[0].strip(), args_split[1].strip()
            await chat.mount(UserMessage(f"[bold]Path:[/bold] {e1} → {e2}"))
            result = kg_path(e1, e2)
            await chat.mount(ToolResult(result["result_summary"]))
            if result.get("detail"):
                await chat.mount(ToolResult(result["detail"]))
            self._add_path_to_reasoning(result)
            chat.scroll_end(animate=False)

        elif cmd == "/query" or cmd == "/q":
            q_parts = arg.split(maxsplit=1)
            if not q_parts:
                await chat.mount(ToolResult("Usage: /query <predicate> [value]"))
                return
            pred = q_parts[0]
            val = q_parts[1] if len(q_parts) > 1 else None
            await chat.mount(UserMessage(f"[bold]Query:[/bold] {pred}" + (f" = {val}" if val else "")))
            result = kg_query(pred, val, subject_type="person")
            await chat.mount(ToolResult(result["result_summary"]))
            self._populate_kg_panel(result.get("entities", []), f"{pred}: {val or 'all'}")
            chat.scroll_end(animate=False)

        elif cmd == "/stats":
            from utils.kg_tools import get_kg
            stats = get_kg().stats()
            await chat.mount(ToolResult(json.dumps(stats, indent=2)))
            chat.scroll_end(animate=False)

        elif cmd == "/help":
            await chat.mount(AssistantMessage(
                "[bold]Commands:[/bold]\n"
                "  /search <name>        — Find entities by name\n"
                "  /neighbors <name>     — Show entity relations (/n for short)\n"
                "  /path <e1>, <e2>      — Find path between entities\n"
                "  /query <pred> [value] — Find entities by relation (/q for short)\n"
                "  /stats                — Show graph statistics\n"
                "\n[bold]Example queries:[/bold]\n"
                "  /search Hillary Clinton\n"
                "  /n Putin\n"
                "  /q educated_at Harvard\n"
                "  /path Lincoln, Bush\n"
                "\n[bold]Free-form questions:[/bold]\n"
                "  Just type a question without / prefix"
            ))
            chat.scroll_end(animate=False)
        else:
            await chat.mount(ToolResult(f"Unknown command: {cmd}. Type /help for commands."))
            chat.scroll_end(animate=False)

    # ── Free-form questions ──

    @work(exclusive=True, thread=False)
    async def _handle_question(self, text: str) -> None:
        chat = self.query_one("#center-panel")
        await chat.mount(UserMessage(f"[bold]You:[/bold] {text}"))
        chat.scroll_end(animate=False)

        if self.graph:
            await self._run_agent_qa(text)
        else:
            # No agent — do basic KG-powered QA
            await self._basic_kg_qa(text)

    async def _basic_kg_qa(self, question: str) -> None:
        """Answer questions using direct KG lookups (no LLM agent)."""
        chat = self.query_one("#center-panel")
        tree = self.query_one("#reasoning-tree", Tree)

        # Add question to reasoning tree
        q_node = tree.root.add(f"[bold]{question}[/bold]")
        q_node.expand()
        self._current_question_node = q_node

        # Simple entity extraction: find capitalized words/phrases
        import re
        words = question.split()
        # Try progressively smaller chunks
        entities_found = []
        for i in range(len(words)):
            for j in range(min(i + 4, len(words)), i, -1):
                phrase = " ".join(words[i:j])
                # Skip common words
                if phrase.lower() in {"who", "what", "where", "when", "which", "how",
                                       "the", "a", "an", "is", "are", "was", "were",
                                       "did", "do", "does", "from", "in", "at", "of",
                                       "same", "video", "appear", "appeared", "have",
                                       "has", "had", "their", "this", "that", "with"}:
                    continue
                if len(phrase) < 3:
                    continue
                result = kg_search(phrase)
                if result["entities"]:
                    entities_found.append((phrase, result["entities"][0]))
                    hop = q_node.add(
                        f"[green]search[/green] '{phrase}' → "
                        f"[bold]{result['entities'][0]['name']}[/bold]"
                    )
                    break

        if not entities_found:
            await chat.mount(AssistantMessage(
                "I couldn't find any matching entities in the knowledge graph. "
                "Try using /search to explore specific names."
            ))
            return

        # For each found entity, show its neighborhood
        all_related = []
        for phrase, entity in entities_found:
            result = kg_neighbors(entity["entity_id"])
            if result["success"]:
                hop = q_node.add(
                    f"[yellow]explore[/yellow] {entity['name']} → "
                    f"{result['result_summary']}"
                )
                self._populate_kg_panel_from_neighbors(result)
                all_related.append((entity, result))

        # Build a response from what we found
        response_parts = []
        for entity, nbr_result in all_related:
            response_parts.append(f"**{entity['name']}** ({entity['type']})")
            if nbr_result.get("detail"):
                # Truncate detail for display
                detail = nbr_result["detail"]
                if len(detail) > 600:
                    detail = detail[:600] + "\n  ..."
                response_parts.append(detail)

        await chat.mount(AssistantMessage("\n\n".join(response_parts)))
        chat.scroll_end(animate=False)

    async def _run_agent_qa(self, question: str) -> None:
        """Run full LangGraph agent for QA."""
        chat = self.query_one("#center-panel")
        tree = self.query_one("#reasoning-tree", Tree)

        q_node = tree.root.add(f"[bold]{question[:60]}{'...' if len(question) > 60 else ''}[/bold]")
        q_node.expand()
        self._current_question_node = q_node

        try:
            input_state = {
                "user_request": question,
                "video_path": "",
                "image_paths": [],
                "max_depth": 8,
                "messages": [],
            }
            config = {"configurable": {"thread_id": self.thread_id}}
            tool_history_seen = 0

            async for event in self.graph.astream(
                input_state, config, stream_mode="updates"
            ):
                for node_name, update in event.items():
                    if node_name == "understand_request":
                        messages = update.get("messages", [])
                        for msg in messages:
                            content = msg.content if hasattr(msg, "content") else str(msg)
                            await chat.mount(AssistantMessage(
                                f"[bold]Agent:[/bold] {content}"
                            ))
                            chat.scroll_end(animate=False)

                    elif node_name == "agent_step":
                        action = update.get("next_action", {})
                        action_type = action.get("action", "unknown")
                        reasoning = action.get("reasoning", "")

                        if action_type == "use_kg":
                            op = action.get("operation", "?")
                            params = action.get("parameters", {})
                            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                            await chat.mount(ToolResult(
                                f"[yellow]KG:[/yellow] [bold]{op}[/bold]({param_str})\n"
                                f"[dim]{reasoning}[/dim]"
                            ))
                            q_node.add(f"[yellow]{op}[/yellow] {param_str[:50]}")
                        elif action_type == "use_web":
                            op = action.get("operation", "?")
                            await chat.mount(ToolResult(
                                f"[cyan]Web:[/cyan] [bold]{op}[/bold] — {reasoning}"
                            ))
                            q_node.add(f"[cyan]{op}[/cyan]")
                        elif action_type == "done":
                            answer = action.get("answer", reasoning)
                            await chat.mount(AssistantMessage(
                                f"[bold green]Answer:[/bold green] {answer}"
                            ))
                            q_node.add(f"[green]✓ done[/green]")
                        else:
                            await chat.mount(ToolResult(
                                f"[dim]{action_type}:[/dim] {reasoning}"
                            ))
                        chat.scroll_end(animate=False)

                    elif node_name == "execute_tool":
                        tool_history = update.get("tool_history", [])
                        new_entries = tool_history[tool_history_seen:]
                        tool_history_seen = len(tool_history)

                        for entry in new_entries:
                            tool = entry.get("tool", "unknown")
                            success = entry.get("success", False)
                            summary = entry.get("result_summary", "")
                            exec_time = entry.get("execution_time", 0)
                            status = "[green]OK[/green]" if success else "[red]FAIL[/red]"

                            await chat.mount(ToolResult(
                                f"[dim]Result:[/dim] {tool} {status} "
                                f"[dim]({exec_time:.1f}s)[/dim]\n{summary}"
                            ))

                            # Update KG panel with entities from KG results
                            if tool.startswith("kg:") and success:
                                detail = entry.get("detail", "")
                                if detail:
                                    await chat.mount(ToolResult(
                                        detail[:800] + ("\n..." if len(detail) > 800 else "")
                                    ))

                            q_node.add(
                                f"{'[green]✓[/green]' if success else '[red]✗[/red]'} "
                                f"{tool} → {summary[:60]}"
                            )
                            chat.scroll_end(animate=False)

                    elif node_name == "summarize_results":
                        summary = update.get("execution_summary", "")
                        if summary:
                            await chat.mount(AssistantMessage(
                                f"[bold]Summary:[/bold]\n{summary}"
                            ))
                            chat.scroll_end(animate=False)

        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            await chat.mount(ToolResult(f"[red]Error: {e}[/red]"))
            chat.scroll_end(animate=False)

    # ── KG Panel population ──

    def _populate_kg_panel(self, entities: list[dict], title: str = "") -> None:
        """Populate the right panel with entity cards."""
        panel = self.query_one("#right-panel")
        # Clear existing content except the title label
        for child in list(panel.children):
            if not isinstance(child, Label) or "panel-title" not in child.classes:
                child.remove()

        if title:
            panel.mount(SectionHeader(title))

        for entity in entities[:30]:
            panel.mount(EntityCard(entity))

    def _populate_kg_panel_from_neighbors(self, result: dict) -> None:
        """Populate KG panel from a neighbors result, grouped by relation."""
        panel = self.query_one("#right-panel")
        for child in list(panel.children):
            if not isinstance(child, Label) or "panel-title" not in child.classes:
                child.remove()

        center = result.get("center_entity", {})
        if center:
            panel.mount(SectionHeader(f"{center.get('name', '?')}"))

        # Group by predicate from detail
        entities = result.get("entities", [])
        by_pred: dict[str, list] = {}
        for e in entities:
            # entities from kg_neighbors don't have predicate, but we can infer from detail
            # For now just list them
            etype = e.get("type", "")
            if etype not in by_pred:
                by_pred[etype] = []
            by_pred[etype].append(e)

        for etype, ents in sorted(by_pred.items()):
            panel.mount(SectionHeader(f"{etype} ({len(ents)})"))
            for e in ents[:15]:
                panel.mount(EntityCard(e))

    def _add_path_to_reasoning(self, result: dict) -> None:
        """Add a path result to the reasoning tree."""
        tree = self.query_one("#reasoning-tree", Tree)
        path_data = result.get("path", [])
        if not path_data:
            return

        path_node = tree.root.add(f"[bold]{result['result_summary']}[/bold]")
        path_node.expand()

        for item in path_data:
            if item["type"] == "entity":
                etype = item.get("entity_type", "")
                color = TYPE_COLORS.get(etype, "white")
                path_node.add_leaf(f"[{color}]{item['canonical_name']}[/{color}] [{etype}]")
            elif item["type"] == "relation":
                pred = item.get("predicate", "?")
                style = PRED_STYLES.get(pred, "dim")
                path_node.add_leaf(f"  [{style}]--{pred}-->[/{style}]")

    # ── Entity exploration (triggered by clicking EntityCard) ──

    @work(exclusive=True, thread=False)
    async def explore_entity(self, entity_name_or_id: str) -> None:
        """Explore an entity when clicked in the KG panel."""
        chat = self.query_one("#center-panel")
        result = kg_neighbors(entity_name_or_id)
        if result["success"]:
            await chat.mount(ToolResult(
                f"[bold]Exploring:[/bold] {result['result_summary']}"
            ))
            if result.get("detail"):
                detail = result["detail"]
                if len(detail) > 1200:
                    detail = detail[:1200] + "\n..."
                await chat.mount(ToolResult(detail))
            self._populate_kg_panel_from_neighbors(result)
            chat.scroll_end(animate=False)

    # ── Actions ──

    def action_clear(self) -> None:
        chat = self.query_one("#center-panel")
        for child in list(chat.children):
            if child.id != "welcome":
                child.remove()
        tree = self.query_one("#reasoning-tree", Tree)
        tree.root.remove_children()
        self.thread_id = f"qa-{uuid4().hex[:8]}"

    def action_focus_right(self) -> None:
        self.query_one("#right-panel").focus()


# ── Entrypoint ──

def main():
    parser = argparse.ArgumentParser(description="KG QA TUI")
    parser.add_argument("--agent", action="store_true", help="Enable LangGraph agent")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename="qa_tui.log", level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    graph = None
    if args.agent:
        from agent import create_pipeline_graph
        graph = create_pipeline_graph()

    app = QATui(graph=graph)
    app.run()


if __name__ == "__main__":
    main()
