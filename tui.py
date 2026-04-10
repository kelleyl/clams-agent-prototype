#!/usr/bin/env python3
"""CLAMS Agent TUI - Terminal User Interface for the CLAMS pipeline agent."""

import argparse
import logging
import os
import re
import sys
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Static, LoadingIndicator

from langgraph.types import Command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Message widgets
# ---------------------------------------------------------------------------

class UserMessage(Static):
    """A user message in the chat."""

    DEFAULT_CSS = """
    UserMessage {
        background: $primary-background;
        margin: 1 2;
        padding: 1 2;
        border: tall $primary;
    }
    """


class AssistantMessage(Static):
    """An assistant message, updated incrementally during streaming."""

    DEFAULT_CSS = """
    AssistantMessage {
        margin: 1 2;
        padding: 1 2;
        border-left: thick $success;
    }
    """


class ToolCallDisplay(Static):
    """Compact display of a tool call or result."""

    DEFAULT_CSS = """
    ToolCallDisplay {
        color: $text-muted;
        margin: 0 4;
        padding: 0 1;
        border-left: thick $warning;
    }
    """


class ErrorDisplay(Static):
    """Error message display."""

    DEFAULT_CSS = """
    ErrorDisplay {
        color: $error;
        margin: 1 2;
        padding: 1 2;
        border: wide $error;
    }
    """


class SystemMessage(Static):
    """System/status message."""

    DEFAULT_CSS = """
    SystemMessage {
        color: $text-muted;
        margin: 1 2;
        padding: 1;
        text-style: italic;
    }
    """


class ApprovalPrompt(Static):
    """Approval prompt shown when the agent requests confirmation."""

    DEFAULT_CSS = """
    ApprovalPrompt {
        background: $warning 20%;
        margin: 1 2;
        padding: 1 2;
        border: tall $warning;
    }
    """


# ---------------------------------------------------------------------------
# File path extraction
# ---------------------------------------------------------------------------

def extract_file_path(text: str) -> str | None:
    """Try to extract a file path from user input."""
    # Match absolute paths like /path/to/file.mp4
    match = re.search(r'(/[^\s]+\.\w{2,5})', text)
    if match:
        return match.group(1)
    # Match ~/relative paths
    match = re.search(r'(~/[^\s]+\.\w{2,5})', text)
    if match:
        return os.path.expanduser(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Main TUI App
# ---------------------------------------------------------------------------

class CLAMSTui(App):
    """CLAMS Agent Terminal User Interface."""

    TITLE = "CLAMS Agent"
    SUB_TITLE = "Pipeline Assistant"

    CSS = """
    Screen {
        layout: vertical;
    }
    #chat-view {
        height: 1fr;
        overflow-y: auto;
    }
    #input-area {
        dock: bottom;
        height: auto;
        max-height: 5;
        padding: 0 2;
    }
    #loading {
        dock: bottom;
        height: 1;
        display: none;
    }
    #loading.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear", "Clear chat"),
    ]

    def __init__(
        self,
        graph,
        gpu_available: bool = False,
        video_path: str = "",
        max_depth: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.gpu_available = gpu_available
        self.video_path = video_path
        self.max_depth = max_depth
        self.thread_id = f"tui-{uuid4().hex[:8]}"
        self._is_streaming = False
        self._awaiting_approval = False
        self._tool_history_seen = 0  # Track displayed tool results

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-view")
        yield LoadingIndicator(id="loading")
        yield Input(
            placeholder="Describe what to analyze... (Enter to send)",
            id="user-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Show welcome message on mount."""
        chat = self.query_one("#chat-view")
        gpu_label = "[green]ON[/green]" if self.gpu_available else "[red]OFF[/red]"
        video_label = f"Video: {self.video_path}" if self.video_path else "No video set (include path in your message)"
        chat.mount(
            SystemMessage(
                f"CLAMS Agent ready. GPU: {gpu_label}\n"
                f"{video_label}\n"
                f"Session: {self.thread_id}"
            )
        )
        self.query_one("#user-input").focus()

    # -----------------------------------------------------------------------
    # Input handling
    # -----------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user pressing Enter."""
        user_text = event.value.strip()
        if not user_text:
            return
        if self._is_streaming and not self._awaiting_approval:
            return  # Don't accept input while streaming (except during approval)

        event.input.value = ""
        chat = self.query_one("#chat-view")

        if self._awaiting_approval:
            # User is responding to an approval request
            await chat.mount(UserMessage(f"[bold]You:[/bold] {user_text}"))
            chat.scroll_end(animate=False)
            self._handle_approval(user_text)
        else:
            # New request
            await chat.mount(UserMessage(f"[bold]You:[/bold] {user_text}"))
            chat.scroll_end(animate=False)
            self._run_graph(user_text)

    # -----------------------------------------------------------------------
    # Graph execution
    # -----------------------------------------------------------------------

    @property
    def _loading(self):
        return self.query_one("#loading")

    def _show_loading(self, show: bool) -> None:
        if show:
            self._loading.add_class("visible")
        else:
            self._loading.remove_class("visible")

    @work(exclusive=True, thread=False)
    async def _run_graph(self, user_input: str) -> None:
        """Invoke the LangGraph agent with the user's request."""
        self._is_streaming = True
        self._show_loading(True)
        chat = self.query_one("#chat-view")

        try:
            # Extract video path from message if not set
            video_path = self.video_path
            if not video_path:
                extracted = extract_file_path(user_input)
                if extracted:
                    video_path = extracted
                    await chat.mount(
                        SystemMessage(f"Detected file: {video_path}")
                    )
                    chat.scroll_end(animate=False)

            self._tool_history_seen = 0

            # Build initial state
            input_state = {
                "user_request": user_input,
                "video_path": video_path,
                "image_paths": [],
                "max_depth": self.max_depth,
                "messages": [],
            }

            config = {"configurable": {"thread_id": self.thread_id}}

            # Stream graph execution
            async for event in self.graph.astream(
                input_state, config, stream_mode="updates"
            ):
                await self._handle_graph_event(event, chat)

        except Exception as e:
            logger.error(f"Graph execution error: {e}", exc_info=True)
            await chat.mount(ErrorDisplay(f"Error: {e}"))
            chat.scroll_end(animate=False)
        finally:
            if not self._awaiting_approval:
                self._is_streaming = False
            self._show_loading(False)
            self.query_one("#user-input").focus()

    @work(exclusive=True, thread=False)
    async def _handle_approval(self, user_response: str) -> None:
        """Resume the graph after user approval/rejection."""
        self._awaiting_approval = False
        self._is_streaming = True
        self._show_loading(True)
        chat = self.query_one("#chat-view")

        try:
            config = {"configurable": {"thread_id": self.thread_id}}

            async for event in self.graph.astream(
                Command(resume=user_response), config, stream_mode="updates"
            ):
                await self._handle_graph_event(event, chat)

        except Exception as e:
            logger.error(f"Graph resume error: {e}", exc_info=True)
            await chat.mount(ErrorDisplay(f"Error: {e}"))
            chat.scroll_end(animate=False)
        finally:
            self._is_streaming = False
            self._show_loading(False)
            self.query_one("#user-input").focus()

    async def _handle_graph_event(self, event: dict, chat) -> None:
        """Process a single graph streaming event and update the UI."""
        # stream_mode="updates" yields {node_name: state_update} dicts
        for node_name, update in event.items():
            if node_name == "__interrupt__":
                # The graph is paused at await_approval.
                # The task analysis was already shown by understand_request.
                await chat.mount(ApprovalPrompt(
                    "Type [bold]approve[/bold] to execute, or [bold]reject[/bold] to cancel."
                ))
                chat.scroll_end(animate=False)
                self._awaiting_approval = True
                return

            elif node_name == "understand_request":
                # Show the task analysis
                messages = update.get("messages", [])
                for msg in messages:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    await chat.mount(AssistantMessage(
                        f"[bold]Agent:[/bold] {content}"
                    ))
                    chat.scroll_end(animate=False)

            elif node_name == "await_approval":
                # Show approval status
                approved = update.get("execution_approved")
                if approved is not None:
                    status = "[green]Approved[/green]" if approved else "[red]Rejected[/red]"
                    await chat.mount(SystemMessage(f"Execution {status}"))
                    chat.scroll_end(animate=False)

            elif node_name == "agent_step":
                # Show agent's tool selection decision
                action = update.get("next_action", {})
                action_type = action.get("action", "unknown")
                reasoning = action.get("reasoning", "")

                if action_type == "use_tool":
                    tool = action.get("tool", "unknown")
                    await chat.mount(ToolCallDisplay(
                        f"[dim]Next tool:[/dim] [bold]{tool}[/bold] — {reasoning}"
                    ))
                elif action_type == "use_ffmpeg":
                    op = action.get("operation", "unknown")
                    await chat.mount(ToolCallDisplay(
                        f"[dim]FFmpeg:[/dim] [bold]{op}[/bold] — {reasoning}"
                    ))
                elif action_type == "use_kg":
                    op = action.get("operation", "unknown")
                    await chat.mount(ToolCallDisplay(
                        f"[dim]KG:[/dim] [bold]{op}[/bold] — {reasoning}"
                    ))
                elif action_type == "done":
                    await chat.mount(SystemMessage(
                        f"Agent finished: {reasoning}"
                    ))
                chat.scroll_end(animate=False)

            elif node_name == "execute_tool":
                # Show only new execution results
                tool_history = update.get("tool_history", [])
                new_entries = tool_history[self._tool_history_seen:]
                self._tool_history_seen = len(tool_history)
                for entry in new_entries:
                    tool = entry.get("tool", "unknown")
                    success = entry.get("success", False)
                    summary = entry.get("result_summary", "")
                    exec_time = entry.get("execution_time", 0)
                    status = "[green]OK[/green]" if success else "[red]FAILED[/red]"
                    await chat.mount(ToolCallDisplay(
                        f"[dim]Result:[/dim] [bold]{tool}[/bold] {status} "
                        f"[dim]({exec_time:.1f}s)[/dim]\n{summary}"
                    ))
                    chat.scroll_end(animate=False)

            elif node_name == "summarize_results":
                # Show the final summary
                summary = update.get("execution_summary", "")
                if summary:
                    await chat.mount(AssistantMessage(
                        f"[bold]Summary:[/bold]\n{summary}"
                    ))
                output_path = update.get("output_mmif_path", "")
                if output_path:
                    await chat.mount(SystemMessage(
                        f"Output saved: {output_path}"
                    ))
                chat.scroll_end(animate=False)

    # -----------------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------------

    def action_clear(self) -> None:
        """Clear the chat view."""
        chat = self.query_one("#chat-view")
        chat.remove_children()
        self.thread_id = f"tui-{uuid4().hex[:8]}"
        self._awaiting_approval = False
        chat.mount(SystemMessage(f"Chat cleared. New session: {self.thread_id}"))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CLAMS Agent TUI")
    parser.add_argument(
        "--gpu", action="store_true",
        help="Enable GPU-requiring tools",
    )
    parser.add_argument(
        "--video", type=str, default="",
        help="Path to video file to process",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10,
        help="Maximum tool-use iterations (default: 10)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model name override",
    )
    parser.add_argument(
        "--provider", type=str, choices=["ollama", "openai"],
        default=None, help="LLM provider override",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging to tui.log",
    )
    args = parser.parse_args()

    # Redirect logging to file so it doesn't interfere with Textual
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        filename="tui.log",
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Override config if flags provided
    if args.model or args.provider:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.get_config()
        if args.provider:
            config.llm.provider = args.provider
        if args.model:
            config.llm.model_name = args.model
        config_manager.save_config(config)

    # Import and create the graph
    from agent import create_pipeline_graph
    graph = create_pipeline_graph()

    app = CLAMSTui(
        graph=graph,
        gpu_available=args.gpu,
        video_path=args.video,
        max_depth=args.max_depth,
    )
    app.run()


if __name__ == "__main__":
    main()
