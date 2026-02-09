"""
IDA Chat Core (Pi-Mono) - Chat backend using pi-mono agent harness.

Drop-in replacement for IDAChatCore that uses pi-mono instead of
Claude Agent SDK. Supports multiple LLM providers via OpenRouter
and other pi-mono backends.
"""

import logging
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from pi_mono_bridge import PiMonoBridge

if TYPE_CHECKING:
    from ida_chat_history import MessageHistory

logger = logging.getLogger("ida-chat.pi")

# Project directory (contains PROMPT.md, USAGE.md, API_REFERENCE.md)
PROJECT_DIR = Path(__file__).parent.resolve() / "project"

PROMPT_FILE = PROJECT_DIR / "PROMPT.md"
IDA_UI_FILE = PROJECT_DIR / "IDA.md"
USAGE_FILE = PROJECT_DIR / "USAGE.md"
API_REFERENCE_FILE = PROJECT_DIR / "API_REFERENCE.md"

# Default model for pi-mono
DEFAULT_MODEL = "openrouter/meta-llama/llama-3.3-70b-instruct:free"


def _load_system_prompt() -> str:
    """Load and compose the system prompt from project files."""
    prompt = ""

    if PROMPT_FILE.exists():
        prompt = PROMPT_FILE.read_text()
    else:
        logger.warning(f"PROMPT.md not found at {PROMPT_FILE}")
        prompt = (
            "You are a binary analysis assistant with access to an IDA Pro database. "
            "Use the ida_script tool to execute analysis code."
        )

    if os.environ.get("IDA_CHAT_INSIDE_IDA") == "1":
        if IDA_UI_FILE.exists():
            prompt += "\n\n" + IDA_UI_FILE.read_text()

    if USAGE_FILE.exists():
        prompt += "\n\n" + USAGE_FILE.read_text()

    if API_REFERENCE_FILE.exists():
        prompt += "\n\n" + API_REFERENCE_FILE.read_text()

    # Adapt prompt for tool-based execution (no <idascript> tags needed)
    prompt += "\n\n## Tool Usage\n\n"
    prompt += (
        "Use the `ida_script` tool to execute Python analysis code against the database. "
        "Do NOT use <idascript> XML tags. Instead, call the ida_script tool with your code. "
        "The tool will return the script's stdout output or error message."
    )

    return prompt


class IDAChatCorePi:
    """Chat backend using pi-mono agent harness.

    Implements the same interface as IDAChatCore but uses pi-mono for
    LLM communication instead of Claude Agent SDK.
    """

    def __init__(
        self,
        db,
        callback,
        script_executor: Callable[[str], str] | None = None,
        verbose: bool = False,
        max_turns: int = 20,
        history: "MessageHistory | None" = None,
        model: str | None = None,
    ):
        """Initialize the pi-mono chat core.

        Args:
            db: An open ida_domain Database instance.
            callback: Handler for output events (ChatCallback protocol).
            script_executor: Optional custom script executor. If None, uses
                default direct execution.
            verbose: If True, report additional stats.
            max_turns: Maximum agentic turns before stopping.
            history: Optional MessageHistory for persisting conversations.
            model: LLM model identifier (e.g. "openrouter/meta-llama/llama-3.3-70b-instruct:free").
        """
        self.db = db
        self.callback = callback
        self.verbose = verbose
        self.max_turns = max_turns
        self.history = history
        self.model = model or os.environ.get("IDA_CHAT_MODEL", DEFAULT_MODEL)
        self._bridge: PiMonoBridge | None = None
        self._cancelled = False
        self._execute_script = script_executor or self._default_execute_script

    def request_cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True
        logger.info("Cancel requested")

    async def connect(self) -> None:
        """Initialize and start the pi-mono agent server."""
        logger.info("=" * 60)
        logger.info(f"Starting pi-mono agent (model={self.model})")

        system_prompt = _load_system_prompt()

        self._bridge = PiMonoBridge()
        await self._bridge.start(
            system_prompt=system_prompt,
            model=self.model,
        )

        logger.info("Pi-mono agent connected and ready")

    async def disconnect(self) -> None:
        """Stop the agent server."""
        if self._bridge:
            await self._bridge.stop()
            self._bridge = None

    def _default_execute_script(self, code: str) -> str:
        """Default script executor using direct exec().

        Args:
            code: Python code to execute with `db` in scope.

        Returns:
            Captured stdout output or error message.
        """
        old_stdout = sys.stdout
        sys.stdout = captured = StringIO()
        try:
            exec(code, {"db": self.db, "print": print})
            return captured.getvalue()
        except Exception as e:
            return f"Script error: {e}"
        finally:
            sys.stdout = old_stdout

    async def process_message(self, user_input: str) -> str:
        """Process a user message through the pi-mono agent.

        The agent loop is handled by pi-mono. When the agent calls the
        ida_script tool, we execute the code locally and return the result.

        Args:
            user_input: The user's message/query.

        Returns:
            Combined script outputs as a string.
        """
        if not self._bridge:
            raise RuntimeError("Not connected. Call connect() first.")

        logger.info("-" * 60)
        logger.info(f"USER MESSAGE: {user_input[:200]}...")

        if self.history:
            self.history.append_user_message(user_input)

        self._cancelled = False
        all_script_outputs: list[str] = []
        current_turn = 0

        async for event in self._bridge.prompt(user_input):
            if self._cancelled:
                logger.info("Operation cancelled by user")
                self.callback.on_error("Operation cancelled")
                break

            if event.type == "thinking":
                self.callback.on_thinking()

            elif event.type == "turn_start":
                current_turn = event.turn
                self.callback.on_turn_start(event.turn, self.max_turns)
                self.callback.on_thinking()

            elif event.type == "text":
                text = event.text
                if text.strip():
                    self.callback.on_thinking_done()
                    self.callback.on_text(text)
                    if self.history:
                        self.history.append_assistant_message(text)

            elif event.type == "text_delta":
                # Streaming text delta - could be used for real-time display
                pass

            elif event.type == "tool_call":
                self.callback.on_thinking_done()

                if event.tool_name == "ida_script":
                    code = event.args.get("code", "")
                    logger.info(f"Executing ida_script ({len(code)} chars)")

                    self.callback.on_script_code(code)

                    # Execute the script
                    try:
                        output = self._execute_script(code)
                        is_error = False
                    except Exception as e:
                        output = f"Script error: {e}"
                        is_error = True

                    all_script_outputs.append(output)

                    if output:
                        self.callback.on_script_output(output)

                    if self.history:
                        self.history.append_script_execution(code, output)

                    # Send result back to agent
                    await self._bridge.send_tool_result(
                        event.tool_call_id,
                        output,
                        is_error=is_error,
                    )
                else:
                    # Unknown tool - send error back
                    await self._bridge.send_tool_result(
                        event.tool_call_id,
                        f"Unknown tool: {event.tool_name}",
                        is_error=True,
                    )

            elif event.type == "turn_end":
                logger.info(f"Turn {event.turn} complete")

            elif event.type == "done":
                logger.info(f"Agent done after {event.turns} turns")
                if self.verbose:
                    self.callback.on_result(event.turns, None)
                break

            elif event.type == "error":
                logger.error(f"Agent error: {event.message}")
                self.callback.on_error(event.message)
                break

        return "\n".join(all_script_outputs) if all_script_outputs else ""
