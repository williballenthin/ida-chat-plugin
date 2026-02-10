"""
IDA Chat Core (Pi-Mono) - Chat backend using embedded JS agent harness.

Drop-in replacement for IDAChatCore that uses pi-mono (via PythonMonkey)
instead of Claude Agent SDK. Supports multiple LLM providers via
OpenRouter and other OpenAI-compatible backends.

The agent runs entirely in-process: JS in SpiderMonkey for the agent
loop, Python for HTTP and tool execution. No subprocess, no Node.js.
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
    """Chat backend using embedded pi-mono agent harness.

    Implements the same interface as IDAChatCore but uses pi-mono for
    LLM communication instead of Claude Agent SDK.

    The agent loop runs synchronously in-process (JS in SpiderMonkey).
    Tool execution happens via Python callback during the JS agent loop.
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

    def connect(self) -> None:
        """Initialize and start the embedded agent harness."""
        logger.info("=" * 60)
        logger.info(f"Starting pi-mono agent (model={self.model})")

        system_prompt = _load_system_prompt()

        self._bridge = PiMonoBridge()
        self._bridge.start(
            system_prompt=system_prompt,
            model=self.model,
            tool_callback=self._execute_script,
        )

        logger.info("Pi-mono agent connected and ready")

    def disconnect(self) -> None:
        """Stop the agent harness."""
        if self._bridge:
            self._bridge.stop()
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

    def process_message(self, user_input: str) -> str:
        """Process a user message through the pi-mono agent.

        The entire agent loop runs synchronously. The JS harness calls
        the LLM (via Python HTTP) and executes tools (via Python callback)
        in a loop until the agent is done.

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

        # The entire agent loop happens in this single call.
        # Tool execution occurs inside the JS harness via the Python callback.
        events = self._bridge.prompt(user_input, self.max_turns)

        for event in events:
            if event.type == "thinking":
                self.callback.on_thinking()

            elif event.type == "turn_start":
                self.callback.on_turn_start(event.turn, self.max_turns)
                self.callback.on_thinking()

            elif event.type == "text":
                text = event.text
                if text.strip():
                    self.callback.on_thinking_done()
                    self.callback.on_text(text)
                    if self.history:
                        self.history.append_assistant_message(text)

            elif event.type == "tool_call":
                self.callback.on_thinking_done()
                code = event.code
                if code:
                    self.callback.on_script_code(code)

            elif event.type == "tool_result":
                output = event.output
                if output:
                    all_script_outputs.append(output)
                    self.callback.on_script_output(output)
                    if self.history:
                        # Find the matching tool_call code
                        self.history.append_script_execution("", output)

            elif event.type == "turn_end":
                logger.info(f"Turn {event.turn} complete")

            elif event.type == "done":
                logger.info(f"Agent done after {event.turns} turns")
                if self.verbose:
                    self.callback.on_result(event.turns, None)

            elif event.type == "error":
                logger.error(f"Agent error: {event.message}")
                self.callback.on_error(event.message)

        return "\n".join(all_script_outputs) if all_script_outputs else ""
