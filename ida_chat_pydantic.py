"""
Pydantic AI backend for IDA Chat.

Uses pydantic-ai with OpenRouter (or any supported provider) to provide
LLM-powered binary analysis with the codemode sandbox for safe script execution.

The LLM is given a single tool -- evaluate_code -- that runs Python scripts
in the sandbox with 28 pre-built IDA analysis functions available as globals.
"""

import logging
from typing import TYPE_CHECKING

from pydantic_ai import Agent, Tool
from pydantic_ai.usage import UsageLimits

from ida_codemode_sandbox import IdaSandbox
from ida_chat_backend import ChatBackend

if TYPE_CHECKING:
    from ida_chat_core import ChatCallback
    from ida_chat_history import MessageHistory

logger = logging.getLogger("ida-chat")

DEFAULT_MODEL = "openrouter:z-ai/glm-4.7-flash"


def _build_system_prompt() -> str:
    """Build the system prompt for the pydantic-ai agent."""
    return (
        "You are a reverse engineering assistant helping analyze a binary in IDA Pro.\n\n"
        "When you need to query or analyze the binary, use the evaluate_code tool to run "
        "Python scripts in a sandboxed environment. The sandbox has 28 pre-built analysis "
        "functions available as globals -- just call them directly (no imports needed).\n\n"
        "WORKFLOW:\n"
        "1. Read the user's question\n"
        "2. Call evaluate_code ONE TIME with a focused script to gather the data you need\n"
        "3. Read the output\n"
        "4. If you need more data, call evaluate_code again (but try to minimize calls)\n"
        "5. When you have enough data, STOP calling tools and write your final text answer\n\n"
        "IMPORTANT: Keep the number of tool calls minimal (1-3 is ideal). "
        "Write comprehensive scripts that gather all needed data in a single call rather than "
        "making many small calls. When you have the information you need, respond with text only.\n\n"
        + IdaSandbox.system_prompt()
    )


class PydanticAIBackend(ChatBackend):
    """Chat backend using pydantic-ai with the codemode sandbox.

    Usage::

        from ida_chat_pydantic import PydanticAIBackend

        backend = PydanticAIBackend(db, callback, model="openrouter:z-ai/glm-4.7-flash")
        core = IDAChatCore(db, callback, backend=backend)
        await core.connect()
        await core.process_message("list functions")

    Args:
        db: An open ``ida_domain.Database`` instance.
        callback: A ``ChatCallback`` for UI event delivery.
        model: pydantic-ai model string (default: OpenRouter GLM-4.7 Flash).
        max_turns: Maximum tool calls before forcing a final response.
        history: Optional ``MessageHistory`` for persistence.
        verbose: If True, report additional stats via callback.
    """

    def __init__(
        self,
        db,
        callback: "ChatCallback",
        *,
        model: str = DEFAULT_MODEL,
        max_turns: int = 20,
        history: "MessageHistory | None" = None,
        verbose: bool = False,
    ):
        self.db = db
        self.callback = callback
        self.max_turns = max_turns
        self.history = history
        self.verbose = verbose
        self._cancelled = False
        self._tool_calls = 0

        # Create sandbox for script execution
        self.sandbox = IdaSandbox(db)

        # Build pydantic-ai agent with a single evaluate_code tool.
        # We pass the tool via the Tool() wrapper so the function can
        # reference `self` via closure.
        self.agent = Agent(
            model,
            instructions=_build_system_prompt(),
            tools=[Tool(self._evaluate_code, takes_ctx=False)],
        )

        # Conversation history for multi-turn (pydantic-ai message format)
        self._message_history: list = []

    def _evaluate_code(self, code: str) -> str:
        """Execute Python code in the IDA analysis sandbox.

        The sandbox provides 28 pre-built analysis functions as globals
        (e.g. enumerate_functions, disassemble_function, get_callers, etc.).
        Use print() to output results. Scripts cannot import modules,
        access the filesystem, or make network calls.

        Args:
            code: Python source code to execute in the sandbox.

        Returns:
            The captured stdout output, or an error message if execution failed.
        """
        self._tool_calls += 1
        if self._tool_calls > self.max_turns:
            return "Maximum number of tool calls reached. Please provide your final analysis."

        self.callback.on_script_code(code)

        output = self.sandbox.execute(code)

        if output:
            self.callback.on_script_output(output)

        # Log to history
        if self.history:
            self.history.append_script_execution(code, output)

        return output or "Script executed successfully with no output."

    async def connect(self) -> None:
        """No persistent connection needed for pydantic-ai."""
        logger.info("PydanticAI backend ready (no persistent connection needed)")

    async def disconnect(self) -> None:
        """No cleanup needed for pydantic-ai."""
        logger.info("PydanticAI backend disconnected")

    def request_cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancelled = True
        logger.info("Cancel requested")

    async def process_message(self, user_input: str) -> str:
        """Process a user message via pydantic-ai's agentic loop.

        pydantic-ai handles the tool loop natively: the agent calls
        evaluate_code as needed, sees results, and continues until it
        produces a final text response.

        Args:
            user_input: The user's message.

        Returns:
            The agent's final text response.
        """
        self._cancelled = False
        self._tool_calls = 0

        logger.info(f"USER MESSAGE: {user_input[:200]}...")

        if self.history:
            self.history.append_user_message(user_input)

        self.callback.on_turn_start(1, self.max_turns)
        self.callback.on_thinking()

        try:
            # Limit requests to prevent runaway loops
            usage_limits = UsageLimits(request_limit=self.max_turns * 3)
            result = await self.agent.run(
                user_input,
                message_history=self._message_history,
                usage_limits=usage_limits,
            )

            self.callback.on_thinking_done()

            output_text = result.output if isinstance(result.output, str) else str(result.output)

            if output_text:
                self.callback.on_text(output_text)
                if self.history:
                    self.history.append_assistant_message(output_text)

            # Save history for next turn
            self._message_history = result.all_messages()

            if self.verbose:
                self.callback.on_result(self._tool_calls, None)

            logger.info(f"Response ({self._tool_calls} tool calls): {output_text[:200]}...")
            return output_text

        except Exception as e:
            logger.error(f"PydanticAI error: {e}", exc_info=True)
            self.callback.on_thinking_done()
            self.callback.on_error(str(e))
            return ""
