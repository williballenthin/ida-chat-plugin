"""End-to-end tests for the IDA Chat session using pi-mono.

Tests the full flow: start agent -> send message -> receive response
with tool execution through the codemode sandbox.

These tests demonstrate:
1. Invoking IDA with a database
2. Sending messages to a chat session
3. Fetching all messages from the chat session
4. Tool execution through the codemode sandbox
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ida_chat_core_pi import IDAChatCorePi
from ida_codemode_sandbox import IdaSandbox


class CollectingCallback:
    """Test callback that collects all events for verification."""

    def __init__(self):
        self.events: list[dict] = []
        self.texts: list[str] = []
        self.scripts: list[str] = []
        self.script_outputs: list[str] = []
        self.errors: list[str] = []
        self.turns: list[int] = []
        self.thinking_count = 0
        self.result_turns = 0
        self.result_cost = None

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        self.turns.append(turn)
        self.events.append({"type": "turn_start", "turn": turn, "max_turns": max_turns})

    def on_thinking(self) -> None:
        self.thinking_count += 1
        self.events.append({"type": "thinking"})

    def on_thinking_done(self) -> None:
        self.events.append({"type": "thinking_done"})

    def on_tool_use(self, tool_name: str, details: str) -> None:
        self.events.append({"type": "tool_use", "tool_name": tool_name, "details": details})

    def on_text(self, text: str) -> None:
        self.texts.append(text)
        self.events.append({"type": "text", "text": text})

    def on_script_code(self, code: str) -> None:
        self.scripts.append(code)
        self.events.append({"type": "script_code", "code": code})

    def on_script_output(self, output: str) -> None:
        self.script_outputs.append(output)
        self.events.append({"type": "script_output", "output": output})

    def on_error(self, error: str) -> None:
        self.errors.append(error)
        self.events.append({"type": "error", "error": error})

    def on_result(self, num_turns: int, cost: float | None) -> None:
        self.result_turns = num_turns
        self.result_cost = cost
        self.events.append({"type": "result", "turns": num_turns, "cost": cost})

    def get_all_messages(self) -> list[dict]:
        """Return all collected events as a message list."""
        return self.events

    def get_full_text(self) -> str:
        """Return all assistant text concatenated."""
        return "\n".join(self.texts)


@pytest.mark.asyncio
class TestChatSessionWithLLM:
    """Full end-to-end chat session tests.

    Requires OPENROUTER_API_KEY to be set.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    async def test_send_message_get_response(self, db):
        """Send a message to the chat session and verify we get a response."""
        callback = CollectingCallback()
        sandbox = IdaSandbox(db)

        core = IDAChatCorePi(
            db=db,
            callback=callback,
            script_executor=sandbox.execute,
            verbose=True,
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )

        await core.connect()
        try:
            result = await core.process_message(
                "What architecture is this binary? Use the ida_script tool to find out."
            )

            # Verify we got events
            messages = callback.get_all_messages()
            assert len(messages) > 0, "No events received"

            # Should have received text response
            assert len(callback.texts) > 0, f"No text response. Events: {[e['type'] for e in messages]}"

            # The agent should have called the ida_script tool
            assert len(callback.scripts) > 0, f"No scripts executed. Events: {[e['type'] for e in messages]}"

            # Script output should contain architecture info
            all_output = "\n".join(callback.script_outputs)
            assert "metapc" in all_output.lower() or "x86" in all_output.lower() or "32" in all_output, \
                f"Expected architecture info in output: {all_output}"

        finally:
            await core.disconnect()

    async def test_fetch_all_messages(self, db):
        """Send multiple messages and verify all are collected."""
        callback = CollectingCallback()
        sandbox = IdaSandbox(db)

        core = IDAChatCorePi(
            db=db,
            callback=callback,
            script_executor=sandbox.execute,
            verbose=True,
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )

        await core.connect()
        try:
            # Send first message
            await core.process_message(
                "Use the ida_script tool to call get_binary_info() and print the md5 hash."
            )
            first_batch = list(callback.events)

            # Send second message
            await core.process_message(
                "Use the ida_script tool to call enumerate_functions() and print how many functions there are."
            )

            # Fetch all messages
            all_messages = callback.get_all_messages()
            assert len(all_messages) > len(first_batch), \
                "Second message should produce additional events"

            # Both prompts should have generated text
            assert len(callback.texts) >= 2, \
                f"Expected at least 2 text responses, got {len(callback.texts)}"

        finally:
            await core.disconnect()

    async def test_script_error_recovery(self, db):
        """Agent should handle script errors and continue."""
        callback = CollectingCallback()
        sandbox = IdaSandbox(db)

        core = IDAChatCorePi(
            db=db,
            callback=callback,
            script_executor=sandbox.execute,
            verbose=True,
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )

        await core.connect()
        try:
            # Ask agent to do something - the sandbox will handle errors gracefully
            await core.process_message(
                "Use the ida_script tool to print the number of functions in this binary."
            )

            # Should have at least one text response
            assert len(callback.texts) > 0 or len(callback.errors) > 0, \
                "Expected either text or error events"

        finally:
            await core.disconnect()


@pytest.mark.asyncio
class TestChatSessionWithoutLLM:
    """Tests that verify the chat session setup without requiring an LLM.

    These test the wiring between components (core, sandbox, callback)
    without making actual API calls.
    """

    def test_callback_protocol(self):
        """CollectingCallback implements the ChatCallback protocol."""
        cb = CollectingCallback()
        cb.on_turn_start(1, 20)
        cb.on_thinking()
        cb.on_thinking_done()
        cb.on_text("hello")
        cb.on_script_code("print(42)")
        cb.on_script_output("42")
        cb.on_error("test error")
        cb.on_result(1, 0.001)

        assert cb.turns == [1]
        assert cb.thinking_count == 1
        assert cb.texts == ["hello"]
        assert cb.scripts == ["print(42)"]
        assert cb.script_outputs == ["42"]
        assert cb.errors == ["test error"]
        assert cb.result_turns == 1

    def test_core_creation(self, db):
        """IDAChatCorePi can be created with sandbox executor."""
        sandbox = IdaSandbox(db)
        callback = CollectingCallback()

        core = IDAChatCorePi(
            db=db,
            callback=callback,
            script_executor=sandbox.execute,
        )

        assert core.db is db
        assert core.model == "openrouter/meta-llama/llama-3.3-70b-instruct:free"
        assert core.max_turns == 20

    def test_core_custom_model(self, db):
        """IDAChatCorePi respects custom model setting."""
        sandbox = IdaSandbox(db)
        callback = CollectingCallback()

        core = IDAChatCorePi(
            db=db,
            callback=callback,
            script_executor=sandbox.execute,
            model="openrouter/google/gemma-3-27b-it:free",
        )

        assert core.model == "openrouter/google/gemma-3-27b-it:free"

    def test_sandbox_executor_works(self, db):
        """Verify sandbox.execute() works as script_executor."""
        sandbox = IdaSandbox(db)
        result = sandbox.execute(
            'info = get_binary_info()\nprint(info["architecture"])'
        )
        assert "metapc" in result

    async def test_core_not_connected_raises(self, db):
        """Calling process_message before connect() raises RuntimeError."""
        sandbox = IdaSandbox(db)
        callback = CollectingCallback()
        core = IDAChatCorePi(db=db, callback=callback, script_executor=sandbox.execute)

        with pytest.raises(RuntimeError, match="Not connected"):
            await core.process_message("hello")
