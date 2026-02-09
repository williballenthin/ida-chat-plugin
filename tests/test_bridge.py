"""Tests for the pi-mono bridge (Python <-> Node.js agent server).

Tests the bridge lifecycle, initialization, and basic communication
without requiring an LLM API key (where possible).
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pi_mono_bridge import PiMonoBridge, AgentEvent, AGENT_SERVER_SCRIPT


class TestBridgeSetup:
    """Verify agent server files exist and Node.js is available."""

    def test_agent_server_script_exists(self):
        assert AGENT_SERVER_SCRIPT.exists(), f"Missing: {AGENT_SERVER_SCRIPT}"

    def test_node_modules_installed(self):
        node_modules = AGENT_SERVER_SCRIPT.parent / "node_modules"
        assert node_modules.is_dir(), "Run: cd agent_server && npm install"

    def test_pi_ai_package_exists(self):
        pi_ai = AGENT_SERVER_SCRIPT.parent / "node_modules" / "@mariozechner" / "pi-ai"
        assert pi_ai.is_dir(), "@mariozechner/pi-ai not installed"

    def test_pi_agent_core_exists(self):
        core = AGENT_SERVER_SCRIPT.parent / "node_modules" / "@mariozechner" / "pi-agent-core"
        assert core.is_dir(), "@mariozechner/pi-agent-core not installed"

    def test_find_node(self):
        node = PiMonoBridge._find_node()
        assert node is not None
        assert os.path.isfile(node)


class TestAgentEvent:
    """Test AgentEvent data class."""

    def test_text_event(self):
        e = AgentEvent({"type": "text", "text": "hello"})
        assert e.type == "text"
        assert e.text == "hello"

    def test_tool_call_event(self):
        e = AgentEvent({
            "type": "tool_call",
            "toolCallId": "tc_123",
            "toolName": "ida_script",
            "args": {"code": "print(42)"},
        })
        assert e.type == "tool_call"
        assert e.tool_call_id == "tc_123"
        assert e.tool_name == "ida_script"
        assert e.args == {"code": "print(42)"}

    def test_done_event(self):
        e = AgentEvent({"type": "done", "turns": 3})
        assert e.type == "done"
        assert e.turns == 3

    def test_defaults(self):
        e = AgentEvent({"type": "unknown"})
        assert e.text == ""
        assert e.tool_call_id == ""
        assert e.turn == 0


@pytest.mark.asyncio
class TestBridgeLifecycle:
    """Test bridge start/stop without sending prompts."""

    async def test_start_and_stop(self):
        """Bridge can start the Node.js server and stop it cleanly."""
        bridge = PiMonoBridge()
        # Use a model that exists in the registry (won't make API calls yet)
        await bridge.start(
            system_prompt="test",
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )
        assert bridge.is_running
        await bridge.stop()
        assert not bridge.is_running

    async def test_double_start_raises(self):
        """Starting an already-running bridge raises RuntimeError."""
        bridge = PiMonoBridge()
        await bridge.start(
            system_prompt="test",
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await bridge.start(system_prompt="test", model="openrouter/meta-llama/llama-3.3-70b-instruct:free")
        finally:
            await bridge.stop()

    async def test_stop_idempotent(self):
        """Stopping a stopped bridge doesn't raise."""
        bridge = PiMonoBridge()
        await bridge.stop()  # Should not raise


@pytest.mark.asyncio
class TestBridgeWithLLM:
    """Tests that require an LLM API key (OpenRouter).

    These tests are skipped if OPENROUTER_API_KEY is not set.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    async def test_simple_prompt(self):
        """Send a simple prompt and receive a text response."""
        bridge = PiMonoBridge()
        await bridge.start(
            system_prompt="You are a helpful assistant. Respond briefly.",
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )
        try:
            events = []
            async for event in bridge.prompt("Say hello in exactly 3 words."):
                events.append(event)

            types = [e.type for e in events]
            assert "done" in types, f"Expected 'done' event, got: {types}"
            # Should have at least a text event
            text_events = [e for e in events if e.type == "text"]
            assert len(text_events) > 0, f"No text events received: {types}"
        finally:
            await bridge.stop()

    async def test_tool_call_flow(self, db):
        """Send a prompt that triggers a tool call and handle it."""
        from ida_codemode_sandbox import IdaSandbox

        sandbox = IdaSandbox(db)
        bridge = PiMonoBridge()
        await bridge.start(
            system_prompt=(
                "You are a binary analyst. "
                "When asked about a binary, ALWAYS use the ida_script tool to run analysis code. "
                "Available functions: enumerate_functions(), get_binary_info(). "
                "Use print() to output results."
            ),
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )
        try:
            events = []
            async for event in bridge.prompt("Use the ida_script tool to call get_binary_info() and print the architecture."):
                events.append(event)

                if event.type == "tool_call" and event.tool_name == "ida_script":
                    code = event.args.get("code", "")
                    output = sandbox.execute(code)
                    await bridge.send_tool_result(
                        event.tool_call_id,
                        output,
                        is_error=False,
                    )

            types = [e.type for e in events]
            assert "tool_call" in types, f"Expected tool_call event, got: {types}"
            assert "done" in types, f"Expected done event, got: {types}"
        finally:
            await bridge.stop()
