"""Tests for the pi-mono bridge (embedded JS via PythonMonkey).

Tests the bridge lifecycle, initialization, and communication
without requiring an LLM API key (where possible).
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pi_mono_bridge import PiMonoBridge, AgentEvent, AGENT_HARNESS


class TestBridgeSetup:
    """Verify agent harness files exist and PythonMonkey is available."""

    def test_agent_harness_exists(self):
        assert AGENT_HARNESS.exists(), f"Missing: {AGENT_HARNESS}"

    def test_pythonmonkey_importable(self):
        import pythonmonkey as pm
        assert pm is not None

    def test_harness_loadable(self):
        import pythonmonkey as pm
        harness = pm.require(str(AGENT_HARNESS))
        assert harness is not None
        assert harness.init is not None
        assert harness.prompt is not None


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
            "code": "print(42)",
        })
        assert e.type == "tool_call"
        assert e.tool_call_id == "tc_123"
        assert e.tool_name == "ida_script"
        assert e.code == "print(42)"

    def test_done_event(self):
        e = AgentEvent({"type": "done", "turns": 3})
        assert e.type == "done"
        assert e.turns == 3

    def test_defaults(self):
        e = AgentEvent({"type": "unknown"})
        assert e.text == ""
        assert e.tool_call_id == ""
        assert e.turn == 0


class TestBridgeLifecycle:
    """Test bridge start/stop."""

    def test_start_and_stop(self):
        """Bridge can start and stop cleanly."""
        bridge = PiMonoBridge()
        bridge.start(
            system_prompt="test",
            model="openrouter/test-model",
            tool_callback=lambda code: "mock",
            api_key="fake-key",
        )
        assert bridge.is_running
        bridge.stop()
        assert not bridge.is_running

    def test_double_start_raises(self):
        """Starting an already-running bridge raises RuntimeError."""
        bridge = PiMonoBridge()
        bridge.start(system_prompt="test", model="openrouter/test", api_key="fake")
        try:
            with pytest.raises(RuntimeError, match="already running"):
                bridge.start(system_prompt="test", model="openrouter/test", api_key="fake")
        finally:
            bridge.stop()

    def test_stop_idempotent(self):
        """Stopping a stopped bridge doesn't raise."""
        bridge = PiMonoBridge()
        bridge.stop()  # Should not raise


class TestBridgeWithMockLLM:
    """Test bridge with mock fetch (no real LLM calls)."""

    def test_simple_prompt_mock(self):
        """Bridge processes a prompt with mocked LLM response."""
        def mock_fetch(url, method, headers_json, body):
            return json.dumps({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "Hello from mock!",
                        "tool_calls": None,
                    }
                }]
            })

        bridge = PiMonoBridge()
        # Manually wire in mock fetch
        import pythonmonkey as pm
        bridge._harness = pm.require(str(AGENT_HARNESS))
        bridge._harness.init("test", "openrouter/test", lambda c: "", mock_fetch, "fake")
        bridge._running = True

        try:
            events = bridge.prompt("Hello")
            types = [e.type for e in events]
            assert "done" in types
            text_events = [e for e in events if e.type == "text"]
            assert len(text_events) > 0
            assert "Hello from mock!" in text_events[0].text
        finally:
            bridge.stop()

    def test_tool_call_mock(self):
        """Bridge handles tool calls with mocked LLM."""
        call_count = [0]

        def mock_fetch(url, method, headers_json, body):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "ida_script",
                                    "arguments": json.dumps({"code": "print(42)"})
                                }
                            }]
                        }
                    }]
                })
            else:
                return json.dumps({
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "The answer is 42.",
                            "tool_calls": None,
                        }
                    }]
                })

        def mock_tool(code):
            return "42\n"

        import pythonmonkey as pm
        bridge = PiMonoBridge()
        bridge._harness = pm.require(str(AGENT_HARNESS))
        bridge._harness.init("test", "openrouter/test", mock_tool, mock_fetch, "fake")
        bridge._running = True

        try:
            events = bridge.prompt("What is 42?")
            types = [e.type for e in events]
            assert "tool_call" in types
            assert "tool_result" in types
            assert "text" in types
            assert "done" in types
            assert call_count[0] == 2
        finally:
            bridge.stop()


class TestBridgeWithLLM:
    """Tests that require an LLM API key (OpenRouter).

    These tests are skipped if OPENROUTER_API_KEY is not set.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY not set")

    def test_simple_prompt(self):
        """Send a simple prompt and receive a text response."""
        bridge = PiMonoBridge()
        bridge.start(
            system_prompt="You are a helpful assistant. Respond briefly.",
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        )
        try:
            events = bridge.prompt("Say hello in exactly 3 words.")
            types = [e.type for e in events]
            assert "done" in types
            text_events = [e for e in events if e.type == "text"]
            assert len(text_events) > 0
        finally:
            bridge.stop()

    def test_tool_call_flow(self, db):
        """Send a prompt that triggers a tool call and handle it."""
        from ida_codemode_sandbox import IdaSandbox

        sandbox = IdaSandbox(db)
        bridge = PiMonoBridge()
        bridge.start(
            system_prompt=(
                "You are a binary analyst. "
                "When asked about a binary, ALWAYS use the ida_script tool to run analysis code. "
                "Available functions: enumerate_functions(), get_binary_info(). "
                "Use print() to output results."
            ),
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
            tool_callback=sandbox.execute,
        )
        try:
            events = bridge.prompt("Use the ida_script tool to call get_binary_info() and print the architecture.")
            types = [e.type for e in events]
            assert "tool_call" in types
            assert "done" in types
        finally:
            bridge.stop()
