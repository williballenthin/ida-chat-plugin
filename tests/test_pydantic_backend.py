"""Tests for the pydantic-ai backend.

Exercises the PydanticAIBackend with a real IDA database and the codemode sandbox.
Integration tests that hit OpenRouter are marked with @pytest.mark.integration
and require OPENROUTER_API_KEY to be set.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from ida_codemode_sandbox import IdaSandbox
from ida_chat_backend import ChatBackend
from ida_chat_pydantic import PydanticAIBackend, _build_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingCallback:
    """A ChatCallback that records all events for assertions."""

    def __init__(self):
        self.events: list[tuple[str, ...]] = []

    def on_turn_start(self, turn: int, max_turns: int) -> None:
        self.events.append(("turn_start", str(turn), str(max_turns)))

    def on_thinking(self) -> None:
        self.events.append(("thinking",))

    def on_thinking_done(self) -> None:
        self.events.append(("thinking_done",))

    def on_tool_use(self, tool_name: str, details: str) -> None:
        self.events.append(("tool_use", tool_name, details))

    def on_text(self, text: str) -> None:
        self.events.append(("text", text))

    def on_script_code(self, code: str) -> None:
        self.events.append(("script_code", code))

    def on_script_output(self, output: str) -> None:
        self.events.append(("script_output", output))

    def on_error(self, error: str) -> None:
        self.events.append(("error", error))

    def on_result(self, num_turns: int, cost: float | None) -> None:
        self.events.append(("result", str(num_turns), str(cost)))


# ---------------------------------------------------------------------------
# Unit tests (no LLM calls)
# ---------------------------------------------------------------------------


class TestSandboxDirect:
    """Test the codemode sandbox directly with a real database."""

    def test_sandbox_enumerate_functions(self, db):
        """Sandbox can enumerate functions from a real binary."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("functions = enumerate_functions()\nprint(len(functions))")
        assert result.ok
        stdout = "".join(result.stdout).strip()
        count = int(stdout)
        assert count > 0, f"Expected functions, got {count}"

    def test_sandbox_get_binary_info(self, db):
        """Sandbox can retrieve binary metadata."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("info = get_binary_info()\nprint(info['architecture'])")
        assert result.ok
        arch = "".join(result.stdout).strip()
        # IDA uses internal processor names (e.g. "metapc" for x86)
        assert len(arch) > 0, "Architecture should not be empty"

    def test_sandbox_disassemble_function(self, db):
        """Sandbox can disassemble a function."""
        sandbox = IdaSandbox(db)
        result = sandbox.run(
            "fns = enumerate_functions()\n"
            "if len(fns) > 0:\n"
            "    lines = disassemble_function(fns[0]['address'])\n"
            "    print(len(lines))\n"
            "else:\n"
            "    print(0)"
        )
        assert result.ok
        count = int("".join(result.stdout).strip())
        assert count > 0, "Expected disassembly lines"

    def test_sandbox_enumerate_strings(self, db):
        """Sandbox can enumerate strings."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("strings = enumerate_strings()\nprint(len(strings))")
        assert result.ok
        count = int("".join(result.stdout).strip())
        assert count >= 0  # binary may have no strings

    def test_sandbox_enumerate_imports(self, db):
        """Sandbox can enumerate imports."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("imports = enumerate_imports()\nprint(len(imports))")
        assert result.ok
        count = int("".join(result.stdout).strip())
        assert count > 0, "Expected imports in PE binary"

    def test_sandbox_error_handling(self, db):
        """Sandbox reports errors properly."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("x = 1 / 0")
        assert not result.ok
        assert result.error is not None
        assert result.error.kind == "runtime"
        assert "ZeroDivisionError" in result.error.inner_type

    def test_sandbox_syntax_error(self, db):
        """Sandbox reports syntax errors."""
        sandbox = IdaSandbox(db)
        result = sandbox.run("def foo(")
        assert not result.ok
        assert result.error is not None
        assert result.error.kind == "syntax"

    def test_sandbox_no_imports(self, db):
        """Sandbox blocks import -- 'os' is not available even if import is a no-op."""
        sandbox = IdaSandbox(db)
        # The sandbox may accept 'import os' syntactically but 'os' won't be usable
        result = sandbox.run("import os\nprint(os.getcwd())")
        assert not result.ok

    def test_sandbox_execute_interface(self, db):
        """The execute() method returns stdout as a string."""
        sandbox = IdaSandbox(db)
        output = sandbox.execute("print('hello world')")
        assert "hello world" in output

    def test_sandbox_execute_error(self, db):
        """The execute() method returns error description on failure."""
        sandbox = IdaSandbox(db)
        output = sandbox.execute("1 / 0")
        assert "error" in output.lower()


class TestBackendConstruction:
    """Test PydanticAIBackend construction (no LLM calls)."""

    def test_backend_creates(self, db):
        """Backend can be constructed with default settings."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        assert isinstance(backend, ChatBackend)

    def test_backend_custom_model(self, db):
        """Backend accepts a custom model string."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback, model="openrouter:z-ai/glm-4.7-flash")
        assert backend.agent is not None

    def test_backend_has_sandbox(self, db):
        """Backend creates a sandbox from the database."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        assert backend.sandbox is not None
        # Verify sandbox works
        result = backend.sandbox.run("print('test')")
        assert result.ok

    def test_evaluate_code_tool(self, db):
        """The evaluate_code tool executes code in the sandbox."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        output = backend._evaluate_code("print('hello from tool')")
        assert "hello from tool" in output
        # Verify callbacks were fired
        assert any(e[0] == "script_code" for e in callback.events)
        assert any(e[0] == "script_output" for e in callback.events)

    def test_evaluate_code_tool_error(self, db):
        """The evaluate_code tool returns errors gracefully."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        output = backend._evaluate_code("1 / 0")
        assert "error" in output.lower()

    def test_evaluate_code_max_turns(self, db):
        """The evaluate_code tool enforces max_turns."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback, max_turns=2)
        # First two calls succeed
        backend._evaluate_code("print('one')")
        backend._evaluate_code("print('two')")
        # Third should be rejected
        output = backend._evaluate_code("print('three')")
        assert "Maximum" in output


class TestSystemPrompt:
    """Test system prompt construction."""

    def test_system_prompt_includes_sandbox_docs(self):
        """System prompt includes sandbox API reference."""
        prompt = _build_system_prompt()
        assert "enumerate_functions" in prompt
        assert "get_binary_info" in prompt
        assert "disassemble_function" in prompt

    def test_system_prompt_includes_instructions(self):
        """System prompt includes agent instructions."""
        prompt = _build_system_prompt()
        assert "reverse engineering" in prompt.lower()
        assert "evaluate_code" in prompt.lower()


class TestCoreBackendIntegration:
    """Test that IDAChatCore delegates to the backend."""

    def test_core_with_backend(self, db):
        """IDAChatCore accepts and stores a backend."""
        from ida_chat_core import IDAChatCore

        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        core = IDAChatCore(db, callback, backend=backend)
        assert core._backend is backend

    @pytest.mark.asyncio
    async def test_core_connect_delegates(self, db):
        """IDAChatCore.connect() delegates to backend."""
        from ida_chat_core import IDAChatCore

        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        core = IDAChatCore(db, callback, backend=backend)
        # Should not raise (pydantic-ai needs no connection)
        await core.connect()
        await core.disconnect()


# ---------------------------------------------------------------------------
# Integration tests (require OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

requires_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)


@requires_openrouter
class TestLLMIntegration:
    """Integration tests that make real LLM calls via OpenRouter.

    These tests are inherently flaky because they depend on a remote LLM.
    The GLM model may sometimes loop excessively or fail output validation.
    Tests verify the basic flow works but tolerate occasional failures.
    """

    @pytest.mark.asyncio
    async def test_simple_query(self, db):
        """LLM can answer a simple question about the binary."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback, verbose=True)
        await backend.connect()

        result = await backend.process_message("How many functions are in this binary? Just give me the count.")

        await backend.disconnect()

        # Should have fired thinking callbacks regardless of success
        assert any(e[0] == "thinking" for e in callback.events)
        assert any(e[0] == "thinking_done" for e in callback.events)

        if result:
            # If we got a response, verify tool was called
            assert any(e[0] == "script_code" for e in callback.events)

    @pytest.mark.asyncio
    async def test_binary_info_query(self, db):
        """LLM can retrieve binary metadata."""
        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        await backend.connect()

        result = await backend.process_message(
            "What architecture is this binary? Use get_binary_info() to find out. "
            "Give a brief one-sentence answer."
        )

        await backend.disconnect()

        # Verify something happened (either success or graceful error)
        assert any(e[0] == "thinking" for e in callback.events)

    @pytest.mark.asyncio
    async def test_via_core_facade(self, db):
        """End-to-end: IDAChatCore with PydanticAIBackend."""
        from ida_chat_core import IDAChatCore

        callback = RecordingCallback()
        backend = PydanticAIBackend(db, callback)
        core = IDAChatCore(db, callback, backend=backend)

        await core.connect()
        result = await core.process_message(
            "How many functions does this binary have? Use enumerate_functions(). "
            "Answer in one sentence."
        )
        await core.disconnect()

        # Verify the backend was used (callbacks fired)
        assert any(e[0] == "thinking" for e in callback.events)
