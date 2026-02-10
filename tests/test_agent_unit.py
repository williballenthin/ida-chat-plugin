"""Layer 1 tests: PydanticAI agent creation and OpenRouter connectivity.

These tests verify:
- Agent creation with various configurations
- OpenRouter model connectivity
- Tool registration
- Conversation history management
- IDAAgent wrapper interface

No IDA Pro required. Uses mocks for sandbox where needed.
"""

from unittest.mock import MagicMock

import pytest

from ida_chat_agent import (
    AgentResponse,
    IDAAgent,
    SandboxDeps,
    create_agent,
)


class TestCreateAgent:
    """Agent factory function."""

    def test_returns_agent(self):
        agent = create_agent()
        assert agent is not None

    def test_agent_has_deps_type(self):
        agent = create_agent()
        assert agent._deps_type is SandboxDeps

    def test_custom_model(self):
        agent = create_agent("google/gemma-3-4b-it:free")
        assert agent is not None

    def test_custom_system_prompt(self):
        agent = create_agent(system_prompt="You are a test agent.")
        assert agent is not None

    def test_agent_has_run_ida_script_tool(self):
        agent = create_agent()
        tools = agent._function_toolset.tools
        assert "run_ida_script" in tools


class TestSandboxDeps:
    """SandboxDeps dataclass."""

    def test_create_with_sandbox(self):
        mock_sandbox = MagicMock()
        deps = SandboxDeps(sandbox=mock_sandbox)
        assert deps.sandbox is mock_sandbox


class TestIDAAgentInit:
    """IDAAgent initialization."""

    def test_create(self):
        mock_sandbox = MagicMock()
        ida_agent = IDAAgent(sandbox=mock_sandbox)
        assert ida_agent.agent is not None
        assert ida_agent.message_history == []

    def test_custom_model(self):
        mock_sandbox = MagicMock()
        ida_agent = IDAAgent(
            sandbox=mock_sandbox,
            model_name="google/gemma-3-4b-it:free",
        )
        assert ida_agent.agent is not None

    def test_clear_history(self):
        mock_sandbox = MagicMock()
        ida_agent = IDAAgent(sandbox=mock_sandbox)
        ida_agent._message_history = [MagicMock()]
        assert len(ida_agent.message_history) == 1
        ida_agent.clear_history()
        assert len(ida_agent.message_history) == 0


class TestOpenRouterConnectivity:
    """Verify real OpenRouter API calls work.

    These tests make actual API calls to OpenRouter using the
    OPENROUTER_API_KEY environment variable.
    """

    @pytest.fixture
    def mock_sandbox(self):
        sandbox = MagicMock()
        sandbox.execute.return_value = "test output"
        return sandbox

    @pytest.mark.asyncio
    async def test_simple_query(self, mock_sandbox):
        """Agent can send a simple query and get a response."""
        ida_agent = IDAAgent(
            sandbox=mock_sandbox,
            system_prompt="You are a helpful assistant. Respond concisely.",
        )
        response = await ida_agent.send("Say 'hello' and nothing else.")
        assert isinstance(response, AgentResponse)
        assert len(response.text) > 0
        assert "hello" in response.text.lower()

    @pytest.mark.asyncio
    async def test_conversation_history_preserved(self, mock_sandbox):
        """Message history grows across multiple sends."""
        ida_agent = IDAAgent(
            sandbox=mock_sandbox,
            system_prompt="You are a helpful assistant. Respond concisely.",
        )
        r1 = await ida_agent.send("My favorite color is blue.")
        assert len(r1.message_history) > 0

        r2 = await ida_agent.send("What is my favorite color?")
        assert len(r2.message_history) > len(r1.message_history)
        assert "blue" in r2.text.lower()

    @pytest.mark.asyncio
    async def test_tool_invocation(self, mock_sandbox):
        """Agent invokes the run_ida_script tool when asked to analyze."""
        mock_sandbox.execute.return_value = "arch: metapc\nbitness: 32"
        ida_agent = IDAAgent(
            sandbox=mock_sandbox,
            system_prompt=(
                "You analyze binaries. Use the run_ida_script tool to "
                "execute analysis code. Always use the tool when asked "
                "about the binary."
            ),
        )
        response = await ida_agent.send(
            "Use run_ida_script to call get_binary_info() and print "
            "the architecture. The code should be: "
            'info = get_binary_info()\nprint("arch: " + info["architecture"])'
        )
        assert mock_sandbox.execute.called
        assert isinstance(response.text, str)
