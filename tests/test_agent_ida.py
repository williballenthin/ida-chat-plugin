"""Layer 2 tests: PydanticAI agent with real IDA database and codemode sandbox.

These tests verify the full stack:
- IDAAgent with a real IdaSandbox backed by a real IDA database
- The agent can use run_ida_script to query the binary
- Conversation history works with real tool invocations

Requires IDA Pro to be installed and the test binary to be available.
"""

import pytest

from ida_chat_agent import AgentResponse, IDAAgent


@pytest.mark.asyncio
class TestSandboxDirect:
    """Verify the sandbox works directly before testing via agent."""

    async def test_sandbox_execute_binary_info(self, sandbox):
        output = sandbox.execute(
            'info = get_binary_info()\n'
            'print(info["architecture"])'
        )
        assert "metapc" in output

    async def test_sandbox_execute_enumerate_functions(self, sandbox):
        output = sandbox.execute("print(len(enumerate_functions()))")
        count = int(output.strip())
        assert count >= 1

    async def test_sandbox_execute_error(self, sandbox):
        output = sandbox.execute("1 / 0")
        assert "Script error" in output


@pytest.mark.asyncio
class TestAgentWithRealIDA:
    """Full integration: PydanticAI agent + IdaSandbox + IDA database."""

    async def test_agent_queries_binary_info(self, sandbox):
        """Agent can retrieve binary metadata via tool call."""
        ida_agent = IDAAgent(
            sandbox=sandbox,
            system_prompt=(
                "You analyze binaries using the run_ida_script tool. "
                "When asked about the binary, use run_ida_script to run "
                "analysis code. Respond with the results."
            ),
        )
        response = await ida_agent.send(
            "What architecture is this binary? Use run_ida_script with this code: "
            'info = get_binary_info()\n'
            'print("arch:" + info["architecture"] + " bits:" + str(info["bitness"]))'
        )
        assert isinstance(response, AgentResponse)
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        # Verify the conversation has more than just the initial request
        # (tool calls happened, producing multiple messages)
        assert len(response.message_history) >= 2

    async def test_agent_lists_functions(self, sandbox):
        """Agent can enumerate functions in the binary."""
        ida_agent = IDAAgent(
            sandbox=sandbox,
            system_prompt=(
                "You analyze binaries using the run_ida_script tool. "
                "Always use the tool when asked about the binary."
            ),
        )
        response = await ida_agent.send(
            "How many functions are in this binary? Use run_ida_script with: "
            'fns = enumerate_functions()\nprint("count:" + str(len(fns)))\n'
            'for f in fns[:3]:\n    print(f["name"])'
        )
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    async def test_agent_conversation_memory(self, sandbox):
        """Agent maintains context across multiple messages."""
        ida_agent = IDAAgent(
            sandbox=sandbox,
            system_prompt=(
                "You analyze binaries using the run_ida_script tool. "
                "Always use the tool when asked about the binary. "
                "Remember information from previous messages."
            ),
        )
        r1 = await ida_agent.send(
            "Use run_ida_script to get the binary architecture: "
            'info = get_binary_info()\nprint(info["architecture"])'
        )
        assert len(r1.message_history) > 0

        r2 = await ida_agent.send(
            "Use run_ida_script to count the imports: "
            'imports = enumerate_imports()\nprint("imports:" + str(len(imports)))'
        )
        assert len(r2.message_history) > len(r1.message_history)
