"""IDA Chat Agent - PydanticAI-based chat agent with codemode sandbox.

Replaces the Claude Agent SDK-based core with a provider-agnostic agent
powered by PydanticAI. Supports any OpenAI-compatible backend via OpenRouter
or direct provider connections.

The agent has a single tool: ``run_ida_script`` which executes analysis
scripts inside the codemode sandbox (Monty interpreter) rather than raw
``exec()``.

Usage::

    from ida_domain import Database
    from ida_codemode_sandbox import IdaSandbox
    from ida_chat_agent import IDAAgent

    with Database.open(path, options) as db:
        sandbox = IdaSandbox(db)
        agent = IDAAgent(sandbox=sandbox)
        response = await agent.send("List all functions in this binary")
        print(response.text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel

from ida_codemode_sandbox import IdaSandbox

logger = logging.getLogger("ida-chat")


@dataclass
class SandboxDeps:
    """Dependencies injected into every agent tool call."""

    sandbox: IdaSandbox


def _build_system_prompt(sandbox_prompt: str) -> str:
    """Build the system prompt for the IDA analysis agent."""
    return (
        "You are an expert binary reverse-engineering assistant.\n"
        "You have access to an IDA Pro database through analysis functions.\n"
        "Use the run_ida_script tool to execute analysis scripts and answer "
        "the user's questions about the binary.\n"
        "Always explain your findings clearly.\n\n"
        + sandbox_prompt
    )


def create_agent(
    model_name: str = "meta-llama/llama-4-scout",
    *,
    provider: str = "openrouter",
    base_url: str | None = None,
    api_key: str | None = None,
    system_prompt: str | None = None,
) -> Agent[SandboxDeps, str]:
    """Create a PydanticAI agent configured for IDA analysis.

    Args:
        model_name: The model identifier (e.g. "meta-llama/llama-4-scout").
        provider: Provider name for OpenAI-compatible APIs.
            Use "openrouter" for OpenRouter (default).
        base_url: Override the provider's base URL.
        api_key: Override the API key (defaults to env var).
        system_prompt: Override the system prompt. If None, uses the
            codemode sandbox's built-in prompt.

    Returns:
        A configured PydanticAI Agent.
    """
    model_kwargs: dict[str, Any] = {"provider": provider}
    if base_url is not None:
        model_kwargs["base_url"] = base_url
    if api_key is not None:
        model_kwargs["api_key"] = api_key

    model = OpenAIChatModel(model_name, **model_kwargs)

    if system_prompt is None:
        system_prompt = _build_system_prompt(IdaSandbox.system_prompt())

    agent: Agent[SandboxDeps, str] = Agent(
        model,
        system_prompt=system_prompt,
        deps_type=SandboxDeps,
    )

    @agent.tool
    async def run_ida_script(ctx: RunContext[SandboxDeps], code: str) -> str:
        """Execute an IDA analysis script in the sandbox.

        The script runs inside a sandboxed Python interpreter with access
        to 28 IDA analysis functions (enumerate_functions, disassemble_function,
        get_binary_info, etc.). Use print() to output results.

        Args:
            code: Python source code to execute. The sandbox supports core
                Python (variables, if/for/while, def, class, list/dict
                comprehensions, f-strings) but NOT imports, file I/O, or
                network access.

        Returns:
            The script's stdout output on success, or an error description
            on failure.
        """
        logger.info(f"Executing sandbox script ({len(code)} chars)")
        result = ctx.deps.sandbox.execute(code)
        logger.info(f"Script result ({len(result)} chars): {result[:200]}")
        return result

    return agent


@dataclass
class AgentResponse:
    """Response from an IDA agent interaction."""

    text: str
    """The agent's final text response."""

    message_history: list[ModelMessage]
    """Full conversation history after this interaction."""


class IDAAgent:
    """High-level wrapper around a PydanticAI agent for IDA analysis.

    Manages conversation state and provides a simple send/receive interface.

    Usage::

        agent = IDAAgent(sandbox=sandbox)
        r1 = await agent.send("What architecture is this binary?")
        print(r1.text)
        r2 = await agent.send("List the first 5 functions")
        print(r2.text)  # agent remembers context from r1
    """

    def __init__(
        self,
        sandbox: IdaSandbox,
        *,
        model_name: str = "meta-llama/llama-4-scout",
        provider: str = "openrouter",
        base_url: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self._sandbox = sandbox
        self._agent = create_agent(
            model_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        self._message_history: list[ModelMessage] = []
        self._deps = SandboxDeps(sandbox=sandbox)

    @property
    def agent(self) -> Agent[SandboxDeps, str]:
        """The underlying PydanticAI agent."""
        return self._agent

    @property
    def message_history(self) -> list[ModelMessage]:
        """Current conversation history."""
        return list(self._message_history)

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._message_history = []

    async def send(self, message: str) -> AgentResponse:
        """Send a message and get the agent's response.

        Maintains conversation history across calls so the agent
        has context from previous interactions.

        Args:
            message: The user's message/query.

        Returns:
            AgentResponse with the text and updated history.
        """
        result = await self._agent.run(
            message,
            deps=self._deps,
            message_history=self._message_history,
        )
        self._message_history = result.all_messages()
        return AgentResponse(
            text=result.output,
            message_history=self._message_history,
        )
