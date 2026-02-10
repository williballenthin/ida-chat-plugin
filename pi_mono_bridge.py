"""
Pi-Mono Bridge - Embedded JS agent harness via PythonMonkey.

Loads the agent harness JS directly into SpiderMonkey (in-process).
HTTP requests are proxied through Python's urllib. Tool execution
is dispatched via a Python callback. No subprocess, no Node.js required.
"""

import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Callable

import pythonmonkey as pm

logger = logging.getLogger("ida-chat.pi-mono")

# Path to the agent harness JS
AGENT_HARNESS = Path(__file__).parent / "agent_server" / "agent_harness.mjs"


class AgentEvent:
    """Event from the agent harness."""

    def __init__(self, data: dict):
        self.type: str = data.get("type", "")
        self.data = data

    @property
    def text(self) -> str:
        return self.data.get("text", "")

    @property
    def tool_call_id(self) -> str:
        return self.data.get("toolCallId", "")

    @property
    def tool_name(self) -> str:
        return self.data.get("toolName", "")

    @property
    def code(self) -> str:
        return self.data.get("code", "")

    @property
    def output(self) -> str:
        return self.data.get("output", "")

    @property
    def turn(self) -> int:
        return int(self.data.get("turn", 0))

    @property
    def turns(self) -> int:
        return int(self.data.get("turns", 0))

    @property
    def message(self) -> str:
        return self.data.get("message", "")

    def __repr__(self) -> str:
        return f"AgentEvent({self.type}, {self.data})"


def _python_fetch(url: str, method: str, headers_json: str, body: str) -> str:
    """HTTP fetch implementation using Python's urllib.

    Called by the JS harness for LLM API requests.
    Returns a JSON string with the response body.
    """
    headers = json.loads(headers_json)
    req = urllib.request.Request(
        url,
        data=body.encode("utf-8") if body else None,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        logger.error(f"HTTP {e.code} from {url}: {error_body[:200]}")
        try:
            # Try to return the error as JSON (OpenAI-format error)
            return error_body
        except Exception:
            return json.dumps({"error": {"message": f"HTTP {e.code}: {error_body[:200]}"}})
    except Exception as e:
        logger.error(f"Fetch error: {e}")
        return json.dumps({"error": {"message": str(e)}})


class PiMonoBridge:
    """Embedded JS agent harness via PythonMonkey.

    The agent runs in-process in SpiderMonkey. No subprocess, no Node.js.

    Usage:
        bridge = PiMonoBridge()
        bridge.start(
            system_prompt="...",
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
            tool_callback=sandbox.execute,
        )
        events = bridge.prompt("Analyze this binary")
        for event in events:
            print(event.type, event.data)
        bridge.stop()
    """

    def __init__(self):
        self._harness = None
        self._running = False

    def start(
        self,
        system_prompt: str = "You are a helpful binary analysis assistant.",
        model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        tool_callback: Callable[[str], str] | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the embedded agent.

        Args:
            system_prompt: System prompt for the agent.
            model: Model identifier in "provider/model-id" format.
            tool_callback: Python function (code: str) -> result: str.
                Called when the LLM invokes the ida_script tool.
            api_key: API key for the provider. If None, reads from
                environment (e.g. OPENROUTER_API_KEY).
        """
        if self._running:
            raise RuntimeError("Bridge already running")

        logger.info(f"Loading agent harness from {AGENT_HARNESS}")
        self._harness = pm.require(str(AGENT_HARNESS))

        # Resolve API key
        if api_key is None:
            provider = model.split("/")[0] if "/" in model else "openrouter"
            providers = dict(self._harness.PROVIDERS)
            env_key = providers.get(provider, {}).get("envKey", "OPENROUTER_API_KEY")
            if isinstance(env_key, str):
                api_key = os.environ.get(env_key, "")
            else:
                api_key = ""
            if not api_key:
                # Try generic fallback
                api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # Default tool callback is a no-op
        if tool_callback is None:
            tool_callback = lambda code: "(no executor configured)"

        self._harness.init(
            system_prompt,
            model,
            tool_callback,
            _python_fetch,
            api_key,
        )

        self._running = True
        logger.info("Agent harness loaded and initialized")

    def prompt(self, message: str, max_turns: int = 20) -> list[AgentEvent]:
        """Send a prompt and run the agent loop to completion.

        The entire agent loop (LLM calls, tool execution) happens
        synchronously in-process. Returns all events at once.

        Args:
            message: User message.
            max_turns: Maximum agent turns.

        Returns:
            List of AgentEvent objects.
        """
        if not self._running or not self._harness:
            raise RuntimeError("Bridge not running. Call start() first.")

        logger.info(f"Prompt: {message[:100]}...")

        result = self._harness.prompt(message, max_turns)

        events = []
        if result and result.get("events"):
            for raw_event in result["events"]:
                events.append(AgentEvent(dict(raw_event)))

        error = result.get("error") if result else None
        if error:
            logger.error(f"Agent error: {error}")
            if not any(e.type == "error" for e in events):
                events.append(AgentEvent({"type": "error", "message": str(error)}))

        return events

    def reset(self) -> None:
        """Reset the conversation history."""
        if self._harness:
            self._harness.reset()

    def stop(self) -> None:
        """Shut down the agent harness."""
        self._running = False
        self._harness = None
        logger.info("Agent harness stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._harness is not None
