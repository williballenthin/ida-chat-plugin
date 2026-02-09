"""
Pi-Mono Bridge - Python <-> Node.js agent server communication.

Manages a Node.js subprocess running the pi-mono agent and communicates
via JSON-line protocol over stdin/stdout.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator, Callable

logger = logging.getLogger("ida-chat.pi-mono")

# Path to the agent server
AGENT_SERVER_DIR = Path(__file__).parent / "agent_server"
AGENT_SERVER_SCRIPT = AGENT_SERVER_DIR / "server.mjs"


class AgentEvent:
    """Event from the pi-mono agent server."""

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
    def args(self) -> dict:
        return self.data.get("args", {})

    @property
    def turn(self) -> int:
        return self.data.get("turn", 0)

    @property
    def turns(self) -> int:
        return self.data.get("turns", 0)

    @property
    def message(self) -> str:
        return self.data.get("message", "")

    def __repr__(self) -> str:
        return f"AgentEvent({self.type}, {self.data})"


class PiMonoBridge:
    """Bridge to the pi-mono agent server running in a Node.js subprocess.

    Usage:
        bridge = PiMonoBridge()
        await bridge.start(system_prompt="...", model="openrouter/meta-llama/llama-3.3-70b-instruct:free")

        async for event in bridge.prompt("Analyze this binary"):
            if event.type == "tool_call":
                result = execute_script(event.args["code"])
                await bridge.send_tool_result(event.tool_call_id, result)
            elif event.type == "text":
                print(event.text)
            elif event.type == "done":
                break

        await bridge.stop()
    """

    def __init__(self, node_path: str | None = None):
        """Initialize the bridge.

        Args:
            node_path: Path to the Node.js binary. Auto-detected if None.
        """
        self._node_path = node_path or self._find_node()
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()
        self._running = False

    @staticmethod
    def _find_node() -> str:
        """Find the Node.js binary."""
        # Check common paths
        for path in ["/opt/node22/bin/node", "/usr/local/bin/node", "/usr/bin/node"]:
            if os.path.isfile(path):
                return path
        # Fall back to PATH
        import shutil
        node = shutil.which("node")
        if node:
            return node
        raise FileNotFoundError("Node.js not found. Install Node.js 20+ to use pi-mono.")

    async def start(
        self,
        system_prompt: str = "You are a helpful binary analysis assistant.",
        model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    ) -> None:
        """Start the agent server and initialize the agent.

        Args:
            system_prompt: System prompt for the agent.
            model: Model identifier in format "provider/model-id".
        """
        if self._running:
            raise RuntimeError("Bridge already running")

        logger.info(f"Starting agent server: {self._node_path} {AGENT_SERVER_SCRIPT}")

        self._process = await asyncio.create_subprocess_exec(
            self._node_path,
            str(AGENT_SERVER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(AGENT_SERVER_DIR),
            env={**os.environ},
        )

        self._running = True

        # Start background stderr reader
        self._stderr_task = asyncio.create_task(self._read_stderr())

        # Start background stdout reader
        self._reader_task = asyncio.create_task(self._read_stdout())

        # Send init message
        self._send({
            "type": "init",
            "systemPrompt": system_prompt,
            "model": model,
        })

        # Wait for ready signal
        event = await asyncio.wait_for(self._event_queue.get(), timeout=30.0)
        if event.type == "error":
            raise RuntimeError(f"Agent init failed: {event.message}")
        if event.type != "ready":
            raise RuntimeError(f"Expected 'ready', got '{event.type}'")

        logger.info("Agent server ready")

    def _send(self, msg: dict) -> None:
        """Send a JSON-line message to the agent server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Agent server not running")
        line = json.dumps(msg) + "\n"
        self._process.stdin.write(line.encode())

    async def _read_stdout(self) -> None:
        """Background task: read JSON lines from agent server stdout."""
        try:
            while self._running and self._process and self._process.stdout:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode().strip())
                    await self._event_queue.put(AgentEvent(data))
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from agent server: {line.decode().strip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"stdout reader error: {e}")

    async def _read_stderr(self) -> None:
        """Background task: read stderr from agent server for logging."""
        try:
            while self._running and self._process and self._process.stderr:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug(f"agent-server: {line.decode().strip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"stderr reader error: {e}")

    async def prompt(self, message: str) -> AsyncIterator[AgentEvent]:
        """Send a prompt and yield events until the agent is done.

        This is an async generator. When a tool_call event is yielded,
        the caller must call send_tool_result() before continuing iteration.

        Args:
            message: User message to send to the agent.

        Yields:
            AgentEvent objects (thinking, text, text_delta, tool_call, turn_start, turn_end, done, error)
        """
        if not self._running:
            raise RuntimeError("Bridge not running. Call start() first.")

        # Drain any stale events
        while not self._event_queue.empty():
            self._event_queue.get_nowait()

        self._send({"type": "prompt", "message": message})

        while True:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                yield AgentEvent({"type": "error", "message": "Agent timeout (120s)"})
                return

            yield event

            if event.type in ("done", "error"):
                return

    async def send_tool_result(
        self,
        tool_call_id: str,
        content: str,
        is_error: bool = False,
    ) -> None:
        """Send a tool execution result back to the agent.

        Args:
            tool_call_id: The tool call ID from the tool_call event.
            content: The execution result (stdout output or error message).
            is_error: Whether the execution failed.
        """
        self._send({
            "type": "tool_result",
            "toolCallId": tool_call_id,
            "content": content,
            "isError": is_error,
        })

    async def stop(self) -> None:
        """Stop the agent server."""
        self._running = False

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass

        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass

        self._process = None
        logger.info("Agent server stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._process is not None and self._process.returncode is None
