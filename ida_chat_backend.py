"""
Chat backend abstraction layer.

Defines the ChatBackend ABC that all LLM backends must implement.
This allows swapping between Claude Agent SDK, pydantic-ai, or other
backends without changing the core or UI code.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ida_chat_core import ChatCallback
    from ida_chat_history import MessageHistory

logger = logging.getLogger("ida-chat")


class ChatBackend(ABC):
    """Abstract base class for LLM chat backends.

    Each backend implements the LLM interaction pattern (connect, send messages,
    process responses, execute scripts) while delegating UI events to a ChatCallback.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Initialize the backend connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up the backend connection."""
        ...

    @abstractmethod
    async def process_message(self, user_input: str) -> str:
        """Process a user message through the agentic loop.

        The backend should:
        1. Send the message to the LLM
        2. Execute any tool calls / scripts
        3. Feed results back to the LLM
        4. Loop until the LLM produces a final text response
        5. Fire appropriate ChatCallback events throughout

        Args:
            user_input: The user's message.

        Returns:
            Combined script outputs as a string.
        """
        ...

    @abstractmethod
    def request_cancel(self) -> None:
        """Request cancellation of the current operation."""
        ...
