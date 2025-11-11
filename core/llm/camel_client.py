"""
Asynchronous helper for invoking CAMEL-backed LLMs.

Wraps CAMEL's `ChatAgent` so async agent code can request completions without
depending on the synchronous API directly.  The client favours OpenRouter
models (falling back to whichever model name is supplied).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from core.logging import log
from core.models.camel_models import CamelModelFactory

try:  # pragma: no cover - exercised at runtime when CAMEL is installed
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
except ImportError as exc:  # pragma: no cover - handled in client
    ChatAgent = None  # type: ignore
    BaseMessage = None  # type: ignore
    CAMEL_IMPORT_ERROR = exc
else:
    CAMEL_IMPORT_ERROR = None


class CamelLLMError(RuntimeError):
    """Raised when the CAMEL LLM client fails to obtain a completion."""


@dataclass
class _CamelLLMConfig:
    """Lightweight configuration for CAMEL LLM invocations."""

    model_name: Optional[str]
    temperature: float = 0.3
    system_role: str = "LLM System"
    user_role: str = "User"


class CamelLLMClient:
    """Async wrapper around CAMEL ChatAgent for single-turn completions."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        system_role: str = "LLM System",
        user_role: str = "User",
        agent_factory: Optional[Callable[[object], object]] = None,
    ) -> None:
        if CAMEL_IMPORT_ERROR is not None:
            raise CamelLLMError("CAMEL is not installed") from CAMEL_IMPORT_ERROR

        self._config = _CamelLLMConfig(
            model_name=model_name,
            temperature=temperature,
            system_role=system_role,
            user_role=user_role,
        )

        try:
            self._model = CamelModelFactory.create_model(
                model_name=self._config.model_name,
                temperature=self._config.temperature,
            )
        except Exception as exc:  # pragma: no cover - depends on provider setup
            raise CamelLLMError(f"Failed to create CAMEL model: {exc}") from exc

        self._agent_factory = agent_factory or self._default_agent_factory

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_retries: int = 3,
        retry_delay: float = 0.8,
    ) -> str:
        """
        Execute a single-turn completion.

        Args:
            system_prompt: Content for the assistant/system message.
            user_prompt: Content for the user message.
            max_retries: Number of retry attempts on transient errors.
            retry_delay: Initial delay (seconds) between retries.

        Returns:
            Raw string content returned by the model.
        """
        if not system_prompt or not user_prompt:
            raise CamelLLMError("Prompts must be non-empty")

        attempt_delay = retry_delay
        last_error: Optional[BaseException] = None

        for attempt in range(1, max_retries + 1):
            try:
                return await asyncio.to_thread(
                    self._invoke,
                    system_prompt,
                    user_prompt,
                )
            except Exception as exc:  # pragma: no cover - depends on provider setup
                last_error = exc
                log.debug(
                    "CAMEL completion attempt %s/%s failed: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt == max_retries:
                    break
                await asyncio.sleep(attempt_delay)
                attempt_delay *= 1.6

        raise CamelLLMError(f"CAMEL completion failed: {last_error}") from last_error

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _default_agent_factory(self, system_message: object) -> object:
        if ChatAgent is None or BaseMessage is None:  # pragma: no cover
            raise CamelLLMError("CAMEL ChatAgent unavailable")
        return ChatAgent(system_message=system_message, model=self._model)

    def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        if ChatAgent is None or BaseMessage is None:  # pragma: no cover
            raise CamelLLMError("CAMEL ChatAgent unavailable")

        system_message = BaseMessage.make_assistant_message(
            role_name=self._config.system_role,
            content=system_prompt,
        )
        agent = self._agent_factory(system_message)

        user_message = BaseMessage.make_user_message(
            role_name=self._config.user_role,
            content=user_prompt,
        )
        response = agent.step(user_message)

        content = ""
        if response is None:
            return content

        msgs = getattr(response, "msgs", None)
        if msgs:
            final_msg = msgs[-1]
            content = getattr(final_msg, "content", "") or ""
        else:
            # Some CAMEL integrations return a dict-like response with text.
            content = getattr(response, "text", "") or getattr(response, "content", "")

        if not content:
            raise CamelLLMError("CAMEL returned empty content")
        return content


