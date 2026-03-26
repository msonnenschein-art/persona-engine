"""Model-agnostic LLM adapter supporting Anthropic and OpenAI."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Iterator


@dataclass
class Message:
    """A chat message."""
    role: str  # "user", "assistant", or "system"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: dict[str, int] | None = None
    stop_reason: str | None = None


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters."""

    @abstractmethod
    def complete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    def stream(self, messages: list[Message], system: str | None = None, **kwargs) -> Iterator[str]:
        """Stream a completion."""
        pass

    @abstractmethod
    async def acomplete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        """Async generate a completion."""
        pass

    @abstractmethod
    async def astream(self, messages: list[Message], system: str | None = None, **kwargs) -> AsyncIterator[str]:
        """Async stream a completion."""
        pass


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    @property
    def async_client(self):
        if self._async_client is None:
            import anthropic
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Anthropic format."""
        return [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]

    def complete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        response = self.client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "",
            messages=self._convert_messages(messages),
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
        )

    def stream(self, messages: list[Message], system: str | None = None, **kwargs) -> Iterator[str]:
        with self.client.messages.stream(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "",
            messages=self._convert_messages(messages),
        ) as stream:
            for text in stream.text_stream:
                yield text

    async def acomplete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        response = await self.async_client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "",
            messages=self._convert_messages(messages),
        )
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
        )

    async def astream(self, messages: list[Message], system: str | None = None, **kwargs) -> AsyncIterator[str]:
        async with self.async_client.messages.stream(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "",
            messages=self._convert_messages(messages),
        ) as stream:
            async for text in stream.text_stream:
                yield text


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI models."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._async_client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    @property
    def async_client(self):
        if self._async_client is None:
            import openai
            self._async_client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def _convert_messages(self, messages: list[Message], system: str | None = None) -> list[dict]:
        """Convert messages to OpenAI format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for m in messages:
            if m.role == "system" and system:
                continue
            result.append({"role": m.role, "content": m.content})
        return result

    def complete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=self._convert_messages(messages, system),
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            stop_reason=choice.finish_reason,
        )

    def stream(self, messages: list[Message], system: str | None = None, **kwargs) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=self._convert_messages(messages, system),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def acomplete(self, messages: list[Message], system: str | None = None, **kwargs) -> LLMResponse:
        response = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=self._convert_messages(messages, system),
        )
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            stop_reason=choice.finish_reason,
        )

    async def astream(self, messages: list[Message], system: str | None = None, **kwargs) -> AsyncIterator[str]:
        stream = await self.async_client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=self._convert_messages(messages, system),
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def create_adapter(provider: str = "anthropic", **kwargs) -> LLMAdapter:
    """Factory function to create an LLM adapter."""
    if provider.lower() == "anthropic":
        return AnthropicAdapter(**kwargs)
    elif provider.lower() == "openai":
        return OpenAIAdapter(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
