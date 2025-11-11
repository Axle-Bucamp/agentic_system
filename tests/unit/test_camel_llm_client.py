import asyncio
from types import SimpleNamespace

import pytest

from core.llm.camel_client import CamelLLMClient, CamelLLMError


class _DummyFactory:
    @staticmethod
    def create_model(**kwargs):
        return SimpleNamespace(config=kwargs)


class _DummyBaseMessage:
    @staticmethod
    def make_assistant_message(role_name: str, content: str):
        return SimpleNamespace(role_name=role_name, content=content)

    @staticmethod
    def make_user_message(role_name: str, content: str):
        return SimpleNamespace(role_name=role_name, content=content)


class _DummyResponse:
    def __init__(self, content: str):
        self.msgs = [SimpleNamespace(content=content)]


class _DummyChatAgent:
    def __init__(self, system_message, model):
        self.system_message = system_message
        self.model = model

    def step(self, user_message):
        return _DummyResponse('{"status":"ok"}')


class _EmptyChatAgent(_DummyChatAgent):
    def step(self, user_message):
        return SimpleNamespace(msgs=[])


@pytest.mark.asyncio
async def test_camel_llm_client_generate(monkeypatch):
    monkeypatch.setattr("core.llm.camel_client.CamelModelFactory", _DummyFactory)
    monkeypatch.setattr("core.llm.camel_client.BaseMessage", _DummyBaseMessage)
    monkeypatch.setattr("core.llm.camel_client.ChatAgent", _DummyChatAgent)

    client = CamelLLMClient(model_name="openrouter_llama_4_maverick_free")
    result = await client.generate("system prompt", "user prompt")
    assert result == '{"status":"ok"}'


@pytest.mark.asyncio
async def test_camel_llm_client_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr("core.llm.camel_client.CamelModelFactory", _DummyFactory)
    monkeypatch.setattr("core.llm.camel_client.BaseMessage", _DummyBaseMessage)
    monkeypatch.setattr("core.llm.camel_client.ChatAgent", _EmptyChatAgent)

    client = CamelLLMClient(model_name="openrouter_llama_4_maverick_free")
    with pytest.raises(CamelLLMError):
        await client.generate("system prompt", "user prompt", max_retries=1)


