"""The real HTTP client and the model router, against a local fake provider."""

import json

import pytest
from fake_llm_server import FakeProvider, chat_reply

from silent_bug_agent.semantic.client import (
    LLMAuthError,
    LLMBadResponse,
    LLMRateLimited,
    LLMUnavailable,
    OpenAICompatibleClient,
    extract_json,
)
from silent_bug_agent.semantic.router import ModelRouter, TASK_MODELS

SECRET = "gsk_test_secret_value_123"


@pytest.fixture
def provider():
    servers = []

    def make(handler):
        server = FakeProvider(handler)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.close()


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure: [1, 2] done') == [1, 2]
    with pytest.raises(LLMBadResponse):
        extract_json("no json here")


def test_complete_sends_openai_request_and_bearer_key(provider):
    server = provider(lambda body: chat_reply("hello"))
    client = OpenAICompatibleClient(server.url, SECRET, "openai/gpt-oss-120b", provider="groq",
                                    reasoning_effort="medium", extra_body={"include_reasoning": False})
    assert client.complete("hi", "be brief") == "hello"
    request = server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == f"Bearer {SECRET}"
    body = request["body"]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_completion_tokens"] > 0 and body["reasoning_effort"] == "medium"
    assert body["include_reasoning"] is False
    assert client.last.prompt_tokens == 50


def test_complete_json_uses_strict_schema_then_falls_back_to_json_mode(provider):
    def handler(body):
        if body["response_format"]["type"] == "json_schema":
            return 400, {"error": {"message": "json_schema not supported"}}, {}
        return chat_reply('{"ok": true}')

    server = provider(handler)
    client = OpenAICompatibleClient(server.url, SECRET, "m")
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
    assert client.complete_json("x", schema=schema) == {"ok": True}
    kinds = [r["body"]["response_format"]["type"] for r in server.requests]
    assert kinds == ["json_schema", "json_object"]
    assert server.requests[0]["body"]["response_format"]["json_schema"]["strict"] is True


def test_rate_limit_and_auth_errors_never_show_the_key(provider):
    limited = provider(lambda body: (429, {"error": {"message": "rate limit"}}, {"retry-after": "999"}))
    with pytest.raises(LLMRateLimited) as caught:
        OpenAICompatibleClient(limited.url, SECRET, "m", max_retries=0).complete("x")
    assert SECRET not in str(caught.value)
    refused = provider(lambda body: (401, {"error": {"message": "invalid key"}}, {}))
    client = OpenAICompatibleClient(refused.url, SECRET, "m")
    with pytest.raises(LLMAuthError) as caught:
        client.complete("x")
    assert SECRET not in str(caught.value) and SECRET not in repr(client)


def test_short_rate_limit_is_retried(provider):
    calls = []

    def handler(body):
        calls.append(1)
        return (429, {"error": {"message": "slow down"}}, {"retry-after": "0"}) if len(calls) == 1 else chat_reply("done")

    server = provider(handler)
    assert OpenAICompatibleClient(server.url, SECRET, "m").complete("x") == "done"
    assert len(calls) == 2


def test_empty_reply_is_an_error_not_an_answer(provider):
    server = provider(lambda body: chat_reply(""))
    with pytest.raises(LLMBadResponse):
        OpenAICompatibleClient(server.url, SECRET, "m").complete("x")


def test_router_is_offline_without_keys():
    router = ModelRouter()
    assert router.available() is False
    assert router.task("fix") is None
    assert "offline" in router.describe()


def test_router_falls_over_from_groq_to_cerebras(provider, monkeypatch):
    groq = provider(lambda body: (429, {"error": {"message": "daily limit"}}, {"retry-after": "999"}))
    cerebras = provider(lambda body: chat_reply('{"answer": 42}'))
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("CEREBRAS_API_KEY", "csk_other")
    monkeypatch.setenv("GROQ_BASE_URL", groq.url)
    monkeypatch.setenv("CEREBRAS_BASE_URL", cerebras.url)
    router = ModelRouter()
    assert router.task("fix").complete_json("x") == {"answer": 42}
    assert groq.requests[0]["body"]["model"] == "openai/gpt-oss-120b"
    assert cerebras.requests[0]["body"]["model"] == "gpt-oss-120b"
    summary = router.summary()
    assert summary["models_used"] == ["cerebras/gpt-oss-120b"] and summary["failed_calls"] == 1


def test_router_raises_when_every_provider_fails(provider, monkeypatch):
    groq = provider(lambda body: (500, {"error": {"message": "down"}}, {}))
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("GROQ_BASE_URL", groq.url)
    with pytest.raises(LLMUnavailable):
        ModelRouter().task("perception").complete("x")


def test_task_assignment_and_override(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    router = ModelRouter()
    assert [c.model for c in router.choices("fix")] == ["openai/gpt-oss-120b"]
    assert [c.model for c in router.choices("perception")] == ["openai/gpt-oss-20b"]
    monkeypatch.setenv("SBA_MODEL_FIX", "groq/qwen/qwen3.8-27b")
    assert router.choices("fix")[0].model == "qwen/qwen3.8-27b"
    assert set(TASK_MODELS) == {"semantic_review", "diagnosis", "fix", "meta", "perception", "validation"}


def test_dotenv_is_read_without_overriding_the_environment(tmp_path, monkeypatch):
    from silent_bug_agent import config

    env = tmp_path / ".env"
    env.write_text('# comment\nGROQ_API_KEY="from-file"\nexport CEREBRAS_API_KEY=abc # trailing\n', encoding="utf-8")
    monkeypatch.delenv("SBA_NO_DOTENV")
    monkeypatch.setenv("CEREBRAS_API_KEY", "already-set")
    config.load_env(env, force=True)
    import os

    assert os.environ["GROQ_API_KEY"] == "from-file"
    assert os.environ["CEREBRAS_API_KEY"] == "already-set"
