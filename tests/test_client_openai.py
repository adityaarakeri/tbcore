"""OpenAI-compatible wire: Ollama, LM Studio, vLLM, OpenAI itself, etc."""

import json

import pytest

from tbcore.accounting import CallLog
from tbcore.client import (
    APIStatusError,
    Client,
    RateLimitError,
    TbError,
    TransientError,
)
from tbcore.models import ModelSpec, make_model_spec


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "model": "qwen3:8b",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello local"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5},
        }
        self.error = error
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return json.loads(json.dumps(self.response))


def ollama_client(tmp_path, transport):
    return Client(
        make_model_spec("qwen3:8b", provider="ollama"),
        transport=transport,
        log=CallLog(tmp_path / "calls.jsonl"),
    )


def test_make_model_spec_ollama_defaults():
    spec = make_model_spec("qwen3:8b", provider="ollama")
    assert isinstance(spec, ModelSpec)
    assert spec.wire == "openai-compat"
    assert spec.base_url == "http://localhost:11434"
    assert spec.api_key_required is False
    assert spec.model_id == "qwen3:8b"


def test_make_model_spec_anthropic_resolves_registry():
    spec = make_model_spec("sonnet")
    assert spec.wire == "anthropic"
    assert spec.api_key_required is True


def test_make_model_spec_rejects_unknown_provider():
    with pytest.raises(ValueError, match="provider"):
        make_model_spec("x", provider="smoketest")


def test_openai_chat_payload_shape(tmp_path):
    transport = FakeTransport()
    client = ollama_client(tmp_path, transport)

    completion = client.complete(
        system="be terse",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.3,
        max_tokens=64,
    )

    assert completion.text == "hello local"
    assert completion.usage.input_tokens == 13
    assert completion.usage.output_tokens == 5
    payload = transport.requests[0].payload
    assert payload["model"] == "qwen3:8b"
    # System prompt travels as the first chat message on this wire.
    assert payload["messages"][0] == {"role": "system", "content": "be terse"}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 64
    # Anthropic-specific fields never appear.
    assert "system" not in payload
    assert "tools" not in payload


def test_no_temperature_key_when_unset(tmp_path):
    transport = FakeTransport()
    client = ollama_client(tmp_path, transport)

    client.complete(None, [{"role": "user", "content": "hi"}])

    assert "temperature" not in transport.requests[0].payload


def test_ollama_needs_no_api_key(tmp_path):
    import os

    old = os.environ.pop("OLLAMA_API_KEY", None)
    try:
        client = Client(make_model_spec("qwen3:8b", provider="ollama"), log=CallLog())
        assert client.api_key is None
    finally:
        if old is not None:
            os.environ["OLLAMA_API_KEY"] = old


def test_bearer_header_sent_when_key_present(tmp_path):
    transport = FakeTransport()
    client = Client(
        make_model_spec("llama3", provider="openai-compat", base_url="http://localhost:8080"),
        api_key="sk-local",
        transport=transport,
        log=CallLog(tmp_path / "calls.jsonl"),
    )

    client.complete(None, [{"role": "user", "content": "hi"}])

    headers = transport.requests[0].headers
    assert headers["Authorization"] == "Bearer sk-local"


def test_cache_prefix_unsupported_on_openai_wire(tmp_path):
    with pytest.raises(TbError, match="prompt-prefix caching"):
        ollama_client(tmp_path, FakeTransport()).complete(
            None, [{"role": "user", "content": "hi"}], cache_prefix="PREFIX"
        )


def test_tools_unsupported_on_openai_wire(tmp_path):
    with pytest.raises(TbError, match="tools"):
        ollama_client(tmp_path, FakeTransport()).complete(
            None, [{"role": "user", "content": "hi"}], tools=[{"name": "x"}]
        )


def test_call_log_records_provider(tmp_path):
    log_path = tmp_path / "calls.jsonl"
    client = ollama_client(tmp_path, FakeTransport())

    client.complete(None, [{"role": "user", "content": "hi"}])

    entry = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert entry["provider"] == "ollama"
    assert entry["model"] == "qwen3:8b"
    assert entry["ok"] is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimitError), (500, TransientError), (401, APIStatusError)],
)
def test_openai_transport_error_mapping(tmp_path, monkeypatch, status, expected):
    import urllib.error

    from tbcore.client import OpenAICompatTransport

    def raise_http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, status, "boom", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    transport = OpenAICompatTransport(base_url="http://localhost:11434")

    with pytest.raises(expected):
        transport.send(type("R", (), {"payload": {}, "headers": {}})())


def test_openai_transport_url_composition(monkeypatch):
    import urllib.request

    from tbcore.client import OpenAICompatTransport

    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        body = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        import io

        return io.BytesIO(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # Without /v1 suffix -> /v1/chat/completions is appended.
    OpenAICompatTransport(base_url="http://localhost:11434").send(
        type("R", (), {"payload": {"a": 1}, "headers": {}})()
    )
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"

    # With /v1 suffix -> only /chat/completions is appended.
    OpenAICompatTransport(base_url="http://localhost:11434/v1").send(
        type("R", (), {"payload": {"a": 1}, "headers": {}})()
    )
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
