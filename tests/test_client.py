import json

import pytest

from tbcore.accounting import CallLog
from tbcore.client import Client, Completion, RateLimitError, TbError, TransportRequest
from tbcore.models import resolve_model

SONNET_ID = "claude-sonnet-4-5-20250929"


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "model": SONNET_ID,
            "content": [{"type": "text", "text": "hello world"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
        self.error = error
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return json.loads(json.dumps(self.response))


def make_log_path(tmp_path):
    return tmp_path / "calls.jsonl"


def make_client(tmp_path, transport):
    return Client(
        "sonnet", api_key="test-key", transport=transport, log=CallLog(make_log_path(tmp_path))
    )


def test_complete_returns_text_usage_and_latency(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport)

    completion = client.complete("be terse", [{"role": "user", "content": "hi"}])

    assert isinstance(completion, Completion)
    assert completion.text == "hello world"
    assert completion.usage.input_tokens == 11
    assert completion.usage.output_tokens == 7
    assert completion.latency_s >= 0.0
    assert len(transport.requests) == 1


def test_call_log_has_one_line_with_required_fields(tmp_path):
    transport = FakeTransport()
    log_path = make_log_path(tmp_path)
    client = Client("sonnet", api_key="test-key", transport=transport, log=CallLog(log_path))

    client.complete("sys", [{"role": "user", "content": "hi"}])

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == SONNET_ID
    assert entry["prompt_tokens"] == 11
    assert entry["completion_tokens"] == 7
    assert isinstance(entry["latency_s"], (int, float))
    assert entry["cost_equiv"] > 0.0


def test_failed_calls_are_also_logged(tmp_path):
    transport = FakeTransport(error=RateLimitError("429"))
    client = make_client(tmp_path, transport)

    with pytest.raises(RateLimitError):
        client.complete(None, [{"role": "user", "content": "hi"}])

    entries = CallLog(make_log_path(tmp_path)).entries()
    assert len(entries) == 1
    assert entries[0]["ok"] is False
    assert "error" in entries[0]


def test_cache_prefix_sent_as_cacheable_on_every_call(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport)
    messages = [{"role": "user", "content": "q"}]

    client.complete(None, messages, cache_prefix="LONG STABLE PREFIX")
    client.complete(None, messages, cache_prefix="LONG STABLE PREFIX")

    assert len(transport.requests) == 2
    for request in transport.requests:
        system_blocks = request.payload["system"]
        assert system_blocks[0]["text"] == "LONG STABLE PREFIX"
        assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_no_cache_marker_without_prefix(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport)

    client.complete("plain system", [{"role": "user", "content": "x"}])

    payload = transport.requests[0].payload
    for block in payload.get("system", []):
        assert "cache_control" not in block


def test_tools_are_passed_through(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport)
    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    client.complete(None, [{"role": "user", "content": "x"}], tools=tools)

    assert transport.requests[0].payload["tools"] == tools


def test_no_tools_key_without_tools(tmp_path):
    transport = FakeTransport()
    client = make_client(tmp_path, transport)

    client.complete(None, [{"role": "user", "content": "x"}])

    assert "tools" not in transport.requests[0].payload


def test_api_key_read_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-123")
    transport = FakeTransport()
    client = Client("sonnet", transport=transport, log=CallLog(make_log_path(tmp_path)))

    client.complete(None, [{"role": "user", "content": "x"}])

    headers = transport.requests[0].headers
    assert headers["x-api-key"] == "env-key-123"
    assert headers["anthropic-version"] == "2023-06-01"


def test_missing_api_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TbError, match="ANTHROPIC_API_KEY"):
        Client("sonnet", log=CallLog(make_log_path(tmp_path)))


def test_registry_short_name_resolves_to_model_id():
    spec = resolve_model("sonnet")
    assert spec.model_id == SONNET_ID
    assert resolve_model("some/raw-id").model_id == "some/raw-id"


def test_http_transport_maps_429_to_rate_limit_error(tmp_path, monkeypatch):
    import urllib.error

    from tbcore.client import HttpTransport

    def raise_http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, "slow down", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    transport = HttpTransport()
    request = TransportRequest(payload={"model": "x"}, headers={})

    with pytest.raises(RateLimitError):
        transport.send(request)


def test_http_transport_maps_5xx_to_transient_error(tmp_path, monkeypatch):
    import urllib.error

    from tbcore.client import HttpTransport, TransientError

    def raise_http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", raise_http_error)
    transport = HttpTransport()
    request = TransportRequest(payload={"model": "x"}, headers={})

    with pytest.raises(TransientError):
        transport.send(request)
