"""Credential-proxy usage observation from bounded JSON and SSE payloads."""

import importlib.util
import json
from decimal import Decimal

import pytest

from tbcore.sandbox import image_context

_spec = importlib.util.spec_from_file_location(
    "tbcore_proxy_usage", image_context("proxy") / "proxy.py"
)
assert _spec is not None and _spec.loader is not None
proxy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proxy)


def _feed(tap, data, sizes=(1, 7, 2, 13)):
    pos = 0
    index = 0
    while pos < len(data):
        chunk = data[pos : pos + sizes[index % len(sizes)]]
        tap.feed(chunk)
        pos += len(chunk)
        index += 1


def test_openai_responses_sse_usage_and_cache_are_split_across_chunks():
    tap = proxy.UsageTap("openai", "gpt-test", stream=True)
    event = {
        "type": "response.completed",
        "response": {
            "id": "resp-1",
            "model": "gpt-test",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 4,
                "input_tokens_details": {"cached_tokens": 6},
            },
        },
    }
    _feed(tap, b"data: " + json.dumps(event).encode() + b"\n\n")
    result = tap.finish(200, complete=True)
    assert result["provider_request_id"] == "resp-1"
    assert result["input_tokens"] == 14
    assert result["cache_read_input_tokens"] == 6
    assert result["output_tokens"] == 4
    assert result["usage_complete"] is True


def test_anthropic_stream_uses_terminal_cumulative_usage():
    tap = proxy.UsageTap("anthropic", "claude-test", stream=True)
    start = {
        "type": "message_start",
        "message": {
            "id": "msg-1",
            "model": "claude-test",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 1,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            },
        },
    }
    delta = {"type": "message_delta", "usage": {"output_tokens": 9}}
    data = b"data: " + json.dumps(start).encode() + b"\n\n"
    data += b"data: " + json.dumps(delta).encode() + b"\n\n"
    data += b'data: {"type":"message_stop"}\n\n'
    _feed(tap, data, (2, 1, 3))
    result = tap.finish(200, complete=True)
    assert result["input_tokens"] == 7
    assert result["output_tokens"] == 9
    assert result["cache_creation_input_tokens"] == 2
    assert result["usage_complete"] is True


def test_json_failed_missing_terminal_and_overflow_are_unknown():
    data = json.dumps(
        {"id": "resp-2", "model": "gpt-test", "usage": {"input_tokens": 5, "output_tokens": 2}}
    ).encode()
    tap = proxy.UsageTap("openai", "gpt-test", stream=False)
    _feed(tap, data)
    assert tap.finish(200, complete=True)["input_tokens"] == 5

    missing = proxy.UsageTap("openai", "gpt-test", stream=True)
    missing.feed(b'data: {"type":"response.output_text.delta"}\n\n')
    assert missing.finish(200, complete=True)["usage_complete"] is False

    failed = proxy.UsageTap("openai", "gpt-test", stream=False)
    failed.feed(data)
    assert failed.finish(500, complete=True)["input_tokens"] is None

    overflow = proxy.UsageTap("openai", "gpt-test", stream=False, max_bytes=8)
    overflow.feed(data)
    assert overflow.finish(200, complete=True)["usage_complete"] is False

    anthropic_missing = proxy.UsageTap("anthropic", "claude-test", stream=False)
    anthropic_missing.feed(b'{"id":"msg-2","usage":{}}')
    assert anthropic_missing.finish(200, complete=True)["usage_complete"] is False


def test_crlf_sse_and_disconnect_keep_usage_uncertain():
    tap = proxy.UsageTap("openai", "gpt-test", stream=True)
    completed = {
        "type": "response.completed",
        "response": {"id": "resp-crlf", "usage": {"input_tokens": 3, "output_tokens": 1}},
    }
    _feed(tap, b"data: " + json.dumps(completed).encode() + b"\r\n\r\n", (1, 2, 1))
    assert tap.finish(200, complete=False)["usage_complete"] is False


def test_proxy_budget_counts_concurrent_calls_and_denies_unknown_price():
    budget = proxy.ProxyBudget(
        max_model_calls=2,
        max_cost_usd=Decimal("0.10"),
        rates={"gpt-test": ("10000", "10000", "10000", "10000")},
    )
    budget.begin("a", "gpt-test")
    budget.begin("b", "gpt-test")
    assert budget.snapshot()["in_flight"] == 2
    assert budget.snapshot()["inflight_can_exceed_limit"] is True
    observation = {
        "model": "gpt-test",
        "usage_complete": True,
        "input_tokens": 4,
        "output_tokens": 4,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    budget.finish("a", observation)
    budget.finish("b", observation)
    assert budget.snapshot()["cost_usd"] == Decimal("0.16")
    with pytest.raises(proxy.ProxyBudgetExceeded):
        budget.begin("c", "gpt-test")

    spend = proxy.ProxyBudget(max_cost_usd=Decimal("1"), rates={})
    with pytest.raises(proxy.ProxyBudgetExceeded, match="unpriced"):
        spend.begin("x", "unknown")


def test_proxy_budget_config_is_validated_and_parsed():
    budget = proxy.parse_budget(
        json.dumps(
            {
                "max_model_calls": 1,
                "max_cost_usd": "0.5",
                "rates": {"gpt-test": ["1", "2", "0.5", "1.5"]},
            }
        )
    )
    assert budget is not None
    budget.begin("first", "gpt-test")
    with pytest.raises(proxy.ProxyBudgetExceeded):
        budget.begin("second", "gpt-test")
    with pytest.raises(ValueError):
        proxy.parse_budget('{"max_model_calls": -1}')
