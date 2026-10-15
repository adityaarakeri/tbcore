import json

import pytest

from tbcore.accounting import CallLog
from tbcore.client import Client, Usage
from tbcore.models import REGISTRY, ModelSpec, make_model_spec, resolve_model


class OkTransport:
    def __init__(self, response):
        self.response = response

    def send(self, request):
        return json.loads(json.dumps(self.response))


ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "hi"}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}
OPENAI_OK = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}


@pytest.mark.parametrize(
    ("short", "model_id", "inp", "out", "read_mult"),
    [
        ("opus", "claude-opus-5-5", 4.0, 20.0, 0.05),
        ("sonnet", "claude-sonnet-5-5", 2.0, 10.0, 0.05),
        ("haiku", "claude-haiku-5-5", 0.10, 0.50, 0.1),
        ("fable", "claude-fable-5-1", 10.0, 50.0, 0.025),
    ],
)
def test_claude_5_registry_entries(short, model_id, inp, out, read_mult):
    spec = REGISTRY[short]

    assert spec.model_id == model_id
    assert spec.input_per_mtok == inp
    assert spec.output_per_mtok == out
    assert spec.cache_read_multiplier == read_mult
    assert spec.cache_write_multiplier == 1.25
    assert spec.priced


def test_cost_uses_cache_multipliers():
    spec = REGISTRY["sonnet"]
    usage = Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
    )

    # 2.0 * (1 + 1.25 + 0.05) + 10.0
    assert spec.cost(usage) == pytest.approx(14.6)


def test_haiku_long_prompt_tier():
    spec = REGISTRY["haiku"]
    # Exactly at the threshold stays on the base tier: 0.10 in, 0.50 out.
    base = Usage(input_tokens=100_000, output_tokens=1_000_000)
    assert spec.cost(base) == pytest.approx(0.01 + 0.50)
    # Cache reads count toward prompt length, pushing this over to 0.50 in / 2.50 out.
    long = Usage(input_tokens=60_000, output_tokens=1_000_000, cache_read_input_tokens=60_000)
    assert spec.cost(long) == pytest.approx((60_000 + 60_000 * 0.1) * 0.50 / 1e6 + 2.50)


def test_openai_cached_tokens_are_not_double_counted(tmp_path):
    response = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 400_000},
        },
    }
    # gpt-4o-mini: 600k uncached at 0.15 + 400k cached at 0.15 * 0.5.
    assert _logged_cost(tmp_path, REGISTRY["gpt-4o-mini"], response) == pytest.approx(
        0.6 * 0.15 + 0.4 * 0.075
    )


def test_priced_property():
    assert not ModelSpec("x", "x").priced
    assert ModelSpec("x", "x", input_per_mtok=1.0).priced
    assert not make_model_spec("m", provider="ollama").priced
    assert not make_model_spec("m", provider="vllm").priced
    assert not resolve_model("some/raw-id").priced
    assert resolve_model("gpt-4o-mini").priced


def _logged_cost(tmp_path, spec, response):
    log = CallLog(tmp_path / "calls.jsonl")
    client = Client(spec, api_key="k", transport=OkTransport(response), log=log)
    client.complete(None, [{"role": "user", "content": "hi"}])
    return log.entries()[0]["cost_equiv"]


def test_priced_spec_logs_numeric_cost(tmp_path):
    assert _logged_cost(tmp_path, "sonnet", ANTHROPIC_OK) > 0.0


@pytest.mark.parametrize(
    ("spec", "response"),
    [
        (resolve_model("claude-unknown-model"), ANTHROPIC_OK),
        (make_model_spec("qwen3:8b", provider="ollama"), OPENAI_OK),
        (make_model_spec("org/model", provider="vllm"), OPENAI_OK),
    ],
)
def test_unpriced_spec_logs_null_cost(tmp_path, spec, response):
    assert _logged_cost(tmp_path, spec, response) is None


def test_failed_call_logs_null_cost(tmp_path):
    class Boom:
        def send(self, request):
            raise RuntimeError("down")

    log = CallLog(tmp_path / "calls.jsonl")
    client = Client("sonnet", api_key="k", transport=Boom(), log=log)

    with pytest.raises(RuntimeError):
        client.complete(None, [{"role": "user", "content": "hi"}])

    assert log.entries()[0]["cost_equiv"] is None
