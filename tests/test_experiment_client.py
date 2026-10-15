"""Direct Client calls retain explicit run and attempt attribution."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from tbcore.accounting import CallLog
from tbcore.budget import BudgetExceeded, BudgetLimits, RunBudget
from tbcore.client import Client, RateLimitError
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore
from tbcore.experiment_usage import CallContext


class Transport:
    def __init__(self, response=None, error=None):
        self.response = response or {
            "id": "provider-1",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
            },
        }
        self.error = error

    def send(self, request):
        if self.error:
            raise self.error
        return self.response


def _store(tmp_path):
    store = ExperimentStore(tmp_path / "store")
    for suffix in ("a", "b"):
        store.add_spec(ExperimentSpec(f"exp-{suffix}", "task", {"model": "sonnet"}))
        store.save_run(
            RunRecord(
                experiment_id=f"exp-{suffix}",
                run_id=f"run-{suffix}",
                trial_id=f"trial-{suffix}",
                attempt_id=f"attempt-{suffix}",
                status="pending",
                created_at="2026-10-14T12:00:00Z",
            )
        )
    return store


def test_concurrent_direct_calls_keep_run_and_attempt_identity(tmp_path):
    store = _store(tmp_path)
    log = CallLog(None)
    client = Client(
        "sonnet", api_key="test", transport=Transport(), log=log, experiment_store=store
    )

    def call(suffix):
        return client.complete(
            None,
            [{"role": "user", "content": suffix}],
            call_context=CallContext(f"run-{suffix}", f"attempt-{suffix}", f"call-{suffix}"),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert [result.text for result in pool.map(call, ("a", "b"))] == ["ok", "ok"]
    for suffix in ("a", "b"):
        events = store.events(f"run-{suffix}")
        assert len(events) == 1
        assert events[0]["event_id"] == f"call-{suffix}"
        fields = events[0]["fields"]
        assert fields["attempt_id"] == f"attempt-{suffix}"
        assert fields["provider_request_id"] == "provider-1"
        assert fields["input_tokens"] == 10
        assert fields["cache_read_input_tokens"] == 3
    assert len(log.entries()) == 2


def test_failure_and_duplicate_observation_do_not_count_twice(tmp_path):
    store = _store(tmp_path)
    client = Client(
        "sonnet",
        api_key="test",
        transport=Transport(error=RateLimitError("busy")),
        log=CallLog(None),
        experiment_store=store,
    )
    context = CallContext("run-a", "attempt-a", "call-failed")
    with pytest.raises(RateLimitError):
        client.complete(None, [{"role": "user", "content": "x"}], call_context=context)
    event = store.events("run-a")[0]
    assert event["fields"]["status"] == "failed"
    assert event["fields"]["input_tokens"] is None
    assert store.append_event("run-a", event["event_id"], event["fields"]) == 1
    assert len(store.events("run-a")) == 1


def test_wrong_attempt_is_rejected_before_provider_call(tmp_path):
    store = _store(tmp_path)
    transport = Transport()
    client = Client("sonnet", api_key="test", transport=transport, experiment_store=store)
    with pytest.raises(ValueError, match="attempt"):
        client.complete(
            None,
            [{"role": "user", "content": "x"}],
            call_context=CallContext("run-a", "attempt-b", "wrong"),
        )


def test_missing_usage_remains_unknown_and_cancellation_is_recorded(tmp_path):
    store = _store(tmp_path)
    response = {"id": "provider-2", "content": [{"type": "text", "text": "ok"}]}
    client = Client(
        "sonnet",
        api_key="test",
        transport=Transport(response=response),
        log=CallLog(None),
        experiment_store=store,
    )
    client.complete(
        None,
        [{"role": "user", "content": "x"}],
        call_context=CallContext("run-a", "attempt-a", "missing"),
    )
    assert store.events("run-a")[0]["fields"]["input_tokens"] is None

    cancelled = Client(
        "sonnet",
        api_key="test",
        transport=Transport(error=asyncio.CancelledError()),
        log=CallLog(None),
        experiment_store=store,
    )
    with pytest.raises(asyncio.CancelledError):
        cancelled.complete(
            None,
            [{"role": "user", "content": "x"}],
            call_context=CallContext("run-a", "attempt-a", "cancelled"),
        )
    assert store.events("run-a")[1]["fields"]["status"] == "cancelled"


def test_proxy_observation_is_attributed_and_duplicate_provider_call_is_not_counted(tmp_path):
    store = _store(tmp_path)
    observation = {
        "run_id": "run-a",
        "attempt_id": "attempt-a",
        "call_id": "proxy-1",
        "provider": "openai",
        "model": "gpt-test",
        "provider_request_id": "resp-1",
        "status": "completed",
        "usage_complete": True,
        "input_tokens": 10,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    assert store.ingest_proxy_usage(observation) == 1
    assert store.ingest_proxy_usage({**observation, "call_id": "proxy-redelivery"}) == 1
    assert len(store.events("run-a")) == 1
    assert store.events("run-a")[0]["fields"]["call_source"] == "credential_proxy"
    with pytest.raises(ValueError, match="attempt"):
        store.ingest_proxy_usage({**observation, "run_id": "run-b", "call_id": "wrong"})


def test_client_budget_rejects_second_call_before_transport(tmp_path):
    store = _store(tmp_path)
    transport = Transport()
    transport.calls = 0
    original_send = transport.send

    def count_send(request):
        transport.calls += 1
        return original_send(request)

    transport.send = count_send
    budget = RunBudget(store, "run-a", BudgetLimits(max_model_calls=1))
    client = Client(
        "sonnet",
        api_key="test",
        transport=transport,
        log=CallLog(None),
        experiment_store=store,
        budget=budget,
    )
    client.complete(
        None,
        [{"role": "user", "content": "x"}],
        call_context=CallContext("run-a", "attempt-a", "first"),
    )
    with pytest.raises(BudgetExceeded):
        client.complete(
            None,
            [{"role": "user", "content": "y"}],
            call_context=CallContext("run-a", "attempt-a", "second"),
        )
    assert transport.calls == 1
