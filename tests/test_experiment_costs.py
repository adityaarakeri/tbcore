"""Immutable per-call price evidence and known/unknown run totals."""

from decimal import Decimal

from tbcore.costs import CostLedger, PriceCard, TokenRates
from tbcore.experiment import ExperimentSpec, RunRecord
from tbcore.experiment_store import ExperimentStore


def _store(tmp_path):
    store = ExperimentStore(tmp_path / "store")
    store.add_spec(ExperimentSpec("exp", "task", {"model": "fixture"}))
    store.save_run(
        RunRecord(
            experiment_id="exp",
            run_id="run",
            trial_id="trial",
            attempt_id="attempt",
            status="pending",
            created_at="2026-10-14T12:00:00Z",
        )
    )
    return store


def _call(
    store,
    call_id,
    model,
    *,
    input_tokens=10,
    output_tokens=2,
    cache_read=3,
    cache_write=0,
    provider_id=None,
):
    store.append_event(
        "run",
        call_id,
        {
            "event_type": "model_call",
            "call_source": "fixture",
            "attempt_id": "attempt",
            "provider": "test",
            "model": model,
            "provider_request_id": provider_id,
            "status": "completed",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        },
    )


def test_known_subtotal_unknown_calls_and_price_snapshot(tmp_path):
    store = _store(tmp_path)
    _call(store, "call-1", "priced", cache_write=4)
    _call(store, "call-2", "priced", input_tokens=None)
    _call(store, "call-3", "unpriced")
    card = PriceCard(
        version="v1",
        source="fixture-rates",
        effective_date="2026-10-01",
        currency="USD",
        rates={"priced": TokenRates("2", "8", "0.2", "2.5")},
    )
    ledger = CostLedger(store)
    ledger.price_run("run", card)
    totals = ledger.run_totals("run")
    expected = (
        Decimal(10) * 2 + Decimal(2) * 8 + Decimal(3) * Decimal("0.2") + Decimal(4) * Decimal("2.5")
    ) / Decimal(1_000_000)
    assert totals.known_subtotal == expected
    assert totals.unknown_calls == 2
    assert totals.complete is False
    assert totals.total is None
    assert totals.attempts["attempt"].known_subtotal == expected
    quote = next(
        event["fields"] for event in store.events("run") if event["event_id"] == "cost:call-1"
    )
    assert quote["rate_version"] == "v1"
    assert quote["currency"] == "USD"
    assert quote["input_rate_per_mtok"] == "2"

    changed = PriceCard(
        version="v2",
        source="fixture-rates",
        effective_date="2026-11-01",
        currency="USD",
        rates={"priced": TokenRates("10", "10", "1", "1")},
    )
    ledger.price_run("run", changed, recalculate=True)
    assert ledger.run_totals("run").known_subtotal == expected
    assert ledger.run_totals("run", rate_version="v2").known_subtotal != expected


def test_duplicate_provider_observation_is_counted_once(tmp_path):
    store = _store(tmp_path)
    _call(store, "direct", "priced", provider_id="provider-1")
    _call(store, "proxy", "priced", provider_id="provider-1")
    card = PriceCard(
        "v1", "fixture", "2026-10-01", "USD", {"priced": TokenRates("1", "1", "1", "1")}
    )
    ledger = CostLedger(store)
    ledger.price_run("run", card)
    totals = ledger.run_totals("run")
    assert totals.call_count == 1
    assert totals.unknown_calls == 0


def test_duplicate_observation_prefers_complete_usage(tmp_path):
    store = _store(tmp_path)
    _call(
        store,
        "agent",
        "priced",
        input_tokens=None,
        output_tokens=None,
        cache_read=None,
        cache_write=None,
        provider_id="provider-1",
    )
    _call(store, "proxy", "priced", provider_id="provider-1")
    card = PriceCard(
        "v1", "fixture", "2026-10-01", "USD", {"priced": TokenRates("1", "1", "1", "1")}
    )
    ledger = CostLedger(store)
    ledger.price_run("run", card)
    totals = ledger.run_totals("run")
    assert totals.call_count == 1
    assert totals.unknown_calls == 0
