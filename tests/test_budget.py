"""Run budgets reject new work and retain uncertainty evidence."""

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from tbcore.budget import BudgetExceeded, BudgetLimits, RunBudget
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


def test_model_and_tool_attempt_limits_count_retries_and_reject_more(tmp_path):
    store = _store(tmp_path)
    budget = RunBudget(store, "run", BudgetLimits(max_model_calls=2, max_tool_calls=1))
    budget.begin_model_call("first", priced=True)
    budget.finish_model_call("first", input_tokens=1, output_tokens=1, cost_usd=Decimal("0.01"))
    budget.begin_model_call("retry", priced=True)
    budget.finish_model_call("retry", input_tokens=1, output_tokens=1, cost_usd=Decimal("0.01"))
    with pytest.raises(BudgetExceeded, match="model-call"):
        budget.begin_model_call("third", priced=True)
    budget.record_tool_call("tool-1")
    with pytest.raises(BudgetExceeded, match="tool-call"):
        budget.record_tool_call("tool-2")
    assert budget.snapshot().model_calls == 2
    assert any(event["fields"]["event_type"] == "budget_denied" for event in store.events("run"))


def test_concurrent_inflight_cost_can_overshoot_and_is_reported(tmp_path):
    store = _store(tmp_path)
    closed = []
    budget = RunBudget(
        store,
        "run",
        BudgetLimits(max_cost_usd=Decimal("0.10"), max_model_calls=2),
        on_cancel=lambda: closed.append(True),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda call_id: budget.begin_model_call(call_id, priced=True), ("a", "b")))
    assert budget.snapshot().in_flight == 2
    assert budget.snapshot().inflight_can_exceed_limit is True
    budget.finish_model_call("a", input_tokens=5, output_tokens=1, cost_usd=Decimal("0.08"))
    budget.finish_model_call("b", input_tokens=5, output_tokens=1, cost_usd=Decimal("0.08"))
    assert budget.snapshot().cost_usd == Decimal("0.16")
    assert budget.snapshot().cancelled is True
    assert closed == [True]
    with pytest.raises(BudgetExceeded):
        budget.begin_model_call("c", priced=True)


def test_unknown_price_usage_and_wall_clock_fail_closed(tmp_path):
    store = _store(tmp_path)
    now = [0.0]
    budget = RunBudget(
        store,
        "run",
        BudgetLimits(max_wall_s=3, max_input_tokens=10, max_cost_usd=Decimal("1")),
        clock=lambda: now[0],
    )
    with pytest.raises(BudgetExceeded, match="unpriced"):
        budget.begin_model_call("unpriced", priced=False)
    budget.begin_model_call("known", priced=True)
    budget.finish_model_call("known", input_tokens=None, output_tokens=None, cost_usd=None)
    assert budget.snapshot().usage_uncertain is True
    with pytest.raises(BudgetExceeded, match="unknown"):
        budget.begin_model_call("next", priced=True)

    closed = []
    second = RunBudget(
        store,
        "run",
        BudgetLimits(max_wall_s=3),
        clock=lambda: now[0],
        on_cancel=lambda: closed.append(True),
    )
    now[0] = 4.0
    with pytest.raises(BudgetExceeded, match="wall-clock"):
        second.check_time()
    assert closed == [True]


def test_measured_token_limit_rejects_further_calls(tmp_path):
    store = _store(tmp_path)
    budget = RunBudget(store, "run", BudgetLimits(max_total_tokens=10))
    budget.begin_model_call("first", priced=False)
    budget.finish_model_call("first", input_tokens=8, output_tokens=3, cost_usd=None)
    assert budget.snapshot().total_tokens == 11
    with pytest.raises(BudgetExceeded):
        budget.begin_model_call("next", priced=False)


def test_cached_tokens_count_toward_token_threshold(tmp_path):
    store = _store(tmp_path)
    budget = RunBudget(store, "run", BudgetLimits(max_input_tokens=10))
    budget.begin_model_call("cached", priced=False)
    budget.finish_model_call(
        "cached",
        input_tokens=2,
        output_tokens=1,
        cost_usd=None,
        cache_read_input_tokens=8,
    )
    assert budget.snapshot().input_tokens == 10
    with pytest.raises(BudgetExceeded):
        budget.begin_model_call("next", priced=False)
