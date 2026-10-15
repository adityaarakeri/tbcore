"""Immutable price snapshots and auditable model-call cost reconciliation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from tbcore.experiment_store import ExperimentStore

_MILLION = Decimal(1_000_000)
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class TokenRates:
    """USD per million tokens, with independent rates for cache categories."""

    input: str
    output: str
    cache_read: str
    cache_write: str

    def __post_init__(self) -> None:
        for value in (self.input, self.output, self.cache_read, self.cache_write):
            rate = Decimal(value)
            if not rate.is_finite() or rate < 0:
                raise ValueError("token rates must be finite and nonnegative")


@dataclass(frozen=True)
class PriceCard:
    version: str
    source: str
    effective_date: str
    currency: str
    rates: Mapping[str, TokenRates]

    def __post_init__(self) -> None:
        if not self.version or not self.source or not self.effective_date:
            raise ValueError("price card provenance is required")
        if self.currency != "USD":
            raise ValueError("only USD price cards are supported")


@dataclass(frozen=True)
class CostTotals:
    call_count: int
    known_subtotal: Decimal
    unknown_calls: int

    @property
    def complete(self) -> bool:
        return self.unknown_calls == 0

    @property
    def total(self) -> Decimal | None:
        return self.known_subtotal if self.complete else None


@dataclass(frozen=True)
class RunCostTotals(CostTotals):
    attempts: Mapping[str, CostTotals]


class CostLedger:
    """Persist original quotes separately from explicitly requested recalculations."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def _calls(self, run_id: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        by_provider_id: dict[tuple[object, object], int] = {}
        for event in self.store.events(run_id):
            fields = event["fields"]
            if fields.get("event_type") != "model_call":
                continue
            provider_id = fields.get("provider_request_id")
            if provider_id:
                key = (fields.get("provider"), provider_id)
                prior_index = by_provider_id.get(key)
                if prior_index is not None:
                    prior = calls[prior_index]["fields"]
                    if any(
                        prior.get(name) is not None
                        and fields.get(name) is not None
                        and prior.get(name) != fields.get(name)
                        for name in _TOKEN_FIELDS
                    ):
                        raise ValueError("duplicate provider request has conflicting usage")
                    if sum(fields.get(name) is not None for name in _TOKEN_FIELDS) > sum(
                        prior.get(name) is not None for name in _TOKEN_FIELDS
                    ):
                        calls[prior_index] = event
                    continue
                by_provider_id[key] = len(calls)
            calls.append(event)
        return calls

    def price_run(self, run_id: str, card: PriceCard, *, recalculate: bool = False) -> None:
        existing = {event["event_id"] for event in self.store.events(run_id)}
        for call in self._calls(run_id):
            call_id = str(call["event_id"])
            event_id = f"recalc:{card.version}:{call_id}" if recalculate else f"cost:{call_id}"
            if event_id in existing:
                continue
            call_fields = call["fields"]
            model = call_fields.get("model")
            rates = card.rates.get(model) if isinstance(model, str) else None
            usage_known = all(
                type(call_fields.get(name)) is int and call_fields[name] >= 0
                for name in _TOKEN_FIELDS
            )
            estimate: Decimal | None = None
            if rates is not None and usage_known:
                estimate = (
                    sum(
                        Decimal(str(call_fields[name])) * Decimal(rate)
                        for name, rate in zip(
                            _TOKEN_FIELDS,
                            (rates.input, rates.output, rates.cache_read, rates.cache_write),
                            strict=True,
                        )
                    )
                    / _MILLION
                )
            self.store.append_event(
                run_id,
                event_id,
                {
                    "event_type": "cost_estimate",
                    "priced_event_id": call_id,
                    "attempt_id": call_fields.get("attempt_id"),
                    "status": (
                        "estimated"
                        if estimate is not None
                        else "unknown-price"
                        if rates is None
                        else "unknown-usage"
                    ),
                    "estimated_cost": str(estimate) if estimate is not None else None,
                    "rate_source": card.source,
                    "rate_version": card.version,
                    "effective_date": card.effective_date,
                    "currency": card.currency,
                    "rate_units": "USD per 1M tokens",
                    "estimation_method": "measured-tokens-times-frozen-rates",
                    "input_rate_per_mtok": rates.input if rates else None,
                    "output_rate_per_mtok": rates.output if rates else None,
                    "cache_read_rate_per_mtok": rates.cache_read if rates else None,
                    "cache_write_rate_per_mtok": rates.cache_write if rates else None,
                },
            )

    def run_totals(self, run_id: str, *, rate_version: str | None = None) -> RunCostTotals:
        calls = self._calls(run_id)
        events = self.store.events(run_id)
        prefix = "cost:" if rate_version is None else f"recalc:{rate_version}:"
        quotes = {
            event["fields"].get("priced_event_id"): event["fields"]
            for event in events
            if event["event_id"].startswith(prefix)
            and event["fields"].get("event_type") == "cost_estimate"
        }
        attempts: dict[str, list[Decimal | None]] = {}
        for call in calls:
            quote = quotes.get(call["event_id"])
            if quote is None:
                raise ValueError("call has no price observation for requested rate version")
            attempt = quote.get("attempt_id")
            if not isinstance(attempt, str):
                raise ValueError("cost observation has no attempt identity")
            value = quote.get("estimated_cost")
            attempts.setdefault(attempt, []).append(
                Decimal(value) if isinstance(value, str) else None
            )

        def aggregate(values: list[Decimal | None]) -> CostTotals:
            return CostTotals(
                len(values),
                sum((value for value in values if value is not None), Decimal(0)),
                sum(value is None for value in values),
            )

        per_attempt = {name: aggregate(values) for name, values in attempts.items()}
        combined = aggregate([value for values in attempts.values() for value in values])
        return RunCostTotals(
            combined.call_count, combined.known_subtotal, combined.unknown_calls, per_attempt
        )
