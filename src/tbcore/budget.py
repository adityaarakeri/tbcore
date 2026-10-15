"""Run-scoped admission limits with explicit in-flight and usage uncertainty."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal

from tbcore.experiment_store import ExperimentStore


class BudgetExceeded(RuntimeError):
    """A run cannot start more work under its configured limits."""


@dataclass(frozen=True)
class BudgetLimits:
    max_wall_s: float | None = None
    max_tool_calls: int | None = None
    max_model_calls: int | None = None
    max_input_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: Decimal | None = None
    max_in_flight: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_tool_calls",
            "max_model_calls",
            "max_input_tokens",
            "max_total_tokens",
            "max_in_flight",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be positive")
        if self.max_wall_s is not None and (
            not math.isfinite(self.max_wall_s) or self.max_wall_s <= 0
        ):
            raise ValueError("max_wall_s must be positive")
        if self.max_cost_usd is not None and (
            not self.max_cost_usd.is_finite() or self.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be positive and finite")


@dataclass(frozen=True)
class BudgetSnapshot:
    model_calls: int
    tool_calls: int
    in_flight: int
    input_tokens: int
    total_tokens: int
    cost_usd: Decimal
    usage_uncertain: bool
    cancelled: bool
    inflight_can_exceed_limit: bool


class RunBudget:
    """Admission is atomic; usage and spend limits remain best-effort after calls start."""

    def __init__(
        self,
        store: ExperimentStore,
        run_id: str,
        limits: BudgetLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        store.get_run(run_id)
        self.store = store
        self.run_id = run_id
        self.limits = limits
        self.clock = clock
        self.started = clock()
        self.on_cancel = on_cancel
        self._lock = threading.RLock()
        self._model_calls: set[str] = set()
        self._tool_calls: set[str] = set()
        self._in_flight: set[str] = set()
        self._input_tokens = 0
        self._total_tokens = 0
        self._cost_usd = Decimal(0)
        self._usage_uncertain = False
        self._cancelled = False

    def _deny(self, call_id: str, reason: str) -> None:
        self.store.append_event(
            self.run_id,
            f"budget:denied:{call_id}",
            {
                "event_type": "budget_denied",
                "status": "denied",
                "budget_reason": reason,
            },
        )
        raise BudgetExceeded(reason)

    def _check_locked(self, call_id: str) -> None:
        if (
            self.limits.max_wall_s is not None
            and self.clock() - self.started >= self.limits.max_wall_s
        ):
            if not self._cancelled:
                self._cancelled = True
                if self.on_cancel is not None:
                    self.on_cancel()
            self._deny(call_id, "wall-clock limit reached")
        if self._usage_uncertain and (
            self.limits.max_input_tokens is not None
            or self.limits.max_total_tokens is not None
            or self.limits.max_cost_usd is not None
        ):
            self._deny(call_id, "unknown usage under configured limit")
        if self._cancelled:
            self._deny(call_id, "run cancelled")

    def check_time(self) -> None:
        with self._lock:
            self._check_locked("time-check")

    def begin_model_call(self, call_id: str, *, priced: bool) -> None:
        with self._lock:
            self._check_locked(call_id)
            if call_id in self._model_calls:
                self._deny(call_id, "duplicate model-call identity")
            if self.limits.max_cost_usd is not None and not priced:
                self._deny(call_id, "unpriced model under spend limit")
            if (
                self.limits.max_model_calls is not None
                and len(self._model_calls) >= self.limits.max_model_calls
            ):
                self._deny(call_id, "model-call limit reached")
            if (
                self.limits.max_in_flight is not None
                and len(self._in_flight) >= self.limits.max_in_flight
            ):
                self._deny(call_id, "in-flight model-call limit reached")
            if self.limits.max_cost_usd is not None and self._cost_usd >= self.limits.max_cost_usd:
                self._deny(call_id, "cost limit reached")
            if (
                self.limits.max_input_tokens is not None
                and self._input_tokens >= self.limits.max_input_tokens
            ):
                self._deny(call_id, "input-token limit reached")
            if (
                self.limits.max_total_tokens is not None
                and self._total_tokens >= self.limits.max_total_tokens
            ):
                self._deny(call_id, "token limit reached")
            self._model_calls.add(call_id)
            self._in_flight.add(call_id)
            self.store.append_event(
                self.run_id,
                f"budget:start:{call_id}",
                {
                    "event_type": "budget_model_start",
                    "status": "in_flight",
                    "in_flight": len(self._in_flight),
                },
            )

    def finish_model_call(
        self,
        call_id: str,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        cost_usd: Decimal | None,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ) -> None:
        cancel_now = False
        with self._lock:
            if call_id not in self._in_flight:
                raise ValueError("model call is not in flight")
            if input_tokens is not None and input_tokens < 0:
                raise ValueError("token usage cannot be negative")
            if output_tokens is not None and output_tokens < 0:
                raise ValueError("token usage cannot be negative")
            if cache_read_input_tokens < 0 or cache_creation_input_tokens < 0:
                raise ValueError("cached token usage cannot be negative")
            if cost_usd is not None and (not cost_usd.is_finite() or cost_usd < 0):
                raise ValueError("cost usage cannot be negative or nonfinite")
            self._in_flight.remove(call_id)
            if (
                input_tokens is None
                or output_tokens is None
                or (self.limits.max_cost_usd is not None and cost_usd is None)
            ):
                self._usage_uncertain = True
            else:
                input_total = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
                self._input_tokens += input_total
                self._total_tokens += input_total + output_tokens
                if cost_usd is not None:
                    self._cost_usd += cost_usd
            exceeded = (
                (
                    self.limits.max_input_tokens is not None
                    and self._input_tokens >= self.limits.max_input_tokens
                )
                or (
                    self.limits.max_total_tokens is not None
                    and self._total_tokens >= self.limits.max_total_tokens
                )
                or (
                    self.limits.max_cost_usd is not None
                    and self._cost_usd >= self.limits.max_cost_usd
                )
            )
            if (
                exceeded
                or self._usage_uncertain
                and (
                    self.limits.max_input_tokens is not None
                    or self.limits.max_total_tokens is not None
                    or self.limits.max_cost_usd is not None
                )
            ):
                cancel_now = not self._cancelled
                self._cancelled = True
            self.store.append_event(
                self.run_id,
                f"budget:finish:{call_id}",
                {
                    "event_type": "budget_model_finish",
                    "status": "uncertain" if self._usage_uncertain else "measured",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": cache_read_input_tokens,
                    "cache_creation_input_tokens": cache_creation_input_tokens,
                    "budget_cost_usd": str(cost_usd) if cost_usd is not None else None,
                    "in_flight": len(self._in_flight),
                },
            )
        if cancel_now and self.on_cancel is not None:
            self.on_cancel()

    def record_tool_call(self, event_id: str) -> None:
        with self._lock:
            self._check_locked(event_id)
            if event_id in self._tool_calls:
                return
            if (
                self.limits.max_tool_calls is not None
                and len(self._tool_calls) >= self.limits.max_tool_calls
            ):
                self._deny(event_id, "tool-call limit reached")
            self._tool_calls.add(event_id)
            self.store.append_event(
                self.run_id,
                f"budget:tool:{event_id}",
                {
                    "event_type": "budget_tool_call",
                    "status": "admitted",
                },
            )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                model_calls=len(self._model_calls),
                tool_calls=len(self._tool_calls),
                in_flight=len(self._in_flight),
                input_tokens=self._input_tokens,
                total_tokens=self._total_tokens,
                cost_usd=self._cost_usd,
                usage_uncertain=self._usage_uncertain,
                cancelled=self._cancelled,
                inflight_can_exceed_limit=bool(self._in_flight)
                and any(
                    (
                        self.limits.max_input_tokens is not None,
                        self.limits.max_total_tokens is not None,
                        self.limits.max_cost_usd is not None,
                    )
                ),
            )
