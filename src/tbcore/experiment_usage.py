"""Explicit identity for model calls attached to experiment attempts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CallContext:
    run_id: str
    attempt_id: str
    call_id: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.run_id, self.attempt_id, self.call_id)
        ):
            raise ValueError("run_id, attempt_id, and call_id must be nonempty strings")
