"""Versioned contracts for experiment inputs and linked run evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

SCHEMA_VERSION = 1
_TERMINAL = frozenset({"completed", "failed", "cancelled", "timed_out", "interrupted"})
_TRANSITIONS = {
    "pending": frozenset({"running", "failed", "cancelled", "timed_out", "interrupted"}),
    "running": _TERMINAL,
}
_COVERAGE = frozenset({"unknown", "partial", "complete"})


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _timestamp(value: object, name: str) -> datetime:
    raw = _nonempty(value, name)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware ISO 8601 timestamp")
    return parsed


def _optional_timestamp(value: object, name: str) -> datetime | None:
    return None if value is None else _timestamp(value, name)


def _configuration(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("configuration must be a JSON object with string keys")
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("configuration must contain finite JSON values") from exc
    result: dict[str, Any] = json.loads(encoded)
    return result


def _decode(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("record must be a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {value.get('schema_version')!r}")
    return value


def _optional_id(value: object, name: str) -> None:
    if value is not None:
        _nonempty(value, name)


def _id_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of IDs")
    ids = tuple(_nonempty(item, name) for item in value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


@dataclass(frozen=True)
class ExperimentSpec:
    """Stable task and configuration shared by repeated trials."""

    experiment_id: str
    task_id: str
    configuration: dict[str, Any]
    workspace_ref: str | None = None
    environment_ref: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version!r}")
        _nonempty(self.experiment_id, "experiment_id")
        _nonempty(self.task_id, "task_id")
        _optional_id(self.workspace_ref, "workspace_ref")
        _optional_id(self.environment_ref, "environment_ref")
        object.__setattr__(self, "configuration", _configuration(self.configuration))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> ExperimentSpec:
        value = _decode(raw)
        return cls(**value)


@dataclass(frozen=True)
class RunRecord:
    """One attempt within a trial; evidence lives in separate linked records."""

    experiment_id: str
    run_id: str
    trial_id: str
    attempt_id: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    parent_attempt_id: str | None = None
    retry_of_attempt_id: str | None = None
    configuration: dict[str, Any] | None = None
    workspace_ref: str | None = None
    environment_ref: str | None = None
    event_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    model_call_ids: tuple[str, ...] = ()
    evaluation_ids: tuple[str, ...] = ()
    usage_coverage: str = "unknown"
    cost_coverage: str = "unknown"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version: {self.schema_version!r}")
        for name in ("experiment_id", "run_id", "trial_id", "attempt_id"):
            _nonempty(getattr(self, name), name)
        for name in (
            "parent_attempt_id",
            "retry_of_attempt_id",
            "workspace_ref",
            "environment_ref",
        ):
            _optional_id(getattr(self, name), name)
        if self.retry_of_attempt_id == self.attempt_id or self.parent_attempt_id == self.attempt_id:
            raise ValueError("an attempt cannot link to itself")
        if self.status not in _TRANSITIONS and self.status not in _TERMINAL:
            raise ValueError(f"invalid run status: {self.status!r}")
        created = _timestamp(self.created_at, "created_at")
        started = _optional_timestamp(self.started_at, "started_at")
        finished = _optional_timestamp(self.finished_at, "finished_at")
        if started is not None and started < created:
            raise ValueError("started_at must follow created_at timestamp")
        if finished is not None and finished < (started or created):
            raise ValueError("finished_at must follow earlier timestamps")
        if self.status == "pending" and started is not None:
            raise ValueError("pending records cannot have started_at")
        if self.status == "running" and started is None:
            raise ValueError("running records require started_at")
        if self.status in _TERMINAL and finished is None:
            raise ValueError("terminal records require finished_at")
        if self.status not in _TERMINAL and finished is not None:
            raise ValueError("unfinished records cannot have finished_at")
        if self.usage_coverage not in _COVERAGE or self.cost_coverage not in _COVERAGE:
            raise ValueError("coverage must be unknown, partial, or complete")
        if self.configuration is not None:
            object.__setattr__(self, "configuration", _configuration(self.configuration))
        for name in ("event_ids", "artifact_ids", "model_call_ids", "evaluation_ids"):
            object.__setattr__(self, name, _id_tuple(getattr(self, name), name))

    def transition(self, status: str, *, at: str) -> RunRecord:
        if status not in _TRANSITIONS.get(self.status, frozenset()):
            raise ValueError(f"invalid run status transition: {self.status} -> {status}")
        if status == "running":
            return replace(self, status=status, started_at=at)
        return replace(self, status=status, finished_at=at)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, raw: str) -> RunRecord:
        value = _decode(raw)
        return cls(**value)
